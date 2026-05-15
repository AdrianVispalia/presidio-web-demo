"""Streamlit app for Presidio — Stacked Full-Width Clean Version."""

import logging
import os
import traceback
from io import BytesIO

import dotenv
import pandas as pd
import pdfplumber
import streamlit as st
import streamlit.components.v1 as components
from annotated_text import annotated_text
from docx import Document
from PIL import Image
from streamlit_tags import st_tags

from openai_fake_data_generator import OpenAIParams
from presidio_helpers import (
    analyze,
    analyzer_engine,
    annotate,
    anonymize,
    create_fake_data,
    get_supported_entities,
)

try:
    from presidio_image_redactor import ImageRedactorEngine

    IMAGE_REDACTOR_AVAILABLE = True
except ImportError:
    IMAGE_REDACTOR_AVAILABLE = False

st.set_page_config(
    page_title="Presidio PII Workspace",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={"About": "https://microsoft.github.io/presidio/"},
)

# Custom minimal CSS styling injection (Reverted theme colors to native Streamlit defaults)
st.markdown(
    """
    <style>
        /* Tighten main view margins and padding significantly */
        .main .block-container {
            padding-top: 0.5rem !important;
        }

        div[data-testid="stAppViewContainer"] {
            padding: 0rem !important;
        }

        /* Consistent typewriter fonts for standard input views */
        textarea {
            font-family: 'SF Mono', Menlo, Monaco, Consolas, monospace !important;
            font-size: 0.9rem !important;
        }

        /* Highlight box matching standard text areas without forcing color palettes */
        .highlight-output-container {
            border: 1px solid rgba(49, 51, 63, 0.2);
            border-radius: 8px;
            padding: 1rem;
            min-height: 300px;
            overflow-y: auto;
        }

        .dark .highlight-output-container {
            border: 1px solid rgba(255, 255, 255, 0.2);
        }

        div[data-testid="stMetricValue"] {
            font-size: 1.8rem !important;
            font-weight: 600;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

dotenv.load_dotenv()
logger = logging.getLogger("presidio-streamlit")
allow_other_models = os.getenv("ALLOW_OTHER_MODELS", False)

# ── Session State Tracking ────────────────────────────────────────────────────
if "live_text_canvas" not in st.session_state:
    st.session_state["live_text_canvas"] = ""
if "last_uploaded_file" not in st.session_state:
    st.session_state["last_uploaded_file"] = None

# ═════════════════════════════════════════════════════════════════════════════
#  SIDEBAR
# ═════════════════════════════════════════════════════════════════════════════
st.sidebar.title("Options")
st.sidebar.markdown("---")

st_device = st.sidebar.radio(
    "Compute Device",
    options=["CPU", "GPU (if available)"],
    index=0,
    help="Select GPU to let spaCy / Transformers use CUDA if a compatible GPU is present.",
)
use_gpu = st_device == "GPU (if available)"
if use_gpu:
    try:
        import spacy

        spacy.prefer_gpu()
        st.sidebar.success("⚡ GPU mode requested and active.")
    except Exception:
        st.sidebar.warning("⚠️ GPU unavailable; falling back to CPU.")

model_list = [
    "spaCy/en_core_web_lg",
    "flair/ner-english-large",
    "HuggingFace/obi/deid_roberta_i2b2",
    "HuggingFace/StanfordAIMI/stanford-deidentifier-base",
    "stanza/en",
    "Azure AI Language",
    "Other",
]
if not allow_other_models:
    model_list.pop()

st_ta_key = st_ta_endpoint = ""

st_model = st.sidebar.selectbox(
    "NER Model Engine",
    model_list,
    index=2,
    help="Select which Named Entity Recognition model package to use for PII detection.",
)

st_model_package = st_model.split("/")[0]
st_model = (
    st_model
    if st_model_package.lower() not in ("spacy", "stanza", "huggingface")
    else "/".join(st_model.split("/")[1:])
)

if st_model == "Other":
    st_model_package = st.sidebar.selectbox(
        "NER Model OSS Package", options=["spaCy", "stanza", "Flair", "HuggingFace"]
    )
    st_model = st.sidebar.text_input("Model Identifier", value="")

if st_model == "Azure AI Language":
    st_ta_key = st.sidebar.text_input(
        "Azure Key", value=os.getenv("TA_KEY", ""), type="password"
    )
    st_ta_endpoint = st.sidebar.text_input(
        "Azure Endpoint",
        value=os.getenv("TA_ENDPOINT", default=""),
        help="Target URL for Azure AI Language PII Services.",
    )

st_operator = st.sidebar.selectbox(
    "Transformation Strategy",
    ["redact", "replace", "synthesize", "highlight", "mask", "hash", "encrypt"],
    index=1,
    help=(
        "- **Redact**: Remove text entirely\n"
        "- **Replace**: Swap with a categorical label\n"
        "- **Synthesize**: Replace with AI-generated fake data\n"
        "- **Highlight**: Visual color-coded inspection layer\n"
        "- **Mask**: Filter characters out with wildcards\n"
        "- **Hash**: Render cryptographic hash strings\n"
        "- **Encrypt**: Symmetric AES encryption"
    ),
)

st_mask_char = "*"
st_number_of_chars = 15
st_encrypt_key = "WmZq4t7w!z%C&F)J"
open_ai_params = None


def set_up_openai_synthesis():
    if os.getenv("OPENAI_TYPE", default="openai") == "Azure":
        openai_api_type = "azure"
        st_openai_api_base = st.sidebar.text_input(
            "Azure OpenAI Base URL",
            value=os.getenv("AZURE_OPENAI_ENDPOINT", default=""),
        )
        openai_key = os.getenv("AZURE_OPENAI_KEY", default="")
        st_deployment_id = st.sidebar.text_input(
            "Deployment Name", value=os.getenv("AZURE_OPENAI_DEPLOYMENT", default="")
        )
        st_openai_version = st.sidebar.text_input(
            "API Version", value=os.getenv("OPENAI_API_VERSION", default="2023-05-15")
        )
    else:
        openai_api_type = "openai"
        st_openai_version = st_openai_api_base = None
        st_deployment_id = ""
        openai_key = os.getenv("OPENAI_KEY", default="")

    st_openai_key = st.sidebar.text_input(
        "OpenAI API Key", value=openai_key, type="password"
    )
    st_openai_model = st.sidebar.text_input(
        "Synthesis Model Target",
        value=os.getenv("OPENAI_MODEL", default="text-davinci-003"),
    )
    return (
        openai_api_type,
        st_openai_api_base,
        st_deployment_id,
        st_openai_version,
        st_openai_key,
        st_openai_model,
    )


if st_operator == "mask":
    st_number_of_chars = st.sidebar.number_input(
        "Mask Length", value=st_number_of_chars, min_value=0, max_value=100
    )
    st_mask_char = st.sidebar.text_input(
        "Mask character", value=st_mask_char, max_chars=1
    )
elif st_operator == "encrypt":
    st_encrypt_key = st.sidebar.text_input(
        "AES Cryptographic Key", value=st_encrypt_key
    )
elif st_operator == "synthesize":
    (
        openai_api_type,
        st_openai_api_base,
        st_deployment_id,
        st_openai_version,
        st_openai_key,
        st_openai_model,
    ) = set_up_openai_synthesis()
    open_ai_params = OpenAIParams(
        openai_key=st_openai_key,
        model=st_openai_model,
        api_base=st_openai_api_base,
        deployment_id=st_deployment_id,
        api_version=st_openai_version,
        api_type=openai_api_type,
    )

st_threshold = st.sidebar.slider(
    "Confidence Cutoff Threshold",
    min_value=0.0,
    max_value=1.0,
    value=0.35,
    help="Minimum confidence score required to flag a detected entity.",
)

st_return_decision_process = st.sidebar.checkbox(
    "Append Explainable Analytics",
    value=False,
    help="Expose internal engine reasoning flags in final structural tables.",
)

with st.sidebar.expander("Dictionary Exclusions & Restrictions", expanded=False):
    st_allow_list = st_tags(
        label="Allowlist Override (Ignore Words)", text="Press enter to add target"
    )
    st_deny_list = st_tags(
        label="Denylist Enforcements (Always Redact)", text="Press enter to add target"
    )

analyzer_params = (st_model_package, st_model, st_ta_key, st_ta_endpoint)


# ═════════════════════════════════════════════════════════════════════════════
#  FILE EXTRACTION HELPERS
# ═════════════════════════════════════════════════════════════════════════════
def extract_text_from_pdf(uploaded_file) -> str:
    with pdfplumber.open(BytesIO(uploaded_file.getvalue())) as pdf:
        return "\n".join(
            [page.extract_text() for page in pdf.pages if page.extract_text()]
        )


def extract_text_from_docx(uploaded_file) -> str:
    doc = Document(BytesIO(uploaded_file.getvalue()))
    return "\n".join([p.text for p in doc.paragraphs if p.text.strip()])


def extract_text_from_image(uploaded_file) -> tuple[str, Image.Image]:
    image = Image.open(BytesIO(uploaded_file.getvalue())).convert("RGB")
    if IMAGE_REDACTOR_AVAILABLE:
        from presidio_image_redactor.tesseract_ocr import TesseractOCR

        ocr = TesseractOCR()
        return ocr.get_text_from_ocr_dict(ocr.perform_ocr(image)), image
    return "", image


def is_image_type(file_type: str) -> bool:
    return file_type in [
        "image/png",
        "image/jpeg",
        "image/jpg",
        "image/tiff",
        "image/bmp",
    ]


# ═════════════════════════════════════════════════════════════════════════════
#  ANALYZER INIT
# ═════════════════════════════════════════════════════════════════════════════
try:
    analyzer = analyzer_engine(*analyzer_params)
except Exception as exc:
    st.error(f"Failed to start backend compliance engines: {exc}")
    st.stop()

st_entities_expander = st.sidebar.expander("Target Entity Scopes")
st_entities = st_entities_expander.multiselect(
    label="Active Detection Handlers",
    options=get_supported_entities(*analyzer_params),
    default=list(get_supported_entities(*analyzer_params)),
)

# ═════════════════════════════════════════════════════════════════════════════
#  MAIN INTERFACE — Full width layout hierarchy
# ═════════════════════════════════════════════════════════════════════════════
st.title("🛡️ Presidio")

# ── INPUT SECTION (Full Width) ────────────────────────────────────────────────
with st.container(border=True):
    st.markdown("### 📥 Input")

    accepted_types = ["txt", "pdf", "docx", "png", "jpg", "jpeg", "tiff", "bmp"]
    uploaded_file = st.file_uploader(
        "Upload a file (TXT, PDF, DOCX, or image)",
        type=accepted_types,
        label_visibility="collapsed",
    )

    pil_image = None

    if (
        uploaded_file is not None
        and uploaded_file != st.session_state["last_uploaded_file"]
    ):
        st.session_state["last_uploaded_file"] = uploaded_file
        file_type = uploaded_file.type

        with st.spinner("Extracting text…"):
            if file_type == "application/pdf":
                extracted = extract_text_from_pdf(uploaded_file)
            elif file_type in [
                "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                "application/msword",
            ]:
                extracted = extract_text_from_docx(uploaded_file)
            elif file_type == "text/plain":
                extracted = uploaded_file.getvalue().decode("utf-8")
            elif is_image_type(file_type):
                extracted, pil_image = extract_text_from_image(uploaded_file)
            else:
                extracted = ""

        st.session_state["live_text_canvas"] = extracted

    if not st.session_state["live_text_canvas"]:
        try:
            with open("demo_text.txt") as f:
                st.session_state["live_text_canvas"] = f.read()
        except FileNotFoundError:
            st.session_state["live_text_canvas"] = (
                "Enter text containing sensitive PII elements here…"
            )

    st_text = st.text_area(
        "Live Text Editor",
        height=300,
        key="live_text_canvas",
        label_visibility="collapsed",
        help="Edit directly, or upload a file above to populate this canvas.",
    )

    if uploaded_file and is_image_type(uploaded_file.type):
        if pil_image is None:
            pil_image = Image.open(BytesIO(uploaded_file.getvalue()))
        st.image(pil_image, caption="Uploaded image", use_container_width=True)

# ── OUTPUT SECTION (Full Width underneath Input Section) ──────────────────────
with st.container(border=True):
    # Splits the title line: 5 parts title space, 1 part button space
    header_col1, header_col2 = st.columns([5, 1])
    with header_col1:
        st.markdown(f"### 📤 Output ({st_operator.upper()})")
    btn_canvas = header_col2.empty()  # Dynamic hook right next to the title

    run_status = st.empty()
    output_canvas = st.empty()
    output_text_payload = None

    if st_text.strip():
        run_status.markdown("⏳ *Analyzing…*")

        st_analyze_results = analyze(
            *analyzer_params,
            text=st_text,
            entities=st_entities,
            language="en",
            score_threshold=st_threshold,
            return_decision_process=st_return_decision_process,
            allow_list=st_allow_list,
            deny_list=st_deny_list,
        )

        if st_operator not in ("highlight", "synthesize"):
            st_anonymize_results = anonymize(
                text=st_text,
                operator=st_operator,
                mask_char=st_mask_char,
                number_of_chars=st_number_of_chars,
                encrypt_key=st_encrypt_key,
                analyze_results=st_analyze_results,
            )
            output_canvas.text_area(
                label="Output",
                value=st_anonymize_results.text,
                height=300,
                label_visibility="collapsed",
            )
            output_text_payload = st_anonymize_results.text

        elif st_operator == "synthesize":
            fake_data = create_fake_data(st_text, st_analyze_results, open_ai_params)
            output_canvas.text_area(
                label="Synthetic Output",
                value=fake_data,
                height=300,
                label_visibility="collapsed",
            )
            output_text_payload = fake_data

        else:  # highlight
            annotated_tokens = annotate(
                text=st_text, analyze_results=st_analyze_results
            )
            with output_canvas:
                st.markdown(
                    '<div class="highlight-output-container">',
                    unsafe_allow_html=True,
                )
                annotated_text(*annotated_tokens)
                st.markdown("</div>", unsafe_allow_html=True)

        run_status.empty()
        # Safely render the copy button up in the header row if text payload exists
        if output_text_payload:
            safe_payload = output_text_payload.replace("`", "\\`").replace("$", "\\$")
            with btn_canvas:
                components.html(
                    f"""
                    <body style="margin: 0; padding: 0; background: transparent; display: flex; justify-content: flex-end; align-items: center;">
                        <script>
                        function copyToClipboard() {{
                            navigator.clipboard.writeText(`{safe_payload}`);
                            const btn = document.getElementById("copyBtn");
                            btn.innerText = "✓ Copied";
                            btn.style.backgroundColor = "#24a0ed";
                            btn.style.borderColor = "#24a0ed";
                            btn.style.color = "#ffffff";
                            setTimeout(() => {{
                                btn.innerText = "📋 Copy";
                                btn.style.backgroundColor = "#4B5563";
                                btn.style.borderColor = "#374151";
                                btn.style.color = "#ffffff";
                            }}, 1500);
                        }}
                        </script>
                        <button id="copyBtn" onclick="copyToClipboard()" style="
                            background-color: #4B5563;
                            color: #ffffff;
                            border: 1px solid #374151;
                            border-radius: 6px;
                            padding: 0.35rem 0.7rem;
                            font-size: 0.85rem;
                            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                            font-weight: 500;
                            cursor: pointer;
                            transition: all 0.2s ease;
                            box-shadow: 0 1px 2px rgba(0,0,0,0.05);
                        ">
                            📋 Copy
                        </button>
                    </body>
                    """,
                    height=40,
                )
    else:
        run_status.info("Output will appear here after analysis.")
        output_canvas.text_area(
            "Output Preview",
            value="",
            height=300,
            disabled=True,
            label_visibility="collapsed",
        )

# ═════════════════════════════════════════════════════════════════════════════
#  FINDINGS TABLE & EXPORT
# ═════════════════════════════════════════════════════════════════════════════
if st_text.strip() and "st_analyze_results" in locals():
    st.markdown("---")
    st.markdown("### 📊 Detected data")

    if st_analyze_results:
        df = pd.DataFrame.from_records([r.to_dict() for r in st_analyze_results])
        df["text"] = [st_text[r.start : r.end] for r in st_analyze_results]
        df_subset = df[["entity_type", "text", "start", "end", "score"]].rename(
            columns={
                "entity_type": "Risk Classification",
                "text": "Extracted String Context",
                "start": "Index Start",
                "end": "Index End",
                "score": "Engine Certainty Confidence",
            }
        )
        if st_return_decision_process:
            analysis_explanation_df = pd.DataFrame.from_records(
                [r.analysis_explanation.to_dict() for r in st_analyze_results]
            )
            df_subset = pd.concat([df_subset, analysis_explanation_df], axis=1)

        st.dataframe(df_subset.reset_index(drop=True), use_container_width=True)

        if output_text_payload:
            st.download_button(
                label="📦 Download output text",
                data=output_text_payload,
                file_name="sanitized_compliance_stream.txt",
                mime="text/plain",
            )
    else:
        st.info("No PII detected above the selected confidence threshold.")

# ── Clarity Analytics ─────────────────────────────────────────────────────────
components.html(
    """
    <script type="text/javascript">
    (function(c,l,a,r,i,t,y){
        c[a]=c[a]||function(){(c[a].q=c[a].q||[]).push(arguments)};
        t=l.createElement(r);t.async=1;t.src="https://www.clarity.ms/tag/"+i;
        y=l.getElementsByTagName(r)[0];y.parentNode.insertBefore(t,y);
    })(window, document, "clarity", "script", "h7f8bp42n8");
    </script>
    """
)
