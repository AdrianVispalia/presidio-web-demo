# Local censorer

1. Install python 3.11

- Windows
> Download the installer here: https://www.python.org/ftp/python/3.11.9/python-3.11.9-amd64.exe

- Linux
```bash
sudo apt update
sudo apt install -y software-properties-common
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt update
sudo apt install -y python3.11
```

- macOS
```bash
brew install python@3.11
```


2. Install uv

- Windows Powershell
```Powershell
irm https://astral.sh/uv/install.ps1 | iex
```

- Linux / macOS
```bash
curl -Ls https://astral.sh/uv/install.sh | sh
```

3. Install the dependencies
```bash
uv sync
```

<details>

To install with GPU acceleration:
```bash
uv sync --extra cuda
```

</details>

4. Download a model
```bash
uv run python -m spacy download en_core_web_lg
```

5. Run the project
```bash
uv run streamlit run presidio_streamlit.py
```
