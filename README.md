<p align="center">
  <img src="https://raw.githubusercontent.com/yen0304/pretok/main/logo.png" alt="pretok logo" width="640">
</p>

<h1 align="center">pretok</h1>

<p align="center">
  <a href="https://github.com/yen0304/pretok/actions/workflows/ci.yml"><img src="https://github.com/yen0304/pretok/actions/workflows/ci.yml/badge.svg?branch=main" alt="CI"></a>
  <a href="https://codecov.io/gh/yen0304/pretok"><img src="https://codecov.io/gh/yen0304/pretok/branch/main/graph/badge.svg" alt="codecov"></a>
  <a href="https://pypi.org/project/pretok/"><img src="https://img.shields.io/pypi/v/pretok.svg" alt="PyPI version"></a>
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.11+-blue.svg" alt="Python 3.11+"></a>
  <a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License: MIT"></a>
  <a href="https://github.com/astral-sh/ruff"><img src="https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json" alt="Ruff"></a>
</p>

> Universal pre-token language adaptation layer for text-based LLMs.

**pretok** enables any Large Language Model to receive input in any human language by automatically translating input text into a language the model supports—all before tokenization, without modifying the model or tokenizer.

## ✨ Features

- **Model-Agnostic**: Works with any text-based LLM (local, remote, open-source, proprietary)
- **Pre-Token Boundary**: All transformations occur on raw text before tokenization
- **Prompt Structure Preservation**: Role markers, delimiters, code blocks, and control tokens are preserved
- **Flexible Translation**: Use any LLM via OpenAI-compatible APIs (OpenRouter, Ollama, vLLM, etc.)
- **Pluggable Backends**: Support for multiple detection and translation engines
- **Explicit Capability Contracts**: Models declare their supported languages

## 🚀 Installation

```bash
pip install pretok
```

Or with uv:

```bash
uv add pretok
```

### Optional Dependencies

```bash
# Language detection
pip install pretok[fasttext]      # FastText (high accuracy)
pip install pretok[langdetect]    # langdetect (pure Python)

# Translation backends
pip install pretok[nllb]          # Meta's NLLB model (local)
pip install pretok[openai]        # OpenAI API

# All features
pip install pretok[all]
```

## 📖 Quick Start

```python
from pretok import Pretok, create_pretok

# Create with default settings
pretok = Pretok(target_language="en")

# Process text
result = pretok.process("Bonjour, comment ca va?")

print(result.processed_text)  # "Hello, how are you?"
print(result.was_modified)    # True
```

### With Model-Specific Optimization

```python
# Auto-detect optimal language from model capabilities
pretok = create_pretok(model_id="gpt-4")     # Uses English
pretok = create_pretok(model_id="qwen-7b")   # Uses Chinese
```

### With Custom Translation Backend

```python
from pretok import Pretok
from pretok.config import LLMTranslatorConfig
from pretok.translation.llm import LLMTranslator

# Use any OpenAI-compatible API
config = LLMTranslatorConfig(
    base_url="https://api.openai.com/v1",  # Or OpenRouter, Ollama, vLLM
    model="gpt-4o-mini",
)
translator = LLMTranslator(config)
pretok = Pretok(target_language="en", translator=translator)
```

### Preserving Prompt Structure

```python
prompt = """<|im_start|>system
You are a helpful assistant.
<|im_end|>
<|im_start|>user
What is the capital of Japan?
<|im_end|>"""

result = pretok.process(prompt)
# Role markers preserved, only content translated
```

### Configuration

Create a `pretok.yaml`:

```yaml
version: "1.0"

pipeline:
  default_detector: langdetect
  cache_enabled: true

translation:
  llm:
    base_url: "https://api.openai.com/v1"
    model: "gpt-4o-mini"

cache:
  memory:
    max_size: 1000
    ttl: 3600
```

```python
from pretok import Pretok
from pretok.config import load_config

config = load_config("pretok.yaml")
pretok = Pretok(config=config)
```

## 🏗️ Architecture

```
Input Text (any language)
        ↓
Segment Parsing (roles, code, text)
        ↓
Language Detection
        ↓
Translation Decision
        ↓
Translation (if needed)
        ↓
Prompt Reconstruction
        ↓
Tokenizer (unchanged)
        ↓
LLM Inference
```

## 📚 Documentation

- [Installation Guide](docs/getting-started/installation.md)
- [Quickstart Tutorial](docs/getting-started/quickstart.md)
- [Configuration Reference](docs/getting-started/configuration.md)
- [API Documentation](docs/api/pipeline.md)

## 🛠️ Development

```bash
# Clone the repository
git clone https://github.com/yen0304/pretok.git
cd pretok

# Install dependencies
uv sync --dev

# Run tests
uv run pytest

# Run linting
uv run ruff check src/ tests/

# Run type checking
uv run mypy src/
```

## 📄 License

MIT License - see [LICENSE](LICENSE) for details.

## 🤝 Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](docs/development/contributing.md) for guidelines.
