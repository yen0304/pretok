# Installation

## Requirements

- Python 3.11 or higher
- pip or uv package manager

## Basic Installation

Install the core package:

```bash
pip install pretok
```

Or using uv:

```bash
uv add pretok
```

## Optional Dependencies

pretok supports various backends for language detection and translation. Install the ones you need:

### Language Detection

```bash
# FastText (recommended for accuracy)
pip install pretok[fasttext]

# langdetect (pure Python, no external dependencies)
pip install pretok[langdetect]

# All detection backends
pip install pretok[detection]
```

### Translation Backends

#### Local Models (requires GPU for best performance)

```bash
# NLLB (Meta's No Language Left Behind)
pip install pretok[nllb]

# M2M100 (Facebook's multilingual model)
pip install pretok[m2m100]

# All local models
pip install pretok[local]
```

#### API-based Translation

```bash
# OpenAI
pip install pretok[openai]

# Google Cloud Translation
pip install pretok[google]

# DeepL
pip install pretok[deepl]

# All API backends
pip install pretok[api]
```

### Caching

```bash
# Redis for distributed caching
pip install pretok[redis]
```

### Everything

```bash
pip install pretok[all]
```

## Development Installation

For contributing to pretok:

```bash
git clone https://github.com/yen0304/pretok.git
cd pretok
uv sync --dev
```

## Verifying Installation

```python
import pretok
print(pretok.__version__)
```
