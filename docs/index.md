# pretok

> Universal pre-token language adaptation layer for text-based LLMs.

**pretok** enables any Large Language Model to receive input in any human language by automatically translating input text into a language the model supports—all before tokenization, without modifying the model or tokenizer.

## Features

- **Model-Agnostic**: Works with any text-based LLM (local, remote, open-source, proprietary)
- **Pre-Token Boundary**: All transformations occur on raw text before tokenization
- **Prompt Structure Preservation**: Role markers, delimiters, code blocks, and control tokens are preserved
- **Pluggable Backends**: Support for multiple detection and translation engines
- **Flexible Translation**: Use any LLM via OpenAI-compatible APIs (OpenRouter, Ollama, vLLM, etc.)
- **Explicit Capability Contracts**: Models declare their supported languages

## Installation

```bash
pip install pretok
```

## Quick Start

```python
from pretok import Pretok, create_pretok

# Create with default settings (targets English)
pretok = Pretok(target_language="en")

# Process text
result = pretok.process("Bonjour, comment ça va?")

print(result.processed_text)  # "Hello, how are you?"
print(result.was_modified)    # True
```

### With Model-Specific Optimization

```python
# Auto-detect optimal language from model capabilities
pretok = create_pretok(model_id="gpt-4")

result = pretok.process("Hello World")
# Uses GPT-4's primary language (English)
```

### With Custom Translation Backend

```python
from pretok import Pretok
from pretok.config import LLMTranslatorConfig
from pretok.translation.llm import LLMTranslator

# Use any OpenAI-compatible API
config = LLMTranslatorConfig(
    base_url="https://api.openai.com/v1",  # Or OpenRouter, Ollama, vLLM, etc.
    model="gpt-4o-mini",
)
translator = LLMTranslator(config)
pretok = Pretok(target_language="en", translator=translator)
```

## Documentation

- [Installation](getting-started/installation.md)
- [Quick Start Guide](getting-started/quickstart.md)
- [Configuration](getting-started/configuration.md)
- [User Guide](user-guide/pipeline.md)
- [API Reference](api/pipeline.md)
