# Quickstart

This guide will help you get started with pretok in just a few minutes.

## Basic Usage

### Creating a Pretok Instance

```python
from pretok import Pretok, create_pretok

# Create with explicit target language
pretok = Pretok(target_language="en")

# Or create with model-based configuration
pretok = create_pretok(model_id="gpt-4")  # Uses GPT-4's primary language
```

### Processing Text

```python
# Simple processing
result = pretok.process("Hola, como estas?")

print(f"Output: {result.processed_text}")
print(f"Was modified: {result.was_modified}")
print(f"Detections: {result.detections}")
```

### Working with Language Detection

```python
# Detect language only (no translation)
detection = pretok.detect("Bonjour le monde")
print(f"Language: {detection.language}")
print(f"Confidence: {detection.confidence}")
```

## Working with Prompts

pretok preserves prompt structure during translation:

```python
prompt = """<|im_start|>system
You are a helpful assistant.
<|im_end|>
<|im_start|>user
Ecrivez un poeme sur Python.
<|im_end|>"""

result = pretok.process(prompt)
# Only the content is translated, markers are preserved
```

## Using Custom Translation Backends

### OpenAI API

```python
from pretok import Pretok
from pretok.config import LLMTranslatorConfig
from pretok.translation.llm import LLMTranslator

config = LLMTranslatorConfig(
    base_url="https://api.openai.com/v1",
    model="gpt-4o-mini",
)
translator = LLMTranslator(config)
pretok = Pretok(target_language="en", translator=translator)
```

### OpenRouter

```python
config = LLMTranslatorConfig(
    base_url="https://openrouter.ai/api/v1",
    model="anthropic/claude-3-haiku",
    # api_key will use OPENAI_API_KEY or OPENROUTER_API_KEY env var
)
```

### Local Ollama

```python
config = LLMTranslatorConfig(
    base_url="http://localhost:11434/v1",
    model="llama3",
    api_key="ollama",  # Ollama doesn't require a real key
)
```

## Configuration

### Using YAML

Create a `pretok.yaml` file:

```yaml
version: "1.0"

pipeline:
  default_detector: langdetect
  cache_enabled: true

translation:
  llm:
    base_url: "https://api.openai.com/v1"
    model: "gpt-4o-mini"
    # api_key_env: OPENAI_API_KEY  # Optional, defaults to OPENAI_API_KEY

cache:
  memory:
    max_size: 1000
    ttl: 3600
```

### Loading Configuration

```python
from pretok.config import load_config

config = load_config("pretok.yaml")
pretok = Pretok(config=config)
```

## Next Steps

- [Configuration Guide](configuration.md) - Learn about all configuration options
- [Pipeline Guide](../user-guide/pipeline.md) - Deep dive into pipeline usage
- [Translation Backends](../user-guide/translation.md) - Configure translation engines
