# Configuration

pretok uses a hierarchical configuration system with sensible defaults.

## Configuration Hierarchy

1. **Built-in defaults** - Sensible out-of-box behavior
2. **Configuration file** - `pretok.yaml` or `pretok.toml`
3. **Runtime overrides** - Programmatic configuration

## Configuration File

pretok automatically searches for configuration files in this order:

1. `pretok.yaml`
2. `pretok.yml`
3. `pretok.toml`
4. `pretok.json`
5. `.pretok.yaml`

### Full Configuration Example

```yaml
version: "1.0"

# Pipeline settings
pipeline:
  default_detector: langdetect
  cache_enabled: true

# Language detection configuration
detection:
  fasttext:
    model_path: /path/to/lid.176.bin
  langdetect:
    seed: 42
  composite:
    detectors: [fasttext, langdetect]
    strategy: voting

# Translation configuration
translation:
  # LLM-based translation (recommended)
  llm:
    base_url: "https://api.openai.com/v1"
    model: "gpt-4o-mini"
    api_key_env: OPENAI_API_KEY
    temperature: 0.3
    max_tokens: 1024
    system_prompt: "You are a professional translator."
    user_prompt_template: "Translate from {source_lang} to {target_lang}:\n{text}"
  
  # HuggingFace models for local translation
  huggingface:
    model: facebook/nllb-200-distilled-600M
    device: cuda
    max_length: 512
    num_beams: 4

# Cache configuration
cache:
  memory:
    max_size: 1000
    ttl: 3600

# Model capabilities
models:
  default:
    supported_languages: [en]
    primary_language: en
  
  gpt-4:
    supported_languages: [en, zh, ja, ko, fr, de, es, it, pt, ru]
    primary_language: en
  
  qwen-7b:
    supported_languages: [zh, en]
    primary_language: zh
  
  llama-2-7b:
    supported_languages: [en]
    primary_language: en

# Segment processing
segment:
  preserve_code_blocks: true
  translate_code_comments: false
  translate_json_strings: false
```

## Environment Variables

Sensitive values can reference environment variables:

```yaml
translation:
  llm:
    api_key_env: OPENAI_API_KEY  # Reads from $OPENAI_API_KEY
```

Or inline with `${VAR}` syntax:

```yaml
cache:
  redis:
    url: ${REDIS_URL}
```

## Programmatic Configuration

```python
from pretok import Pipeline
from pretok.config import PretokConfig, PipelineConfig, TranslationConfig

config = PretokConfig(
    pipeline=PipelineConfig(
        default_detector="langdetect",
        default_translator="openai",
        confidence_threshold=0.8,
    ),
    translation=TranslationConfig(
        openai={"model": "gpt-4"},
    ),
)

pipeline = Pipeline(config=config)
```

## Configuration Validation

Configuration is validated on load:

```python
from pretok.config import load_config

try:
    config = load_config("pretok.yaml")
except ConfigurationError as e:
    print(f"Invalid configuration: {e}")
```
