# Pipeline

The `Pretok` class is the central orchestration component of pretok.

## Overview

The pipeline coordinates:

1. **Segment Parsing** - Parse prompt into segments (text, code, markers)
2. **Language Detection** - Detect language of text segments
3. **Translation Decision** - Determine if translation is needed
4. **Translation** - Translate non-optimal language segments
5. **Reconstruction** - Rebuild prompt with translated content

## Creating a Pretok Instance

### Basic Creation

```python
from pretok import Pretok

# With explicit target language
pretok = Pretok(target_language="en")
```

### With Model ID

```python
from pretok import create_pretok

# Auto-detect optimal language from model capabilities
pretok = create_pretok(model_id="gpt-4")  # Uses English
pretok = create_pretok(model_id="qwen-7b")  # Uses Chinese
```

### With Custom Components

```python
from pretok import Pretok
from pretok.detection.langdetect_backend import LangDetectDetector
from pretok.translation.llm import LLMTranslator
from pretok.config import LLMTranslatorConfig

detector = LangDetectDetector()
translator = LLMTranslator(LLMTranslatorConfig(
    base_url="https://api.openai.com/v1",
    model="gpt-4o-mini",
))

pretok = Pretok(
    target_language="en",
    detector=detector,
    translator=translator,
)
```

### From Configuration File

```python
from pretok import Pretok
from pretok.config import load_config

config = load_config("pretok.yaml")
pretok = Pretok(config=config)
```

## Processing Text

### Basic Processing

```python
result = pretok.process("Bonjour le monde")

print(result.processed_text)  # Translated text
print(result.was_modified)    # True if text was changed
print(result.original_text)   # Original input
```

### Detection Only

```python
detection = pretok.detect("Hello world")
print(f"Language: {detection.language}")
print(f"Confidence: {detection.confidence}")
```

### Translation Only

```python
translation = pretok.translate(
    text="Bonjour",
    target_language="en",
    source_language="fr",
)
print(translation.translated_text)
```

## Pipeline Result

The `PipelineResult` contains detailed processing information:

```python
result = pretok.process(text)

# Basic results
result.original_text      # Original input
result.processed_text     # Output text
result.was_modified       # Whether text changed

# Detailed information
result.segments           # List of parsed segments
result.detections        # Language detection results
result.translations      # Translation results
result.from_cache        # Whether result was cached
result.metadata          # Additional metadata
```

## Segment Preservation

pretok preserves prompt structure:

```python
prompt = """<|im_start|>system
You are a helpful assistant.
<|im_end|>
<|im_start|>user
Translate this French text.
<|im_end|>"""

result = pretok.process(prompt)
# Role markers preserved, only content translated
```

### Preserved Elements

- **Role Markers**: `<|im_start|>`, `[INST]`, etc.
- **Control Tokens**: `<|endoftext|>`, `<s>`, `</s>`
- **Code Blocks**: Content inside ``` markers
- **Delimiters**: `###`, `---`, etc.

## Caching

Enable caching to avoid redundant translations:

```python
from pretok import Pretok
from pretok.pipeline.cache import MemoryCache

cache = MemoryCache(max_size=1000, ttl=3600)
pretok = Pretok(target_language="en", cache=cache)

# First call - translates
result1 = pretok.process("Bonjour")

# Second call - uses cache
result2 = pretok.process("Bonjour")
print(result2.from_cache)  # True
```

## Error Handling

```python
from pretok.detection import DetectionError
from pretok.translation import TranslationError

try:
    result = pretok.process(text)
except DetectionError as e:
    print(f"Detection failed: {e}")
except TranslationError as e:
    print(f"Translation failed: {e}")
```
