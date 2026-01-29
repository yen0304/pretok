# Segment Processing

pretok intelligently parses prompts to preserve structure during translation.

## Segment Types

| Type | Description | Translatable |
|------|-------------|--------------|
| `CONTENT` | Regular text content | Yes |
| `ROLE_MARKER` | Role indicators (system, user, assistant) | No |
| `CONTROL_TOKEN` | Special tokens like `<\|endoftext\|>` | No |
| `DELIMITER` | Structural separators | No |
| `CODE_BLOCK` | Fenced code blocks | No |
| `JSON_SCHEMA` | JSON/structured data | No |
| `WHITESPACE` | Whitespace-only segments | No |

## Supported Formats

### ChatML

```
<|im_start|>system
You are a helpful assistant.
<|im_end|>
<|im_start|>user
Hello!
<|im_end|>
```

### Llama

```
[INST] <<SYS>>
You are a helpful assistant.
<</SYS>>

Hello! [/INST]
```

### Alpaca

```
### Instruction:
Translate this text.

### Input:
Bonjour

### Response:
```

## Custom Markers

Register custom markers for non-standard formats:

```python
from pretok.segment import SegmentProcessor

processor = SegmentProcessor()
processor.register_role_marker(r"\[ROLE:(\w+)\]")
processor.register_control_token(r"<\|custom\|>")
```

## Code Block Handling

Code blocks are preserved by default:

```python
prompt = '''Explain this code:

```python
def hello():
    print("Hello")
```
'''

result = pipeline.adapt(prompt, model_id="llama-2-7b")
# Code block is preserved, only surrounding text is translated
```

## JSON Handling

JSON structures are preserved:

```python
prompt = '''Parse this JSON:

{"name": "John", "city": "Paris"}
'''

result = pipeline.adapt(prompt, model_id="llama-2-7b")
# JSON is preserved exactly
```
