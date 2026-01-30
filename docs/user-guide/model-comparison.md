# Translation Model Comparison

This document records the performance of different Ollama models in pretok translation tasks.

> **Test Date**: 2026-01-30  
> **pretok Version**: 0.1.1

## Tested Models

| Model | Size | Description |
|-------|------|-------------|
| gemma3:latest | 3.3 GB | Google Gemma 3 general-purpose model |
| qwen3:latest | 5.2 GB | Alibaba Qwen 3 general-purpose model |
| translategemma:4b | 3.3 GB | Google dedicated translation model |
| gpt-oss:20b | 13 GB | GPT-OSS 20B general-purpose model |

## Test Results

### Basic Translation Tests

| Test Case | Original (Chinese) | gemma3 | qwen3 (with max_tokens) | translategemma | gpt-oss:20b (with max_tokens) |
|-----------|-------------------|--------|-------------------------|----------------|-------------------------------|
| Code Request | 請幫我寫一個計算費波那契數列的 Python 函數 | ✅ Please help me write a Python function to calculate the Fibonacci sequence. | ✅ Please write a Python function to calculate the Fibonacci sequence | ✅ Please help me write a Python function to calculate the Fibonacci sequence. | ✅ Please help me write a Python function that calculates the Fibonacci sequence. |
| Daily Conversation | 今天天氣很好，適合出門散步 | ✅ Today the weather is very good, suitable for going out for a walk. | ✅ Today's weather is great, perfect for going out for a walk. | ✅ The weather is very nice today, perfect for a walk. | ✅ Today the weather is very good, suitable for going out for a walk. |
| Technical Terms | 機器學習是人工智慧的一個重要分支 | ✅ Machine learning is an important branch of artificial intelligence. | ✅ Machine learning is an important branch of artificial intelligence. | ✅ Machine learning is an important branch of artificial intelligence. | ✅ Machine learning is an important branch of artificial intelligence. |

> **Note**: qwen3 and gpt-oss:20b require `max_tokens` configuration. See [Configuration for Thinking Models](#configuration-for-thinking-models-qwen3-deepseek-r1-etc) below.

### Translation Speed (seconds)

| Test Case | gemma3 | qwen3 (with max_tokens) | translategemma | gpt-oss:20b (with max_tokens) |
|-----------|--------|-------------------------|----------------|-------------------------------|
| Code Request | 2.97s | 8.48s | 3.51s | 5.44s |
| Daily Conversation | 0.42s | 13.02s | 0.37s | 6.36s |
| Technical Terms | 0.32s | 9.68s | 0.31s | 3.85s |

> **Note**: qwen3 is slower because it uses internal reasoning (thinking) before output. gpt-oss:20b is slower due to its larger model size (20B parameters).

### ChatML Format Test

Testing whether pretok correctly preserves special tokens and translates content when processing ChatML format prompts:

**Input:**
```
<|im_start|>system
你是一個專業的程式助手。
<|im_end|>
<|im_start|>user
請解釋什麼是遞迴函數
<|im_end|>
<|im_start|>assistant
```

| Model | Structure Preserved | Content Translated | Time |
|-------|--------------------|--------------------|------|
| gemma3:latest | ✅ | ✅ You are a professional programming assistant. / Please explain what a recursive function is. | 3.71s |
| qwen3:latest | ✅ | ❌ Content cleared | 5.06s |
| translategemma:4b | ✅ | ✅ You are a professional programming assistant. / Please explain what a recursive function is. | 2.88s |

## Conclusions and Recommendations

### Recommended Models

1. **translategemma:4b** ⭐ Recommended
   - Designed specifically for translation, best translation quality
   - Fastest speed
   - More natural and fluent translations (e.g., "perfect for a walk" vs "suitable for going out for a walk")

2. **gemma3:latest** ✅ Usable
   - General-purpose model with good translation quality
   - Moderate speed
   - Can be used as an alternative

3. **gpt-oss:20b** ✅ Usable (with max_tokens)
   - Large model (20B) with good translation quality
   - Slower due to model size (~5-6s per translation)
   - Requires `max_tokens=500` configuration

4. **qwen3:latest** ⚠️ Requires Configuration
   - Good translation quality when properly configured
   - Slower due to thinking process
   - **Requires `max_tokens=500+` configuration** (see below)

### Usage Example

```python
from pretok import Pretok
from pretok.config import LLMTranslatorConfig
from pretok.translation.llm import LLMTranslator

# Recommended configuration - using translategemma
config = LLMTranslatorConfig(
    api_key="ollama",
    model="translategemma:4b",  # Recommended
    base_url="http://localhost:11434/v1",
)
translator = LLMTranslator(config)
pretok = Pretok(target_language="en", translator=translator)
```

### Configuration for Thinking Models (qwen3, DeepSeek-R1, etc.)

Thinking models use internal reasoning tokens (`<think>...</think>`) before producing output.
The default `max_tokens` calculation (`len(text) * 4`) is insufficient for these models.

Some larger models like `gpt-oss:20b` may also require explicit `max_tokens` configuration.

**Solution**: Set `max_tokens` explicitly or increase the multiplier:

```python
# Option 1: Set explicit max_tokens (recommended for thinking models)
config = LLMTranslatorConfig(
    api_key="ollama",
    model="qwen3:latest",  # or gpt-oss:20b
    base_url="http://localhost:11434/v1",
    max_tokens=500,  # Sufficient for thinking + translation
)

# Option 2: Increase the multiplier
config = LLMTranslatorConfig(
    api_key="ollama",
    model="qwen3:latest",
    base_url="http://localhost:11434/v1",
    max_tokens_multiplier=15,  # 15x input length
)
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `max_tokens` | None | Fixed token limit. Overrides multiplier if set. |
| `max_tokens_multiplier` | 4 | Multiplier for `len(text) * multiplier`. |

## Notes

- Test results may vary depending on hardware configuration and model version
- Thinking models (qwen3, DeepSeek-R1) require higher `max_tokens` settings
- It is recommended to conduct your own tests before actual use
