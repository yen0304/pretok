# Translation Backends

pretok supports flexible translation backends, allowing you to use any LLM via OpenAI-compatible APIs or local HuggingFace models.

## LLM Translation (Recommended)

The `LLMTranslator` works with any OpenAI-compatible API, giving you maximum flexibility.

### OpenAI

```python
from pretok.translation.llm import LLMTranslator
from pretok.config import LLMTranslatorConfig

config = LLMTranslatorConfig(
    base_url="https://api.openai.com/v1",
    model="gpt-4o-mini",
    # api_key defaults to OPENAI_API_KEY env var
)
translator = LLMTranslator(config)
```

### OpenRouter

Access hundreds of models through OpenRouter:

```python
config = LLMTranslatorConfig(
    base_url="https://openrouter.ai/api/v1",
    model="anthropic/claude-3-haiku",
    # Set OPENROUTER_API_KEY or OPENAI_API_KEY
)
translator = LLMTranslator(config)
```

### Local Ollama

Run translations locally with Ollama:

```python
config = LLMTranslatorConfig(
    base_url="http://localhost:11434/v1",
    model="llama3",
    api_key="ollama",  # Ollama doesn't require a real key
)
translator = LLMTranslator(config)
```

### vLLM / LM Studio

Any OpenAI-compatible server works:

```python
config = LLMTranslatorConfig(
    base_url="http://localhost:8000/v1",  # vLLM server
    model="meta-llama/Llama-2-7b-chat-hf",
    api_key="not-needed",
)
```

### Custom Prompts

Customize translation prompts:

```python
config = LLMTranslatorConfig(
    base_url="https://api.openai.com/v1",
    model="gpt-4o-mini",
    system_prompt="You are a professional translator. Translate accurately.",
    user_prompt_template="Translate from {source_lang} to {target_lang}:\n{text}",
)
```

## HuggingFace Translation

For local, offline translation using HuggingFace models.

### NLLB (No Language Left Behind)

Meta's multilingual model supporting 200+ languages:

```python
from pretok.translation.huggingface import HuggingFaceTranslator
from pretok.config import HuggingFaceTranslatorConfig

config = HuggingFaceTranslatorConfig(
    model="facebook/nllb-200-distilled-600M",
    device="cuda",  # or "cpu", "auto"
)
translator = HuggingFaceTranslator(config)
```

### M2M100

Facebook's many-to-many multilingual model:

```python
config = HuggingFaceTranslatorConfig(
    model="facebook/m2m100_418M",
    device="auto",
)
translator = HuggingFaceTranslator(config)
```

## Translation Result

All translators return a `TranslationResult`:

```python
result = translator.translate(
    text="Bonjour le monde",
    target_language="en",
    source_language="fr",
)

print(result.source_text)       # "Bonjour le monde"
print(result.translated_text)   # "Hello world"
print(result.source_language)   # "fr"
print(result.target_language)   # "en"
print(result.was_translated)    # True
print(result.translator)        # "llm" or "huggingface"
```

## Choosing a Backend

| Backend | Pros | Cons |
|---------|------|------|
| **LLM (OpenAI)** | High quality, many languages | Requires API key, costs money |
| **LLM (OpenRouter)** | Access to many models, competitive pricing | Requires API key |
| **LLM (Ollama)** | Free, local, private | Requires local setup, slower |
| **HuggingFace NLLB** | Free, local, 200+ languages | Requires GPU for speed |
| **HuggingFace M2M100** | Free, local, good quality | Requires GPU for speed |

## Custom Translator

Implement the `Translator` protocol for custom backends:

```python
from pretok.translation import Translator, TranslationResult

class MyTranslator(Translator):
    @property
    def name(self) -> str:
        return "my_translator"

    def translate(
        self,
        text: str,
        target_language: str,
        source_language: str | None = None,
    ) -> TranslationResult:
        # Your translation logic
        translated = my_translate_function(text, target_language)
        return TranslationResult(
            source_text=text,
            translated_text=translated,
            source_language=source_language or "auto",
            target_language=target_language,
            translator=self.name,
        )
```
