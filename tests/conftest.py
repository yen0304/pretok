"""pretok test configuration and fixtures."""

import pytest


@pytest.fixture
def sample_texts() -> dict[str, str]:
    """Sample texts in various languages for testing."""
    return {
        "en": "Hello, how are you?",
        "fr": "Bonjour, comment ça va?",
        "de": "Hallo, wie geht es dir?",
        "es": "Hola, ¿cómo estás?",
        "zh": "你好，你好吗？",
        "ja": "こんにちは、お元気ですか？",
        "ko": "안녕하세요, 어떻게 지내세요?",
        "ru": "Привет, как дела?",
        "ar": "مرحبا، كيف حالك؟",
        "pt": "Olá, como você está?",
    }


@pytest.fixture
def chatml_prompt() -> str:
    """Sample ChatML format prompt."""
    return """<|im_start|>system
You are a helpful assistant.
<|im_end|>
<|im_start|>user
Hello, how are you?
<|im_end|>"""


@pytest.fixture
def llama_prompt() -> str:
    """Sample Llama format prompt."""
    return """[INST] <<SYS>>
You are a helpful assistant.
<</SYS>>

Hello, how are you? [/INST]"""


@pytest.fixture
def alpaca_prompt() -> str:
    """Sample Alpaca format prompt."""
    return """### Instruction:
Translate this text to French.

### Input:
Hello, how are you?

### Response:"""
