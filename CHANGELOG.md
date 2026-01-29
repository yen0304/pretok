# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2025-01-XX

### Added

#### Core Features
- **Pretok Pipeline**: Main `Pretok` class for text processing with language detection and translation
- **PipelineResult**: Rich result object with segments, detections, translations, and caching info
- **Factory Function**: `create_pretok()` for model-specific configuration

#### Language Detection
- **LangDetect Backend**: Pure Python language detection using `langdetect` library
- **FastText Backend**: High-accuracy detection with auto model download support
- **Composite Detector**: Multiple strategies (voting, weighted_average, fallback_chain)
- **DetectionResult**: Immutable result with language, confidence, and metadata

#### Translation
- **LLMTranslator**: Flexible translation using any OpenAI-compatible API
  - OpenAI, OpenRouter, Ollama, vLLM, LM Studio support
  - Customizable system/user prompts
- **HuggingFaceTranslator**: Local translation with NLLB, M2M100 models
- **TranslationResult**: Full context including source/target text and metadata

#### Segment Processing
- **PromptLexer**: Parse prompts into segments (text, code, roles, tokens)
- **Format Detection**: ChatML, Llama, Alpaca, Vicuna, Mistral formats
- **Structure Preservation**: Role markers, code blocks, and control tokens preserved

#### Model Capabilities
- **ModelCapability**: Define supported languages and token efficiency
- **ModelRegistry**: Pattern-based model lookup with defaults
- **Built-in Profiles**: GPT-4, Claude, Llama, Qwen, Mistral

#### Configuration
- **PretokConfig**: Pydantic-based configuration with validation
- **File Support**: YAML, TOML, JSON configuration files
- **Environment Variables**: `${VAR}` syntax for sensitive values
- **Auto-discovery**: Automatic config file detection

#### Caching
- **MemoryCache**: LRU cache with TTL support
- **Cache Stats**: Hit rate, size, and performance metrics

#### Developer Experience
- **Type Hints**: Full type annotations with `py.typed` marker
- **Strict mypy**: All modules pass strict type checking
- **Ruff**: Linting and formatting with comprehensive rules
- **pytest**: 182 tests with hypothesis property-based testing
- **CI/CD**: GitHub Actions for testing and releases
- **Documentation**: MkDocs with Material theme

### Technical Details
- Python 3.11+ required
- Zero required dependencies (detection/translation backends optional)
- Protocol-based interfaces for extensibility
