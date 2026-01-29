# Architecture

## Overview

pretok is designed as a modular, pluggable system for pre-token language adaptation.

```
┌─────────────────────────────────────────────────────────────────────────┐
│                              PreTok Pipeline                             │
├─────────────────────────────────────────────────────────────────────────┤
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐ │
│  │   Language   │  │    Model     │  │   Segment    │  │ Translation  │ │
│  │  Detection   │──│  Capability  │──│  Processing  │──│   Engine     │ │
│  │   Module     │  │   Registry   │  │    Module    │  │    Module    │ │
│  └──────────────┘  └──────────────┘  └──────────────┘  └──────────────┘ │
│         │                 │                 │                 │         │
│         ▼                 ▼                 ▼                 ▼         │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │                      Configuration System                         │  │
│  └──────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────┘
```

## Processing Flow

1. **Input** - Raw text in any language
2. **Segment Parsing** - Split into typed segments
3. **Language Detection** - Identify source language
4. **Capability Check** - Determine if translation needed
5. **Translation** - Translate content segments
6. **Reconstruction** - Reassemble the prompt
7. **Output** - Text ready for tokenization

## Module Design

### Plugin Architecture

Detection and translation backends implement protocols:

```python
class LanguageDetector(Protocol):
    def detect(self, text: str) -> DetectionResult: ...

class Translator(Protocol):
    async def translate(
        self, text: str, source: str, target: str
    ) -> TranslationResult: ...
```

### Configuration System

Three-tier configuration hierarchy:
1. Built-in defaults
2. Configuration file
3. Runtime overrides

### Caching Strategy

Translation caching with pluggable backends:
- In-memory LRU (default)
- Redis (distributed)
- SQLite (persistent)

## Design Principles

1. **Model-Agnostic** - No dependency on specific LLMs
2. **Pre-Token Boundary** - Raw text transformations only
3. **Structure Preservation** - Maintain prompt integrity
4. **Pluggable Backends** - Easy to extend
5. **Explicit Contracts** - Declared capabilities
