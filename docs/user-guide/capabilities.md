# Model Capabilities

Define which languages each model supports.

## Capability Registry

```python
from pretok.capability import CapabilityRegistry, ModelCapability

registry = CapabilityRegistry()

# Register a model
registry.register(ModelCapability(
    model_id="llama-2-7b",
    supported_languages=frozenset(["en"]),
    primary_language="en",
))
```

## Built-in Profiles

pretok includes profiles for common models:

```python
from pretok.capability import load_builtin_profiles

registry = CapabilityRegistry()
load_builtin_profiles(registry)

# Now includes GPT-4, Llama-2, etc.
```

## Configuration

Define capabilities in your config file:

```yaml
models:
  gpt-4:
    supported_languages: [en, zh, ja, ko, fr, de, es]
    primary_language: en
  
  llama-2-7b:
    supported_languages: [en]
    primary_language: en
  
  my-custom-model:
    supported_languages: [en, fr]
    primary_language: en
    fallback_language: en
```

## Checking Translation Requirements

```python
needs_translation, target_lang = registry.requires_translation(
    model_id="llama-2-7b",
    source_lang="fr",
)

print(needs_translation)  # True
print(target_lang)  # "en"
```

## Querying Capabilities

```python
# Find models supporting a language
models = registry.find_models_supporting("zh")

# Get all capabilities
for cap in registry.list_capabilities():
    print(f"{cap.model_id}: {cap.supported_languages}")
```
