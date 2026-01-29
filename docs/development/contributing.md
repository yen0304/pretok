# Contributing to pretok

Thank you for your interest in contributing to pretok!

## Development Setup

1. Clone the repository:

```bash
git clone https://github.com/yen0304/pretok.git
cd pretok
```

2. Install dependencies with uv:

```bash
uv sync --dev
```

3. Install pre-commit hooks:

```bash
uv run pre-commit install
```

## Code Quality

We use the following tools:

- **ruff** - Linting and formatting
- **mypy** - Type checking
- **pytest** - Testing

### Running Checks

```bash
# Linting
uv run ruff check src/ tests/

# Formatting
uv run ruff format src/ tests/

# Type checking
uv run mypy src/

# Tests
uv run pytest
```

## Pull Request Process

1. Fork the repository
2. Create a feature branch: `git checkout -b feat/my-feature`
3. Make your changes
4. Run all checks: `uv run ruff check && uv run mypy src/ && uv run pytest`
5. Commit with conventional commits: `git commit -m "feat: add new feature"`
6. Push and create a Pull Request

## Commit Convention

We use [Conventional Commits](https://www.conventionalcommits.org/):

- `feat:` - New features
- `fix:` - Bug fixes
- `docs:` - Documentation changes
- `test:` - Test changes
- `chore:` - Maintenance tasks
- `refactor:` - Code refactoring

## Testing

- Write tests for new features
- Maintain >80% code coverage
- Use pytest fixtures for common setup
- Mark slow tests with `@pytest.mark.slow`

## Documentation

- Update docs for user-facing changes
- Include docstrings for public APIs
- Add examples where helpful
