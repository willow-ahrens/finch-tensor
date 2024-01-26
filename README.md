# Finch

This is the beginnings of a sparse tensor library for Python, backed by the
[Finch.jl](https://github.com/willow-ahrens/Finch.jl) tensor compiler.

## Installation

Finch is available on PyPi, and can be installed with pip:
```bash
pip install finch
```

## Contributing

### Packaging

Finch uses [poetry](https://python-poetry.org/) for packaging.

To install for development, clone the repository and run:
```bash
poetry install --with test
```
to install the current project and dev dependencies.

### Pre-commit hooks

To add pre-commit hooks, run:
```bash
poetry run pre-commit install
```

### Testing

Finch uses [pytest](https://docs.pytest.org/en/latest/) for testing. To run the
tests:

```bash
poetry run pytest
```
