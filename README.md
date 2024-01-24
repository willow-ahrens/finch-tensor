# finch-tensor

This is the beginnings of a sparse tensor library for Python, backed by the
[Finch.jl](https://github.com/willow-ahrens/Finch.jl) tensor compiler.

## Installation

`finch-tensor` is available on PyPi, and can be installed with pip:
```bash
pip install finch-tensor
```

## Contributing

### Local setup

`conda` can be used for creating a local development setup:

```bash
git clone https://github.com/willow-ahrens/finch-tensor.git
cd finch-tensor/
conda create --name finch-tensor-dev python=3.9
conda activate finch-tensor-dev
pip install .
```

### Testing

`finch-tensor` uses [pytest](https://docs.pytest.org/en/latest/) for testing. To run the
tests:

```bash
pytest tests
```
