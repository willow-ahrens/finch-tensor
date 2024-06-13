from pathlib import Path

from .julia import jl
from .tensor import Tensor


def read(filename: Path | str) -> Tensor:
    julia_obj = jl.fread(str(filename))
    return Tensor(julia_obj)


def write(filename: Path | str, tns: Tensor) -> None:
    jl.fwrite(str(filename), tns._obj)
