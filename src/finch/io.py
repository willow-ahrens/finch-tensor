from .julia import jl
from .tensor import Tensor


def read(filename: str) -> Tensor:
    julia_obj = jl.fread(filename)
    return Tensor(julia_obj)


def write(filename: str, tns: Tensor) -> None:
    jl.fwrite(filename, tns._obj)
