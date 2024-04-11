from typing import Any, Literal, Union

import juliacall as jc
import numpy as np

from .julia import jl

OrderType = Union[Literal["C", "F"], tuple[int, ...], None]

TupleOf3Arrays = tuple[np.ndarray, np.ndarray, np.ndarray]

JuliaObj = jc.AnyValue

DType = jl.DataType

spmatrix = Any