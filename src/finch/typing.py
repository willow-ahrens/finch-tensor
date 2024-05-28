from typing import Any, Literal, Union

import juliacall as jc
import numpy as np


OrderType = Union[Literal["C", "F"], tuple[int, ...], None]

TupleOf3Arrays = tuple[np.ndarray, np.ndarray, np.ndarray]

JuliaObj = jc.AnyValue

DType = jc.AnyValue  # represents jl.DataType

spmatrix = Any

Device = Union[Literal["cpu"], None]
