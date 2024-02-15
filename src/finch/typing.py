from typing import Any, Literal, Union

import juliacall as jc
import numpy as np

OrderType = Union[Literal["C", "F"], tuple[int, ...], None]

TupleOf3Arrays = tuple[np.ndarray, np.ndarray, np.ndarray]

JuliaObj = jc.AnyValue

spmatrix = Any
