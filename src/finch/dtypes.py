from .julia import jl
from .typing import DType


int: DType = jl.Int
int8: DType = jl.Int8
int16: DType = jl.Int16
int32: DType = jl.Int32
int64: DType = jl.Int64
uint: DType = jl.UInt
uint8: DType = jl.UInt8
uint16: DType = jl.UInt16
uint32: DType = jl.UInt32
uint64: DType = jl.UInt64
float16: DType = jl.Float16
float32: DType = jl.Float32
float64: DType = jl.Float64
complex64: DType = jl.ComplexF32
complex128: DType = jl.ComplexF64
bool: DType = jl.Bool
