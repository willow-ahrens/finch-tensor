import numpy as np

from .julia import jl


int_: jl.DataType = jl.Int
int8: jl.DataType = jl.Int8
int16: jl.DataType = jl.Int16
int32: jl.DataType = jl.Int32
int64: jl.DataType = jl.Int64
uint: jl.DataType = jl.UInt
uint8: jl.DataType = jl.UInt8
uint16: jl.DataType = jl.UInt16
uint32: jl.DataType = jl.UInt32
uint64: jl.DataType = jl.UInt64
float16: jl.DataType = jl.Float16
float32: jl.DataType = jl.Float32
float64: jl.DataType = jl.Float64
complex64: jl.DataType = jl.ComplexF32
complex128: jl.DataType = jl.ComplexF64
bool: jl.DataType = jl.Bool

number: jl.DataType = jl.Number
complex: jl.DataType = jl.Complex
integer: jl.DataType = jl.Integer
abstract_float: jl.DataType = jl.AbstractFloat

jl_to_np_dtype = {
    int_: np.int_,
    int8: np.int8,
    int16: np.int16,
    int32: np.int32,
    int64: np.int64,
    uint8: np.uint8,
    uint16: np.uint16,
    uint32: np.uint32,
    uint64: np.uint64,
    float16: np.float16,
    float32: np.float32,
    float64: np.float64,
    complex64: np.complex64,
    complex128: np.complex128,
    bool: np.bool_,
    None: None,
}
