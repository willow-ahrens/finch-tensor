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
