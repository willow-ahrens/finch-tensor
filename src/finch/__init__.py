import juliapkg
juliapkg.add("Finch", "9177782c-1635-4eb9-9bfb-d9dfa25e6bce", version="0.6")
import juliacall
juliapkg.resolve()
from juliacall import Main as jl

jl.seval("using Finch")

_plus = jl.seval("Base.:+")
_minus = jl.seval("Base.:-")

class Tensor:
    def __init__(self, data):
        self.data = data
    def ndim(self):
        return jl.ndims(self.data)
    def dtype(self):
        return jl.eltype(self.data)
    def __pos__(self):
        return Tensor(jl.Base.broadcast(_plus, self.data))
    def __neg__(self):    
        return Tensor(jl.Base.broadcast(_minus, self.data))
    def __add__(self, other):
        return Tensor(jl.Base.broadcast(_plus, self.data, other.data))

def fsprand(*args):
    return Tensor(jl.fsprand(*args))