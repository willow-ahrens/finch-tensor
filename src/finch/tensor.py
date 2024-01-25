from .julia import jl

class Tensor:
    def __init__(self, data):
        self.data = data
    def ndim(self):
        return jl.ndims(self.data)
    def dtype(self):
        return jl.eltype(self.data)
    def __pos__(self):
        return Tensor(jl.Base.broadcast(jl.seval("+"), self.data))
    def __neg__(self):    
        return Tensor(jl.Base.broadcast(jl.seval("-"), self.data))
    def __add__(self, other):
        return Tensor(jl.Base.broadcast(jl.seval("+"), self.data, other.data))

def fsprand(*args):
    return Tensor(jl.fsprand(*args))