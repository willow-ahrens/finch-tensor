import julia
import os
from julia import Pkg
Pkg.activate(
    os.path.dirname(
        os.path.dirname(
            os.path.dirname(
                os.path.abspath(__file__)))))
Pkg.instantiate()
from julia import Finch
from julia import Base
from julia import Main

_plus = Main.eval("Base.:+")
_minus = Main.eval("Base.:-")

class Tensor:
    def __init__(self, data):
        self.data = data
    def ndim(self):
        return Base.ndims(self.data)
    def dtype(self):
        return Base.eltype(self.data)
    def __pos__(self):
        return Tensor(Base.broadcast(_plus, self.data))
    def __neg__(self):    
        return Tensor(Base.broadcast(_minus, self.data))
    def __add__(self, other):
        return Tensor(Base.broadcast(_plus, self.data, other.data))

def fsprand(*args):
    return Tensor(Finch.fsprand(*args))