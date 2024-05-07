from functools import wraps

from .julia import jl
from .tensor import Tensor


def compiled(func):
    @wraps(func)
    def wrapper_func(*args, **kwargs):
        new_args = []
        for arg in args:
            if isinstance(arg, Tensor) and not jl.isa(arg._obj, jl.Finch.LazyTensor):
                new_args.append(Tensor(jl.Finch.LazyTensor(arg._obj)))
            else:
                new_args.append(arg)

        result = func(*new_args, **kwargs)
        result_tensor = Tensor(jl.Finch.compute(result._obj))

        return result_tensor

    return wrapper_func


def lazy(tensor: Tensor):
    if tensor.is_computed():
        return Tensor(jl.Finch.LazyTensor(tensor._obj))
    return tensor


def compute(tensor: Tensor, *, verbose: bool = False):
    if not tensor.is_computed():
        return Tensor(jl.Finch.compute(tensor._obj, verbose=verbose))
    return tensor
