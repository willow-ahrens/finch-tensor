import abc
from .julia import jl

def check_valid_order(order: tuple[int, ...], /, *, ndim: int | None = None) -> None:
    if ndim is not None and len(order) != ndim:
        raise ValueError(f"len(order) != ndim, {order=}, {ndim=}")

    if sorted(order) != list(range(len(order))):
        raise ValueError(f"sorted(order) != range(len(order)), {order=}")


def get_inverse_order(order: tuple[int, ...], /) -> tuple[int, ...]:
    check_valid_order(order)
    aorder = [0] * len(order)

    for pos, o in enumerate(order):
        aorder[o] = pos

    return tuple(aorder)


def get_topological_shape(
    shape: tuple[int | None, ...], /, *, order: tuple[int, ...]
) -> tuple[int | None, ...]:
    aorder = get_inverse_order(order)
    return tuple(shape[o] for o in aorder)

class Display(abc.ABC):
    def __repr__(self):
        return jl.sprint(jl.show, self._obj)

    def __str__(self):
        return jl.sprint(jl.show, jl.MIME("text/plain"), self._obj)
