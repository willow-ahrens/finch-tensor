import abc
import typing

from .julia import jl
from . import levels
from . import utils
from dataclasses import dataclass


@dataclass
class Format:
    levels: tuple[levels.AbstractLevel, ...]
    order: tuple[int, ...]
    leaf: levels.AbstractLeafLevel

    def __init__(
        self,
        *,
        levels: tuple[levels.AbstractLevel, ...],
        order: tuple[int, ...] | None,
        leaf: levels.AbstractLeafLevel,
    ) -> None:
        if order is None:
            order = tuple(range(len(levels)))

        utils.check_valid_order(order, ndim=len(levels))
        self.order = order
        self.levels = levels
        self.leaf = leaf

    def _construct(self, *, fill_value, dtype: jl.DataType, data=None):
        out_level = self.leaf._construct(dtype=dtype, fill_value=fill_value)
        for level in reversed(self.levels):
            out_level = level._construct(inner_level=out_level)

        swizzle_args = map(lambda x: x + 1, reversed(self.order))
        if data is None:
            return jl.swizzle(jl.Tensor(out_level), *swizzle_args)

        return jl.swizzle(jl.Tensor(out_level, data), *swizzle_args)

class FlexibleFormat(abc.ABC):
    def _construct(self, *, ndim: int, fill_value, dtype: jl.DataType, data=None):
        return self._get_format(ndim)._construct(fill_value=fill_value, dtype=dtype, data=data)

    @abc.abstractmethod
    def _get_format(self, ndim: int, /) -> Format:
        pass

@dataclass
class Dense(FlexibleFormat):
    order: typing.Literal["C", "F"] | tuple[int, ...] = "C"
    shape: tuple[int | None, ...] | None = None

    def __post_init__(self) -> None:
        if isinstance(self.order, tuple):
            utils.check_valid_order(self.order)

            if self.shape is not None and len(self.order) != len(self.shape):
                raise ValueError(f"len(self.order) != len(self.shape), {self.order}, {self.shape}")

    def _get_format(self, ndim: int) -> Format:
        super()._get_format(ndim)
        match self.order:
            case "C":
                order = tuple(range(ndim))
            case "F":
                order = tuple(reversed(range(ndim)))
            case _:
                order = self.order

        utils.check_valid_order(order, ndim=ndim)
        
        shape = self.shape
        if shape is None:
            shape = (None,) * ndim
        
        if len(shape) != ndim:
            raise ValueError(f"len(self.shape != ndim), {shape=}, {ndim=}")
        
        topological_shape = utils.get_topological_shape(shape, order=order)
        lvls = tuple(levels.Dense(dim=dim) for dim in topological_shape)

        return Format(levels=lvls, order=order, leaf=levels.Element())