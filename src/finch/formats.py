import abc
import typing

import numpy as np

from .julia import jl
from . import levels as levels_module
from . import utils
from dataclasses import dataclass


@dataclass
class Format:
    levels: tuple[levels_module.AbstractLevel, ...]
    order: tuple[int, ...]
    leaf: levels_module.AbstractLeafLevel

    def __init__(
        self,
        *,
        levels: tuple[levels_module.AbstractLevel, ...],
        order: tuple[int, ...] | None,
        leaf: levels_module.AbstractLeafLevel,
    ) -> None:
        if order is None:
            order = tuple(range(len(levels)))

        utils.check_valid_order(order, ndim=len(levels))
        self.order = order
        self.levels = levels
        self.leaf = leaf

    def _construct(
        self, *, fill_value, dtype: jl.DataType, data: np.ndarray | None = None
    ):
        if data is not None:
            data_order = tuple(
                x[0]
                for x in sorted(enumerate(reversed(data.strides)), key=lambda x: x[1])
            )
            data_inv_order = utils.get_inverse_order(data_order)
            data_raw = data.transpose(data_inv_order)

        out_level = self.leaf._construct(dtype=dtype, fill_value=fill_value)
        for level in reversed(self.levels):
            out_level = level._construct(inner_level=out_level)

        reversed_order = tuple(reversed(self.order))
        swizzle_args = map(lambda x: reversed_order[x] + 1, data_order)
        if data is None:
            return jl.swizzle(jl.Tensor(out_level), *swizzle_args)

        return jl.swizzle(jl.Tensor(out_level, data_raw), *swizzle_args)


CSR = Format(
    levels=(levels_module.Dense(), levels_module.SparseList()),
    order=(0, 1),
    leaf=levels_module.Element(),
)
CSC = Format(
    levels=(levels_module.Dense(), levels_module.SparseList()),
    order=(1, 0),
    leaf=levels_module.Element(),
)


class FlexibleFormat(abc.ABC):
    def _construct(self, *, ndim: int, fill_value, dtype: jl.DataType, data=None):
        return self._get_format(ndim)._construct(
            fill_value=fill_value, dtype=dtype, data=data
        )

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
                raise ValueError(
                    f"len(self.order) != len(self.shape), {self.order}, {self.shape}"
                )

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
        lvls = tuple(levels_module.Dense(dim=dim) for dim in topological_shape)

        return Format(levels=lvls, order=order, leaf=levels_module.Element())
