import abc


from .julia import jl
from . import dtypes
from dataclasses import dataclass


class _Display(abc.ABC):
    def __repr__(self):
        return jl.sprint(jl.show, self._obj)

    def __str__(self):
        return jl.sprint(jl.show, jl.MIME("text/plain"), self._obj)


class AbstractLeafLevel(abc.ABC):
    @abc.abstractmethod
    def _construct(self, *, dtype, fill_value):
        ...


# LEVEL
class AbstractLevel(abc.ABC):
    @abc.abstractmethod
    def _construct(self, *, inner_level):
        ...


# core levels
@dataclass
class Dense(AbstractLevel):
    dim: int | None = None
    index_type: jl.DataType = dtypes.int64

    def _construct(self, *, inner_level) -> jl.Dense:
        if self.dim is None:
            return jl.Dense[self.index_type](inner_level)

        return jl.Dense[self.index_type](inner_level, self.dim)


@dataclass
class Element(AbstractLeafLevel):
    def _construct(self, *, dtype: jl.DataType, fill_value) -> jl.Element:
        return jl.Element[fill_value, dtype]()


@dataclass
class Pattern(AbstractLeafLevel):
    def _construct(self, *, dtype, fill_value) -> jl.Pattern:
        from .dtypes import bool

        if dtype != bool:
            raise TypeError("`Pattern` can only have `dtype=bool`.")
        if dtype(fill_value) != dtype(False):
            raise TypeError("`Pattern` can only have `fill_value=False`.")

        return jl.Pattern()


# advanced levels
@dataclass
class SparseList(AbstractLevel):
    index_type: jl.DataType = dtypes.int64
    pos_type: jl.DataType = dtypes.uint64
    crd_type: jl.DataType = dtypes.uint64

    def _construct(self, *, inner_level) -> jl.SparseList:
        return jl.SparseList[self.index_type, self.pos_type, self.crd_type](inner_level)
