from abc import abstractmethod
from typing import Tuple

import numpy as np
import juliacall as jc

from .julia import jl


class _Display:
    def __repr__(self):
        return jl.sprint(jl.show, self._obj)

    def __str__(self):
        return jl.sprint(jl.show, jl.MIME("text/plain"), self._obj)


class Tensor(_Display):
    def __init__(self, lvl=None, arr=None, jl_data=None):
        if arr is not None and not isinstance(arr, np.ndarray):
            raise ValueError("For now only numpy input allowed")

        if lvl is not None and arr is not None and jl_data is None:
            self._obj = jl.Tensor(lvl._obj, np.array(arr, order="F"))
        elif jl_data is not None:
            self._obj = jl_data
        else:
            raise ValueError(
                "either `lvl` and numpy `arr` should be provided or a raw julia object."
            )

    @property
    def dtype(self) -> jc.TypeValue:
        return jl.eltype(self._obj)

    @property
    def ndim(self) -> int:
        return jl.ndims(self._obj)

    @property
    def shape(self) -> Tuple[int, ...]:
        return jl.size(self._obj)

    @property
    def size(self) -> int:
        return np.prod(self.shape)

    def __pos__(self):
        return Tensor(jl_data=jl.Base.broadcast(jl.seval("+"), self._get_obj()))

    def __add__(self, other):
        return Tensor(jl_data=jl.Base.broadcast(jl.seval("+"), self._get_obj(), other._get_obj()))

    def __mul__(self, other):
        return Tensor(jl_data=jl.Base.broadcast(jl.seval("*"), self._get_obj(), other._get_obj()))

    def __sub__(self, other):
        return Tensor(jl_data=jl.Base.broadcast(jl.seval("-"), self._get_obj(), other._get_obj()))

    def __truediv__(self, other):
        return Tensor(jl_data=jl.Base.broadcast(jl.seval("/"), self._get_obj(), other._get_obj()))

    def todense(self) -> np.ndarray:
        obj = self._get_obj()
        shape = jl.size(obj)

        # create fully dense array and access `val`
        dense_lvls = jl.Element(jl.default(obj))
        for _ in shape:
            dense_lvls = jl.Dense(dense_lvls)
        dense_tensor = jl.Tensor(dense_lvls, obj).lvl
        for _ in shape:
            dense_tensor = dense_tensor.lvl

        result = np.array(dense_tensor.val).reshape(shape, order="F")
        return np.array(result, order="C")

    def _get_obj(self) -> jc.AnyValue:
        return self._get_materialized_obj() if self._is_swizzle() else self._obj

    def _is_swizzle(self) -> bool:
        return hasattr(self._obj, "body")

    def _get_materialized_obj(self) -> "Tensor":
        return jl.Tensor(self._get_lvl_description(), self._obj)

    def _get_lvl_description(self) -> jc.AnyValue:
        # TODO: move to the top of the file
        lvls_dict = {"Dense": Dense, "Element": Element, "Pattern": Pattern, "SparseList": SparseList}

        obj = self._obj
        fill_value = jl.default(obj)
        while hasattr(obj, "body"):
            obj = obj.body
        obj = obj.lvl

        lvls = []
        while not hasattr(obj, "val"):
            name = str(obj).split("{")[0]
            lvls.append(lvls_dict[name])
            obj = obj.lvl

        descr = Element(fill_value)
        for lvl in lvls[::-1]:
            descr = lvl(descr)
        return descr._obj


class COO(Tensor):
    def __init__(self, coords, data, shape, fill_value=0.0):
        assert coords.ndim == 2
        ndim = len(shape)

        lvl = jl.Element(data.dtype.type(fill_value), jl.Vector(data))
        ptr = jl.Vector[jl.Int]([1, len(data) + 1])
        tbl = tuple(jl.Vector(coords[i, :] + 1) for i in range(ndim))

        jl_data = jl.SparseCOO[ndim](lvl, shape, ptr, tbl)

        self._obj = jl.Tensor(jl_data)


class _Compressed2D(Tensor):
    def __init__(self, arg, shape, fill_value=0.0):
        assert isinstance(arg, tuple) and len(arg) == 3
        assert len(shape) == 2

        data, indices, indptr = arg
        dtype = data.dtype.type
        data = jl.Vector(data)
        indices = jl.Vector(indices + 1)
        indptr = jl.Vector(indptr + 1)

        lvl = jl.Element(dtype(fill_value), data)
        self._obj = self.get_jl_data(shape, lvl, indptr, indices)

    @abstractmethod
    def get_jl_data(
        self,
        shape: Tuple[int, int],
        lvl: jc.AnyValue,
        indptr: jc.VectorValue,
        indices: jc.VectorValue,
    ) -> jc.AnyValue:
        ...


class CSC(_Compressed2D):
    def get_jl_data(self, shape, lvl, indptr, indices):
        return jl.Tensor(
            jl.Dense(jl.SparseList(lvl, shape[0], indptr, indices), shape[1])
        )


class CSR(_Compressed2D):
    def get_jl_data(self, shape, lvl, indptr, indices):
        return jl.swizzle(
            jl.Tensor(
                jl.Dense(jl.SparseList(lvl, shape[0], indptr, indices), shape[1])
            ),
            2,
            1,
        )


class CSF(Tensor):
    def __init__(self, arg, shape, fill_value=0.0):
        assert isinstance(arg, tuple) and len(arg) == 3

        data, indices_list, indptr_list = arg
        dtype = data.dtype.type

        assert len(indices_list) == len(shape) - 1
        assert len(indptr_list) == len(shape) - 1

        data = jl.Vector(data)
        indices_list = [jl.Vector(i + 1) for i in indices_list]
        indptr_list = [jl.Vector(i + 1) for i in indptr_list]

        lvl = jl.Element(dtype(fill_value), data)
        for size, indices, indptr in zip(shape[:-1], indices_list, indptr_list):
            lvl = jl.SparseList(lvl, size, indptr, indices)

        jl_data = jl.Dense(lvl, shape[-1])
        self._obj = jl.Tensor(jl_data)


def fsprand(*args):
    return Tensor(jl_data=jl.fsprand(*args))


def permute_dims(x: Tensor, axes: tuple[int, ...]):
    axes = tuple(i + 1 for i in axes)
    return Tensor(jl_data=jl.swizzle(x._obj, *axes))


# LEVELS


class AbstractLevel(_Display):
    pass


# core levels


class Dense(AbstractLevel):
    def __init__(self, lvl):
        self._obj = jl.Dense(lvl._obj)


class Element(AbstractLevel):
    def __init__(self, val):
        self._obj = jl.Element(val)


class Pattern(AbstractLevel):
    def __init__(self):
        self._obj = jl.Pattern()


# advanced levels


class SparseList(AbstractLevel):
    def __init__(self, lvl):
        self._obj = jl.SparseList(lvl._obj)


class SparseByteMap(AbstractLevel):
    def __init__(self, lvl):
        self._obj = jl.SparseByteMap(lvl._obj)


class RepeatRLE(AbstractLevel):
    def __init__(self, lvl):
        self._obj = jl.RepeatRLE(lvl._obj)


class SparseVBL(AbstractLevel):
    def __init__(self, lvl):
        self._obj = jl.SparseVBL(lvl._obj)


class SparseCOO(AbstractLevel):
    def __init__(self, ndim, lvl):
        self._obj = jl.SparseCOO[ndim](lvl._obj)


class SparseHash(AbstractLevel):
    def __init__(self, ndim, lvl):
        self._obj = jl.SparseHash[ndim](lvl._obj)
