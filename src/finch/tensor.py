from abc import abstractmethod
from typing import Union

import numpy as np
import juliacall as jc
import scipy.sparse as sp

from .julia import jl


class _Display:
    def __repr__(self):
        return jl.sprint(jl.show, self._obj)

    def __str__(self):
        return jl.sprint(jl.show, jl.MIME("text/plain"), self._obj)


class Tensor(_Display):
    """
    A wrapper class for Finch.Tensor and Finch.SwizzleArray.

    Two constructors are supported: `lvl`+`arr`, or `jl_data`.

    Parameters
    ----------
    lvl : AbstractLevel, optional
        Levels description.
    arr : ndarray, optional
        NumPy array that should fill `lvl`.
    jl_data : Finch.SwizzleArray, optional
        Raw Julia object.
    order : str or tuple[int, ...], optional
        Order of the tensor. The order numbers the dimensions from the fastest to slowest.
        The leaf nodes have mode `0` and the root node has mode `n-1`. If the tensor was square
        of size `N`, then `N .^ order == strides`. Available options are "C" (row-major),
        "F" (column-major), or a custom order. Default: row-major.

    Returns
    -------
    Tensor
        Python wrapper for Finch Tensor.

    Examples
    --------
    >>> import numpy as np
    >>> import finch
    >>> x = finch.Tensor(lvl=finch.Dense(finch.Element(0)), arr=np.arange(3))
    >>> x.todense()
    array([0, 1, 2])
    """
    row_major = "C"
    column_major = "F"

    def __init__(self, lvl=None, arr=None, jl_data=None, order=None):
        if arr is not None and not isinstance(arr, np.ndarray):
            raise ValueError("For now only numpy input allowed.")

        # constructor for levels description and NumPy array input
        if lvl is not None and arr is not None and jl_data is None:
            order = self.preprocess_order(order, arr.ndim)
            inv_order = tuple(i - 1 for i in jl.invperm(order))
            self._obj = jl.swizzle(jl.Tensor(lvl._obj, arr.transpose(inv_order)), *order)
        # constructor for a raw julia object
        elif jl_data is not None and lvl is None and arr is None:
            if order is not None:
                raise ValueError("When passing a julia object order can't be provided.")
            if not jl.isa(jl_data, jl.Finch.SwizzleArray):
                raise ValueError("`jl_data` must be a SwizzleArray.")
            self._obj = jl_data
        else:
            raise ValueError(
                "Either `lvl` and numpy `arr` should be provided or a raw julia object."
            )

    def __pos__(self):
        return Tensor(jl_data=jl.Base.broadcast(jl.seval("+"), self._obj))

    def __add__(self, other):
        return Tensor(jl_data=jl.Base.broadcast(jl.seval("+"), self._obj, other._obj))

    def __mul__(self, other):
        return Tensor(jl_data=jl.Base.broadcast(jl.seval("*"), self._obj, other._obj))

    def __sub__(self, other):
        return Tensor(jl_data=jl.Base.broadcast(jl.seval("-"), self._obj, other._obj))

    def __truediv__(self, other):
        return Tensor(jl_data=jl.Base.broadcast(jl.seval("/"), self._obj, other._obj))

    @property
    def dtype(self) -> jc.TypeValue:
        return jl.eltype(self._obj)

    @property
    def ndim(self) -> int:
        return jl.ndims(self._obj)

    @property
    def shape(self) -> tuple[int, ...]:
        return jl.size(self._obj)

    @property
    def size(self) -> int:
        return np.prod(self.shape)

    @property
    def _is_dense(self) -> bool:
        lvl = self._obj.body.lvl
        for _ in self.shape:
            if not jl.isa(self._obj, jl.Finch.Dense):
                return False
            lvl = lvl.lvl
        return True

    @property
    def _order(self) -> tuple[int, ...]:
        return jl.typeof(self._obj).parameters[1]

    @classmethod
    def preprocess_order(
        cls, order: Union[str, tuple[int, ...], None], ndim: int
    ) -> tuple[int, ...]:
        if order == cls.column_major:
            permutation = tuple(range(1, ndim + 1))
        elif order == cls.row_major or order is None:
            permutation = tuple(range(1, ndim + 1)[::-1])
        elif isinstance(order, tuple):
            if min(order) == 0:
                order = tuple(i + 1 for i in order)
            if (
                len(order) == ndim and
                all([i in order for i in range(1, ndim + 1)])
            ):
                permutation = order
            else:
                raise ValueError(f"Custom order is not a permutation: {order}.")
        else:
            raise ValueError(f"order must be 'C', 'F' or a tuple, but is: {type(order)}.")

        return permutation

    def get_order(self, zero_indexing=True) -> tuple[int, ...]:
        order = self._order
        if zero_indexing:
            order = tuple(i - 1 for i in order)
        return order

    def get_inv_order(self, zero_indexing=True) -> tuple[int, ...]:
        inv_order = jl.invperm(self._order)
        if zero_indexing:
            inv_order = tuple(i - 1 for i in inv_order)
        return inv_order

    def todense(self) -> np.ndarray:
        obj = self._obj

        if self._is_dense:
            # don't materialize a dense finch tensor
            shape = jl.size(obj.body)
            dense_tensor = obj.body.lvl
        else:
            # create materialized dense array
            shape = jl.size(obj)
            dense_lvls = jl.Element(jl.default(obj))
            for _ in range(self.ndim):
                dense_lvls = jl.Dense(dense_lvls)
            dense_tensor = jl.Tensor(dense_lvls, obj).lvl  # materialize

        for _ in range(self.ndim):
            dense_tensor = dense_tensor.lvl

        result = np.asarray(jl.reshape(dense_tensor.val, shape))
        return result.transpose(self.get_inv_order()) if self._is_dense else result

    def permute_dims(self, axes: tuple[int, ...]) -> "Tensor":
        axes = tuple(i + 1 for i in axes)
        new_obj = jl.swizzle(self._obj, *axes)
        new_tensor = Tensor(jl_data=new_obj)
        return new_tensor


class COO(Tensor):
    def __init__(self, coords, data, shape, fill_value=0.0, order=Tensor.column_major):
        assert len(coords) == 2
        ndim = len(shape)
        order = self.preprocess_order(order, ndim)

        lvl = jl.Element(data.dtype.type(fill_value), jl.Vector(data))
        ptr = jl.Vector[jl.Int]([1, len(data) + 1])
        tbl = tuple(jl.Vector(arr + 1) for arr in coords)

        jl_data = jl.SparseCOO[ndim](lvl, shape, ptr, tbl)
        jl_data = jl.swizzle(jl.Tensor(jl_data), *order)

        super().__init__(jl_data=jl_data)

    @classmethod
    def from_scipy_sparse(cls, x):
        if not isinstance(x, sp.coo_matrix):
            raise ValueError(f"Input must be a scipy coo matrix, but it's: {type(x)}")

        return cls(coords=(x.col, x.row), data=x.data, shape=x.shape[::-1], order=cls.row_major)


class _Compressed2D(Tensor):
    def __init__(self, arg, shape, fill_value=0.0, order=Tensor.row_major):
        assert isinstance(arg, tuple) and len(arg) == 3
        assert len(shape) == 2

        data, indices, indptr = arg
        dtype = data.dtype.type
        data = jl.Vector(data)
        indices = jl.Vector(indices + 1)
        indptr = jl.Vector(indptr + 1)

        lvl = jl.Element(dtype(fill_value), data)
        jl_data = jl.swizzle(
            jl.Tensor(
                jl.Dense(jl.SparseList(lvl, shape[0], indptr, indices), shape[1])
            ),
            *self.get_permutation(order),
        )

        super().__init__(jl_data=jl_data)

    @abstractmethod
    def get_permutation(self, order: str) -> tuple[int, int]:
        ...

    @classmethod
    @abstractmethod
    def get_scipy_class(cls) -> Union[sp.csc_matrix, sp.csr_matrix]:
        ...

    @classmethod
    @abstractmethod
    def preprocess_scipy_shape(cls, shape: tuple[int, int]) -> tuple[int, int]:
        ...

    @classmethod
    def from_scipy_sparse(cls, x):
        scipy_class = cls.get_scipy_class()
        if not isinstance(x, scipy_class):
            raise ValueError(f"Input must be a {scipy_class} but it's: {type(x)}")

        indices = np.array(x.indices, dtype=x.data.dtype)
        indptr = np.array(x.indptr, dtype=x.data.dtype)
        return cls(
            arg=(x.data, indices, indptr),
            shape=cls.preprocess_scipy_shape(x.shape),
            order=cls.column_major,
        )


class CSC(_Compressed2D):
    def get_permutation(self, order: str) -> tuple[int, int]:
        return (2, 1) if order == Tensor.row_major else (1, 2)

    @classmethod
    def get_scipy_class(cls) -> Union[sp.csc_matrix, sp.csr_matrix]:
        return sp.csc_matrix

    @classmethod
    @abstractmethod
    def preprocess_scipy_shape(cls, shape: tuple[int, int]) -> tuple[int, int]:
        return shape


class CSR(_Compressed2D):
    def get_permutation(self, order: str) -> tuple[int, int]:
        return (1, 2) if order == Tensor.row_major else (2, 1)

    @classmethod
    def get_scipy_class(cls) -> Union[sp.csc_matrix, sp.csr_matrix]:
        return sp.csr_matrix

    @classmethod
    @abstractmethod
    def preprocess_scipy_shape(cls, shape: tuple[int, int]) -> tuple[int, int]:
        return shape[::-1]


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

        jl_data = jl.swizzle(jl.Tensor(jl.Dense(lvl, shape[-1])), *range(1, len(shape) + 1))

        super().__init__(jl_data=jl_data)


def fsprand(*args, order=None):
    return Tensor(jl_data=jl.fsprand(*args), order=order)


def permute_dims(x: Tensor, axes: tuple[int, ...]):
    return x.permute_dims(axes)


# LEVELS


class AbstractLevel(_Display):
    pass


# core levels


class Dense(AbstractLevel):
    def __init__(self, lvl, shape=None):
        args = [lvl._obj]
        if shape is not None:
            args.append(shape)
        self._obj = jl.Dense(*args)


class Element(AbstractLevel):
    def __init__(self, fill_value, data=None):
        args = [fill_value]
        if data is not None:
            args.append(data)
        self._obj = jl.Element(*args)


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
