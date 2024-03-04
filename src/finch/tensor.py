from typing import Any, Optional, Union

import numpy as np
from numpy.core.numeric import normalize_axis_index, normalize_axis_tuple

from .julia import jl
from .levels import _Display, Dense, Element, Storage
from .typing import OrderType, JuliaObj, spmatrix, TupleOf3Arrays


class Tensor(_Display):
    """
    A wrapper class for Finch.Tensor and Finch.SwizzleArray.

    Constructors
    ------------
    Tensor(scipy.sparse.spmatrix)
        Construct a Tensor out of a `scipy.sparse` object. Supported formats are: `COO`, `CSC`, and `CSR`.
    Tensor(numpy.ndarray)
        Construct a Tensor out of a NumPy array object. This is a no-copy operation.
    Tensor(Storage)
        Initialize a Tensor with a `storage` description. `storage` can already hold data.
    Tensor(julia_object)
        Tensor created from a compatible raw Julia object. Must be a `SwizzleArray`.
        This is a no-copy operation.

    Parameters
    ----------
    obj : np.ndarray or scipy.sparse or Storage or Finch.SwizzleArray
        Input to construct a Tensor. It's a no-copy operation of for NumPy and SciPy input. For Storage
        it's levels' description with order. The order numbers the dimensions from the fastest to slowest.
        The leaf nodes have mode `0` and the root node has mode `n-1`. If the tensor was square of size `N`,
        then `N .^ order == strides`. Available options are "C" (row-major), "F" (column-major), or a custom
        order. Default: row-major.
    fill_value : np.number, optional
        Only used when `arr : np.ndarray` is passed.

    Returns
    -------
    Tensor
        Python wrapper for Finch Tensor.

    Examples
    --------
    >>> import numpy as np
    >>> import finch
    >>> arr2d = np.arange(6).reshape((2, 3))
    >>> t1 = finch.Tensor(arr2d)
    >>> t1.todense()
    array([[0, 1, 2],
           [3, 4, 5]])
    >>> np.shares_memory(t1.todense(), arr2d)
    True
    >>> storage = finch.Storage(finch.Dense(finch.SparseList(finch.Element(1))), order="C")
    >>> t2 = t1.to_device(storage)
    >>> t2.todense()
    array([[0, 1, 2],
           [3, 4, 5]])
    """
    row_major = "C"
    column_major = "F"

    def __init__(
        self,
        obj: Union[np.ndarray, spmatrix, Storage, JuliaObj],
        /,
        *,
        fill_value: np.number = 0.0,
    ):
        if _is_scipy_sparse_obj(obj):  # scipy constructor
            jl_data = self._from_scipy_sparse(obj)
            self._obj = jl_data
        elif isinstance(obj, np.ndarray):  # numpy constructor
            jl_data = self._from_numpy(obj, fill_value=fill_value)
            self._obj = jl_data
        elif isinstance(obj, Storage):  # from-storage constructor
            order = self.preprocess_order(obj.order, self.get_lvl_ndim(obj.levels_descr._obj))
            self._obj = jl.swizzle(jl.Tensor(obj.levels_descr._obj), *order)
        elif jl.isa(obj, jl.Finch.SwizzleArray):  # raw-Julia-object constructor
            self._obj = obj
        else:
            raise ValueError(
                "Either `arr`, `storage` or a raw julia object should be provided."
            )

    def __pos__(self):
        return Tensor(jl.Base.broadcast(jl.seval("+"), self._obj))

    def __add__(self, other):
        return Tensor(jl.Base.broadcast(jl.seval("+"), self._obj, other._obj))

    def __mul__(self, other):
        return Tensor(jl.Base.broadcast(jl.seval("*"), self._obj, other._obj))

    def __sub__(self, other):
        return Tensor(jl.Base.broadcast(jl.seval("-"), self._obj, other._obj))

    def __truediv__(self, other):
        return Tensor(jl.Base.broadcast(jl.seval("/"), self._obj, other._obj))

    def __getitem__(self, key):
        if not isinstance(key, tuple):
            key = (key,)
        key = _expand_ellipsis(key, self.shape)
        key = _add_missing_dims(key, self.shape)
        key = _add_plus_one(key, self.shape)

        result = self._obj[key]
        if jl.isa(result, jl.Finch.SwizzleArray):
            return Tensor(result)
        elif jl.isa(result, jl.Finch.Tensor):
            return Tensor(jl.swizzle(result, *range(1, jl.ndims(result) + 1)))
        else:
            return result

    @property
    def dtype(self) -> np.dtype:
        return jl.eltype(self._obj.body)

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
            if not jl.isa(lvl, jl.Finch.Dense):
                return False
            lvl = lvl.lvl
        return True

    @property
    def _order(self) -> tuple[int, ...]:
        return jl.typeof(self._obj).parameters[1]

    @classmethod
    def preprocess_order(
        cls, order: OrderType, ndim: int
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

    @classmethod
    def get_lvl_ndim(cls, lvl: JuliaObj) -> int:
        ndim = 0
        while True:
            ndim += 1
            lvl = lvl.lvl
            if jl.isa(lvl, jl.Finch.Element):
                break
        return ndim

    def get_order(self, zero_indexing: bool = True) -> tuple[int, ...]:
        order = self._order
        if zero_indexing:
            order = tuple(i - 1 for i in order)
        return order

    def get_inv_order(self, zero_indexing: bool = True) -> tuple[int, ...]:
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
        new_tensor = Tensor(new_obj)
        return new_tensor

    def to_device(self, device: Storage) -> "Tensor":
        return Tensor(self._from_other_tensor(self, storage=device))

    @classmethod
    def _from_other_tensor(cls, tensor: "Tensor", storage: Optional[Storage]) -> JuliaObj:
        order = cls.preprocess_order(storage.order, tensor.ndim)
        return jl.swizzle(
            jl.Tensor(storage.levels_descr._obj, tensor._obj.body), *order
        )

    @classmethod
    def _from_numpy(cls, arr: np.ndarray, fill_value: np.number) -> JuliaObj:
        order_char = "F" if np.isfortran(arr) else "C"
        order = cls.preprocess_order(order_char, arr.ndim)
        inv_order = tuple(i - 1 for i in jl.invperm(order))

        lvl = Element(arr.dtype.type(fill_value), arr.reshape(-1, order=order_char))
        for i in inv_order:
            lvl = Dense(lvl, arr.shape[i])
        return jl.swizzle(jl.Tensor(lvl._obj), *order)

    @classmethod
    def _from_scipy_sparse(cls, x) -> JuliaObj:
        if x.format == "coo":
            return cls.construct_coo_jl_object(
                coords=(x.col, x.row), data=x.data, shape=x.shape[::-1], order=Tensor.row_major
            )
        elif x.format == "csc":
            return cls.construct_csc_jl_object(
                arg=(x.data, x.indices, x.indptr),
                shape=x.shape,
            )
        elif x.format == "csr":
            return cls.construct_csr_jl_object(
                arg=(x.data, x.indices, x.indptr),
                shape=x.shape,
            )
        else:
            raise ValueError(f"Unsupported SciPy format: {type(x)}")

    @classmethod
    def construct_coo_jl_object(cls, coords, data, shape, order, fill_value=0.0) -> JuliaObj:
        assert len(coords) == 2
        ndim = len(shape)
        order = cls.preprocess_order(order, ndim)

        lvl = jl.Element(data.dtype.type(fill_value), data)
        ptr = jl.Vector[jl.Int]([1, len(data) + 1])
        tbl = tuple(jl.PlusOneVector(arr) for arr in coords)

        jl_data = jl.swizzle(jl.Tensor(jl.SparseCOO[ndim](lvl, shape, ptr, tbl)), *order)
        return jl_data

    @classmethod
    def construct_coo(cls, coords, data, shape, order=row_major, fill_value=0.0) -> "Tensor":
        return Tensor(cls.construct_coo_jl_object(coords, data, shape, order, fill_value))

    @staticmethod
    def _construct_compressed2d_jl_object(
        arg: TupleOf3Arrays,
        shape: tuple[int, ...],
        order: tuple[int, ...],
        fill_value: np.number = 0.0,
    ) -> JuliaObj:
        assert isinstance(arg, tuple) and len(arg) == 3
        assert len(shape) == 2

        data, indices, indptr = arg
        dtype = data.dtype.type
        indices = jl.PlusOneVector(indices)
        indptr = jl.PlusOneVector(indptr)

        lvl = jl.Element(dtype(fill_value), data)
        jl_data = jl.swizzle(
            jl.Tensor(jl.Dense(jl.SparseList(lvl, shape[0], indptr, indices), shape[1])), *order
        )
        return jl_data

    @classmethod
    def construct_csc_jl_object(cls, arg: TupleOf3Arrays, shape: tuple[int, ...]) -> JuliaObj:
        return cls._construct_compressed2d_jl_object(
            arg=arg, shape=shape, order=(1, 2)
        )

    @classmethod
    def construct_csc(cls, arg: TupleOf3Arrays, shape: tuple[int, ...]) -> "Tensor":
        return Tensor(cls.construct_csc_jl_object(arg, shape))

    @classmethod
    def construct_csr_jl_object(cls, arg: TupleOf3Arrays, shape: tuple[int, ...]) -> JuliaObj:
        return cls._construct_compressed2d_jl_object(
            arg=arg, shape=shape[::-1], order=(2, 1)
        )

    @classmethod
    def construct_csr(cls, arg: TupleOf3Arrays, shape: tuple[int, ...]) -> "Tensor":
        return Tensor(cls.construct_csr_jl_object(arg, shape))

    @staticmethod
    def construct_csf_jl_object(
        arg: TupleOf3Arrays, shape: tuple[int, ...], fill_value: np.number = 0.0
    ) -> JuliaObj:
        assert isinstance(arg, tuple) and len(arg) == 3

        data, indices_list, indptr_list = arg
        dtype = data.dtype.type

        assert len(indices_list) == len(shape) - 1
        assert len(indptr_list) == len(shape) - 1

        indices_list = [jl.PlusOneVector(i) for i in indices_list]
        indptr_list = [jl.PlusOneVector(i) for i in indptr_list]

        lvl = jl.Element(dtype(fill_value), data)
        for size, indices, indptr in zip(shape[:-1], indices_list, indptr_list):
            lvl = jl.SparseList(lvl, size, indptr, indices)

        jl_data = jl.swizzle(jl.Tensor(jl.Dense(lvl, shape[-1])), *range(1, len(shape) + 1))
        return jl_data

    @classmethod
    def construct_csf(cls, arg: TupleOf3Arrays, shape: tuple[int, ...]) -> "Tensor":
        return Tensor(cls.construct_csf_jl_object(arg, shape))


def fsprand(*args, order=None):
    return Tensor(jl.fsprand(*args))


def permute_dims(x: Tensor, axes: tuple[int, ...]):
    return x.permute_dims(axes)


def astype(x: Tensor, dtype: jl.DataType, /, *, copy: bool = True):
    if not copy:
        if x.dtype == dtype:
            return x
        else:
            raise ValueError("Unable to avoid a copy while casting in no-copy mode.")
    else:
        finch_tns = x._obj.body
        result = jl.copyto_b(
            jl.similar(finch_tns, jl.default(finch_tns), dtype), finch_tns
        )
        return Tensor(jl.swizzle(result, *x.get_order(zero_indexing=False)))


def _is_scipy_sparse_obj(x):
    return hasattr(x, "__module__") and x.__module__.startswith("scipy.sparse")


def _slice_plus_one(s: slice, size: int) -> range:
    step = s.step if s.step is not None else 1
    start_default = size if step < 0 else 1
    stop_default = 1 if step < 0 else size

    if s.start is not None:
        start = normalize_axis_index(s.start, size) + 1 if s.start < size else size
    else:
        start = start_default

    if s.stop is not None:
        stop_offset = 2 if step < 0 else 0
        stop = normalize_axis_index(s.stop, size) + stop_offset if s.stop < size else size
    else:
        stop = stop_default

    return jl.range(start=start, step=step, stop=stop)


def _add_plus_one(key: tuple, shape: tuple[int, ...]) -> tuple:
    new_key = ()
    for idx, size in zip(key, shape):
        if isinstance(idx, int):
            new_key += (normalize_axis_index(idx, size) + 1,)
        elif isinstance(idx, slice):
            new_key += (_slice_plus_one(idx, size),)
        elif isinstance(idx, (list, np.ndarray, tuple)):
            idx = normalize_axis_tuple(idx, size)
            new_key += (jl.Vector([i + 1 for i in idx]),)
        elif idx is None:
            raise IndexError("'None' in the index key isn't supported")
        else:
            new_key += (idx,)
    return new_key


def _expand_ellipsis(key: tuple, shape: tuple[int, ...]) -> tuple:
    ndim = len(shape)
    ellipsis_pos = None
    key_without_ellipsis = ()
    # first we need to find the ellipsis and confirm it's the only one
    for pos, idx in enumerate(key):
        if idx == Ellipsis:
            if ellipsis_pos is None:
                ellipsis_pos = pos
            else:
                raise IndexError("an index can only have a single ellipsis ('...')")
        else:
            key_without_ellipsis += (idx,)
    key = key_without_ellipsis

    # then we expand ellipsis with a full range
    if ellipsis_pos is not None:
        ellipsis_indices = range(ellipsis_pos, ellipsis_pos + ndim - len(key))
        new_key = ()
        key_iter = iter(key)
        for i in range(ndim):
            if i in ellipsis_indices:
                new_key = new_key + (jl.range(start=1, stop=shape[i]),)
            else:
                new_key = new_key + (next(key_iter),)
        key = new_key
    return key

def _add_missing_dims(key: tuple, shape: tuple[int, ...]) -> tuple:
    for i in range(len(key), len(shape)):
        key = key + (jl.range(start=1, stop=shape[i]),)
    return key
