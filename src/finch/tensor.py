from typing import Callable, Optional, Union

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
        Tensor created from a compatible raw Julia object. Must be a `SwizzleArray` or `LazyTensor`.
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
    row_major: str = "C"
    column_major: str = "F"

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
        elif jl.isa(obj, jl.Finch.Tensor):  # raw-Julia-object constructors
            self._obj = jl.swizzle(obj, *tuple(range(1, jl.ndims(obj) + 1)))
        elif jl.isa(obj, jl.Finch.SwizzleArray) or jl.isa(obj, jl.Finch.LazyTensor):
            self._obj = obj
        else:
            raise ValueError(
                "Either `arr`, `storage` or a raw julia object should be provided."
            )

    def __pos__(self):
        return self._elemwise_op("+")

    def __neg__(self):
        return self._elemwise_op("-")

    def __add__(self, other):
        return self._elemwise_op(".+", other)

    def __mul__(self, other):
        return self._elemwise_op(".*", other)

    def __sub__(self, other):
        return self._elemwise_op(".-", other)

    def __truediv__(self, other):
        return self._elemwise_op("./", other)

    def __floordiv__(self, other):
        return self._elemwise_op(".//", other)

    def __mod__(self, other):
        return self._elemwise_op("rem", other)

    def __pow__(self, other):
        return self._elemwise_op(".^", other)

    def __matmul__(self, other):
        raise NotImplementedError

    def __abs__(self):
        return self._elemwise_op("abs")

    def __invert__(self):
        return self._elemwise_op("~")

    def __and__(self, other):
        return self._elemwise_op("&", other)

    def __or__(self, other):
        return self._elemwise_op("|", other)

    def __xor__(self, other):
        return self._elemwise_op("xor", other)

    def __lshift__(self, other):
        return self._elemwise_op("<<", other)

    def __rshift__(self, other):
        return self._elemwise_op(">>", other)

    def _elemwise_op(self, op: str, other: Optional["Tensor"] = None) -> "Tensor":
        if other is None:
            result = jl.broadcast(jl.seval(op), self._obj)
        else:
            axis_x1, axis_x2 = range(self.ndim, 0, -1), range(other.ndim, 0, -1)
            # inverse swizzle, so `broadcast` appends new dims to the front
            result = jl.broadcast(
                jl.seval(op), 
                jl.permutedims(self._obj, tuple(axis_x1)),
                jl.permutedims(other._obj, tuple(axis_x2)),
            )
            # swizzle back to the original order
            result = jl.permutedims(result, tuple(range(jl.ndims(result), 0, -1)))

        return Tensor(result)

    def __getitem__(self, key):
        if not isinstance(key, tuple):
            key = (key,)
        key = _expand_ellipsis(key, self.shape)
        key = _add_missing_dims(key, self.shape)
        key = _add_plus_one(key, self.shape)

        result = self._obj[key]
        if jl.isa(result, jl.Finch.SwizzleArray) or jl.isa(result, jl.Finch.LazyTensor):
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

    def is_computed(self) -> bool:
        return not jl.isa(self._obj, jl.Finch.LazyTensor)

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


def random(shape, density=0.01, random_state=None):
    args = [*shape, density]
    if random_state is not None:
        if isinstance(random_state, np.random.Generator):
            seed = random_state.integers(np.iinfo(np.int32).max)
        else:
            seed = random_state
        rng = jl.default_rng(seed)
        args = [rng] + args
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


def _reduce(x: Tensor, fn: Callable, axis, dtype):
    if axis is not None:
        axis = normalize_axis_tuple(axis, x.ndim)
        axis = tuple(i + 1 for i in axis)
        result = fn(x._obj, dims=axis)
    else:
        result = fn(x._obj)

    if jl.isa(result, jl.Finch.Tensor) or jl.isa(result, jl.Finch.LazyTensor):
        result = Tensor(result)
    else:
        result = np.array(result)
    return result


def sum(
    x: Tensor,
    /,
    *,
    axis: Union[int, tuple[int, ...], None] = None,
    dtype: Union[jl.DataType, None] = None,
    keepdims: bool = False,
) -> Tensor:
    return _reduce(x, jl.sum, axis, dtype)


def prod(
    x: Tensor,
    /,
    *,
    axis: Union[int, tuple[int, ...], None] = None,
    dtype: Union[jl.DataType, None] = None,
    keepdims: bool = False,
) -> Tensor:
    return _reduce(x, jl.prod, axis, dtype)


def add(x1: Tensor, x2: Tensor, /) -> Tensor:
    return x1 + x2


def subtract(x1: Tensor, x2: Tensor, /) -> Tensor:
    return x1 - x2


def multiply(x1: Tensor, x2: Tensor, /) -> Tensor:
    return x1 * x2


def divide(x1: Tensor, x2: Tensor, /) -> Tensor:
    return x1 / x2


def floor_divide(x1: Tensor, x2: Tensor, /) -> Tensor:
    return x1 // x2


def pow(x1: Tensor, x2: Tensor, /) -> Tensor:
    return x1 ** x2


def positive(x: Tensor, /) -> Tensor:
    return +x


def negative(x: Tensor, /) -> Tensor:
    return -x


def abs(x: Tensor, /) -> Tensor:
    return x.__abs__()


def cos(x: Tensor, /) -> Tensor:
    return x._elemwise_op("cos")


def cosh(x: Tensor, /) -> Tensor:
    return x._elemwise_op("cosh")


def acos(x: Tensor, /) -> Tensor:
    return x._elemwise_op("acos")


def acosh(x: Tensor, /) -> Tensor:
    return x._elemwise_op("acosh")


def sin(x: Tensor, /) -> Tensor:
    return x._elemwise_op("sin")


def sinh(x: Tensor, /) -> Tensor:
    return x._elemwise_op("sinh")


def asin(x: Tensor, /) -> Tensor:
    return x._elemwise_op("asin")


def asinh(x: Tensor, /) -> Tensor:
    return x._elemwise_op("asinh")


def tan(x: Tensor, /) -> Tensor:
    return x._elemwise_op("tan")


def tanh(x: Tensor, /) -> Tensor:
    return x._elemwise_op("tanh")


def atan(x: Tensor, /) -> Tensor:
    return x._elemwise_op("atan")


def atanh(x: Tensor, /) -> Tensor:
    return x._elemwise_op("atanh")


def atan2(x: Tensor, other: Tensor, /) -> Tensor:
    return x._elemwise_op("atand", other)


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
