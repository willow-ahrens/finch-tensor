import numpy as np
from numpy.testing import assert_equal
import pytest

import finch


arr2d = np.array([[1, 2, 0, 0], [0, 1, 0, 1]])

arr1d = np.array([1, 1, 2, 3])


def test_eager(arr3d):
    A_finch = finch.Tensor(arr3d)
    B_finch = finch.Tensor(arr2d)

    result = finch.multiply(A_finch, B_finch)

    assert_equal(result.todense(), np.multiply(arr3d, arr2d))


def test_lazy_mode(arr3d):
    A_finch = finch.Tensor(arr3d)
    B_finch = finch.Tensor(arr2d)
    C_finch = finch.Tensor(arr1d)

    @finch.compiled
    def my_custom_fun(arr1, arr2, arr3):
        temp = finch.multiply(arr1, arr2)
        temp = finch.divide(temp, arr3)
        reduced = finch.sum(temp, axis=(0, 1))
        return finch.add(temp, reduced)

    result = my_custom_fun(A_finch, B_finch, C_finch)

    temp = np.divide(np.multiply(arr3d, arr2d), arr1d)
    expected = np.add(temp, np.sum(temp, axis=(0, 1)))
    assert_equal(result.todense(), expected)

    A_lazy = finch.lazy(A_finch)
    B_lazy = finch.lazy(B_finch)
    mul_lazy = finch.multiply(A_lazy, B_lazy)
    result = finch.compute(mul_lazy)

    assert_equal(result.todense(), np.multiply(arr3d, arr2d))


@pytest.mark.parametrize(
    "meth_name", ["__pos__", "__neg__", "__abs__"],
)
def test_elemwise_ops_1_arg(arr3d, meth_name):
    A_finch = finch.Tensor(arr3d)

    actual = getattr(A_finch, meth_name)()
    expected = getattr(arr3d, meth_name)()

    assert_equal(actual.todense(), expected)


@pytest.mark.parametrize(
    "meth_name",
    ["__add__", "__mul__", "__sub__", "__truediv__", # "__floordiv__", "__mod__",
     "__pow__", "__and__", "__or__", "__xor__", "__lshift__", "__rshift__"],
)
def test_elemwise_ops_2_args(arr3d, meth_name):
    arr2d = np.array([[2, 3, 2, 3], [3, 2, 3, 2]])
    A_finch = finch.Tensor(arr3d)
    B_finch = finch.Tensor(arr2d)

    actual = getattr(A_finch, meth_name)(B_finch)
    expected = getattr(arr3d, meth_name)(arr2d)

    assert_equal(actual.todense(), expected)


@pytest.mark.parametrize("func_name", ["sum", "prod"])
@pytest.mark.parametrize("axis", [None, -1, 1, (0, 1), (0, 1, 2)])
@pytest.mark.parametrize("dtype", [None])
def test_reductions(arr3d, func_name, axis, dtype):
    A_finch = finch.Tensor(arr3d)

    actual = getattr(finch, func_name)(A_finch, axis=axis, dtype=dtype)
    expected = getattr(np, func_name)(arr3d, axis=axis, dtype=dtype)

    if isinstance(actual, finch.Tensor):
        actual = actual.todense()

    assert_equal(actual, expected)


@pytest.mark.parametrize(
    "storage",
    [
        None,
        (
            finch.Storage(finch.SparseList(finch.Element(np.int64(0))), order="C"),
            finch.Storage(finch.Dense(finch.SparseList(finch.Element(np.int64(0)))), order="C"),
            finch.Storage(
                finch.Dense(finch.SparseList(finch.SparseList(finch.Element(np.int64(0))))),
                order="C",
            ),
        )
    ]
)
def test_tensordot(arr3d, storage):
    A_finch = finch.Tensor(arr1d)
    B_finch = finch.Tensor(arr2d)
    C_finch = finch.Tensor(arr3d)
    if storage is not None:
        A_finch = A_finch.to_device(storage[0])
        B_finch = B_finch.to_device(storage[1])
        C_finch = C_finch.to_device(storage[2])

    actual = finch.tensordot(B_finch, B_finch)
    expected = np.tensordot(arr2d, arr2d)
    assert_equal(actual.todense(), expected)

    actual = finch.tensordot(B_finch, B_finch, axes=(1, 1))
    expected = np.tensordot(arr2d, arr2d, axes=(1, 1))
    assert_equal(actual.todense(), expected)

    actual = finch.tensordot(C_finch, finch.permute_dims(C_finch, (2, 1, 0)), axes=((2, 0), (0, 2)))
    expected = np.tensordot(arr3d, arr3d.T, axes=((2, 0), (0, 2)))
    assert_equal(actual.todense(), expected)

    actual = finch.tensordot(C_finch, A_finch, axes=(2, 0))
    expected = np.tensordot(arr3d, arr1d, axes=(2, 0))
    assert_equal(actual.todense(), expected)


def test_matmul(arr2d, arr3d):
    A_finch = finch.Tensor(arr2d)
    B_finch = finch.Tensor(arr2d.T)
    C_finch = finch.permute_dims(A_finch, (1, 0))
    D_finch = finch.Tensor(arr3d)

    actual = A_finch @ B_finch
    expected = arr2d @ arr2d.T
    assert_equal(actual.todense(), expected)

    actual = A_finch @ C_finch
    assert_equal(actual.todense(), expected)

    with pytest.raises(ValueError, match="Both tensors must be 2-dimensional"):
        A_finch @ D_finch
