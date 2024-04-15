import numpy as np
from numpy.testing import assert_equal
import pytest
import sparse

import finch


@pytest.mark.parametrize("dtype", [np.int64, np.float64, np.complex128])
@pytest.mark.parametrize("order", ["C", "F", None])
def test_wrappers(dtype, order):
    A = np.array([[0, 0, 4], [1, 0, 0], [2, 0, 5], [3, 0, 0]], dtype=dtype, order=order)
    B = np.array(np.stack([A, A], axis=2, dtype=dtype), order=order)

    B_finch = finch.Tensor(B)

    storage = finch.Storage(
        finch.Dense(finch.SparseList(finch.SparseList(finch.Element(dtype(0.0))))), order=order
    )
    B_finch = B_finch.to_device(storage)

    assert_equal(B_finch.todense(), B)

    storage = finch.Storage(
        finch.Dense(finch.Dense(finch.Element(dtype(1.0)))), order=order
    )
    A_finch = finch.Tensor(A).to_device(storage)

    assert_equal(A_finch.todense(), A)
    assert A_finch.todense().dtype == A.dtype and B_finch.todense().dtype == B.dtype


@pytest.mark.parametrize("dtype", [np.int64, np.float64, np.complex128])
@pytest.mark.parametrize("order", ["C", "F", None])
def test_no_copy_fully_dense(dtype, order, arr3d):
    arr = np.array(arr3d, dtype=dtype, order=order)
    arr_finch = finch.Tensor(arr)
    arr_todense = arr_finch.todense()

    assert_equal(arr_todense, arr)
    assert np.shares_memory(arr_todense, arr)


def test_coo(rng):
    coords = (
        np.asarray([0, 1, 2, 3, 4], dtype=np.intp),
        np.asarray([0, 1, 2, 3, 4], dtype=np.intp),
    )
    data = rng.random(5)

    arr_pydata = sparse.COO(np.vstack(coords), data, shape=(5, 5))
    arr = arr_pydata.todense()
    arr_finch = finch.Tensor.construct_coo(coords, data, shape=(5, 5))

    assert_equal(arr_finch.todense(), arr)
    assert arr_finch.todense().dtype == data.dtype


@pytest.mark.parametrize(
    "classes",
    [(sparse._compressed.CSC, finch.Tensor.construct_csc),
     (sparse._compressed.CSR, finch.Tensor.construct_csr)],
)
def test_compressed2d(rng, classes):
    sparse_class, finch_class = classes
    indices, indptr, data = np.arange(5), np.arange(6), rng.random(5)

    arr_pydata = sparse_class((data, indices, indptr), shape=(5, 5))
    arr = arr_pydata.todense()
    arr_finch = finch_class((data, indices, indptr), shape=(5, 5))

    assert_equal(arr_finch.todense(), arr)
    assert arr_finch.todense().dtype == data.dtype


def test_csf(arr3d):
    arr = arr3d
    dtype = np.int64

    data = np.array([4, 1, 2, 1, 1, 2, 5, -1, 3, 3], dtype=dtype)
    indices_list = [
        np.array([1, 0, 1, 2, 0, 1, 2, 1, 0, 2], dtype=dtype),
        np.array([0, 1, 0, 1, 0, 1], dtype=dtype),
    ]
    indptr_list = [
        np.array([0, 1, 4, 5, 7, 8, 10], dtype=dtype), np.array([0, 2, 4, 5, 6], dtype=dtype)
    ]

    arr_finch = finch.Tensor.construct_csf((data, indices_list, indptr_list), shape=(3, 2, 4))

    assert_equal(arr_finch.todense(), arr)
    assert arr_finch.todense().dtype == data.dtype


@pytest.mark.parametrize(
    "permutation", [(0, 1, 2), (2, 1, 0), (0, 2, 1), (1, 2, 0), (2, 0, 1)]
)
@pytest.mark.parametrize("order", ["C", "F"])
def test_permute_dims(arr3d, permutation, order):
    arr = np.array(arr3d, order=order)
    storage = finch.Storage(
        finch.Dense(finch.SparseList(finch.SparseList(finch.Element(0)))), order=order
    )

    arr_finch = finch.Tensor(arr).to_device(storage)

    actual = finch.permute_dims(arr_finch, permutation)
    expected = np.transpose(arr, permutation)

    assert_equal(actual.todense(), expected)

    actual = finch.permute_dims(actual, permutation)
    expected = np.transpose(expected, permutation)

    assert_equal(actual.todense(), expected)


@pytest.mark.parametrize("order", ["C", "F"])
def test_astype(arr3d, order):
    arr = np.array(arr3d, order=order)
    storage = finch.Storage(
        finch.Dense(finch.SparseList(finch.SparseList(finch.Element(np.int64(0))))), order=order
    )
    arr_finch = finch.Tensor(arr).to_device(storage)

    result = finch.astype(arr_finch, finch.int64)
    assert_equal(result.todense(), arr)
    assert arr_finch != result

    result = finch.astype(arr_finch, finch.int64, copy=False)
    assert_equal(result.todense(), arr)
    assert arr_finch == result

    result = finch.astype(arr_finch, finch.float32)
    assert_equal(result.todense(), arr.astype(np.float32))

    with pytest.raises(ValueError, match="Unable to avoid a copy while casting in no-copy mode."):
        finch.astype(arr_finch, finch.float64, copy=False)


@pytest.mark.parametrize("random_state", [42, np.random.default_rng(42)])
def test_random(random_state):
    result = finch.random((10, 20, 30), density=0.0, random_state=random_state)
    expected = sparse.random((10, 20, 30), density=0.0, random_state=random_state)

    assert_equal(result.todense(), expected.todense())


@pytest.mark.parametrize("order", ["C", "F"])
@pytest.mark.parametrize("format", ["coo", "csr", "csc", "csf", "dense", None])
def test_asarray(arr2d, arr3d, order, format):
    arr = arr3d if format == "csf" else arr2d
    arr = np.array(arr, order=order)
    arr_finch = finch.Tensor(arr)

    if (format, order) in [("csr", "F"), ("csc", "C")]:
        with pytest.raises(ValueError, match="Invalid order for (csr|csc)"):
            finch.asarray(arr_finch, format=format)
    else:
        result = finch.asarray(arr_finch, format=format)
        assert_equal(result.todense(), arr)


@pytest.mark.parametrize(
    "order_and_format",
    [("C", None), ("F", None), ("C", "coo"), ("F", "coo"),("F", "csc")],
)
def test_where(order_and_format):
    order, format = order_and_format
    cond = np.array(
        [[True, False, False, False],
         [False, True, True, False],
         [True, False, True, True]],
        order=order,
    )
    arr1 = np.array([[0, 0, 0, 1], [0, 2, 0, 3], [1, 0, 0, 5]], order=order)
    arr2 = np.array([10, 20, 30, 40], order=order)

    tns_cond = finch.asarray(cond, format=format)
    arr1_cond = finch.asarray(arr1, format=format)
    arr2_cond = finch.asarray(arr2)

    actual = finch.where(tns_cond, arr1_cond, arr2_cond)
    expected = np.where(cond, arr1, arr2)

    assert_equal(actual.todense(), expected)


@pytest.mark.parametrize("order", ["C", "F"])
@pytest.mark.parametrize(
    "format_shape",
    [("coo", (80,)), ("coo", (10, 5, 8)), ("csf", (10, 5, 8)), ("csr", (5, 10)), ("csc", (5, 10))],
)
def test_nonzero(order, format_shape):
    format, shape = format_shape
    if (format, order) in [("csr", "F"), ("csc", "C")]:
        pytest.skip("invalid format+order")
    rng = np.random.default_rng(0)
    arr = rng.random(shape)
    arr = np.array(arr, order=order)
    mask = arr < 0.8
    arr[mask] = 0.0

    tns = finch.asarray(arr, format=format)

    actual = finch.nonzero(tns)
    expected = np.nonzero(arr)
    for actual_i, expected_i in zip(actual, expected):
        assert_equal(actual_i, expected_i)
