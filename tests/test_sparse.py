import numpy as np
from numpy.testing import assert_equal
import pytest
import sparse

import finch


@pytest.fixture
def arr3d():
    return np.array(
        [
            [[0, 1, 0, 0], [1, 0, 0, 3]],
            [[4, 0, -1, 0], [2, 2, 0, 0]],
            [[0, 0, 0, 0], [1, 5, 0, 3]],
        ]
    )


@pytest.fixture
def rng():
    return np.random.default_rng(42)


@pytest.mark.parametrize("dtype", [np.int64, np.float64, np.complex128])
@pytest.mark.parametrize("order", ["C", "F"])
def test_wrappers(dtype, order):
    A = np.array([[0, 0, 4], [1, 0, 0], [2, 0, 5], [3, 0, 0]], dtype=dtype)
    B = np.stack([A, A], axis=2, dtype=dtype)
    scalar = finch.Tensor(finch.Element(2), np.array(2), order=order)

    levels = finch.Dense(finch.SparseList(finch.SparseList(finch.Element(dtype(0.0)))))
    B_finch = finch.Tensor(levels, B, order=order)

    assert_equal(B_finch.todense(), B)
    #assert_equal((B_finch * scalar + B_finch).todense(), B * 2 + B)

    levels = finch.Dense(finch.Dense(finch.Element(dtype(1.0))))
    A_finch = finch.Tensor(levels, A, order=order)

    assert_equal(A_finch.todense(), A)
    #assert_equal((A_finch * scalar + A_finch).todense(), A * 2 + A)

    assert A_finch.todense().dtype == A.dtype and B_finch.todense().dtype == B.dtype


@pytest.mark.parametrize("dtype", [np.int64, np.float64, np.complex128])
@pytest.mark.parametrize("order", ["C", "F"])
def test_no_copy_fully_dense(dtype, order, arr3d):
    arr = np.array(arr3d, dtype=dtype, order=order)
    arr_in = arr.transpose(None) if order == "F" else arr
    arr_shape = arr_in.shape

    levels = finch.Dense(
        finch.Dense(
            finch.Dense(
                finch.Element(dtype(0.0), arr_in.reshape(-1)),
                arr_shape[2]
            ),
            arr_shape[1]
        ),
        arr_shape[0]
    )
    arr_finch = finch.Tensor(lvl=levels, order=order)
    arr_todense = arr_finch.todense()

    assert_equal(arr_todense, arr)
    assert np.shares_memory(arr_todense, arr)


def test_coo(rng):
    coords = np.asarray(
        [[0, 1, 2, 3, 4], [0, 1, 2, 3, 4]],
        dtype=np.intp,
    )
    data = rng.random(5)
    scalar = finch.Tensor(finch.Element(2), np.array(2))

    arr_pydata = sparse.COO(coords, data, shape=(5, 5))
    arr = arr_pydata.todense()
    arr_finch = finch.COO(coords, data, shape=(5, 5))

    assert_equal(arr_finch.todense(), arr)
    #assert_equal((arr_finch * scalar + arr_finch).todense(), arr * 2 + arr)

    assert arr_finch.todense().dtype == data.dtype


@pytest.mark.parametrize(
    "classes",
    [(sparse._compressed.CSC, finch.CSC), (sparse._compressed.CSR, finch.CSR)],
)
def test_compressed2d(rng, classes):
    sparse_class, finch_class = classes
    indices, indptr, data = np.arange(5), np.arange(6), rng.random(5)
    scalar = finch.Tensor(finch.Element(2), np.array(2))

    arr_pydata = sparse_class((data, indices, indptr), shape=(5, 5))
    arr = arr_pydata.todense()
    arr_finch = finch_class((data, indices, indptr), shape=(5, 5))

    assert_equal(arr_finch.todense(), arr)
    #assert_equal((arr_finch * scalar + arr_finch).todense(), arr * 2 + arr)

    assert arr_finch.todense().dtype == data.dtype


def test_csf(arr3d):
    arr = arr3d
    dtype = np.int64
    scalar = finch.Tensor(finch.Element(2), np.array(2))

    data = np.array([4, 1, 2, 1, 1, 2, 5, -1, 3, 3], dtype=dtype)
    indices_list = [
        np.array([1, 0, 1, 2, 0, 1, 2, 1, 0, 2], dtype=dtype),
        np.array([0, 1, 0, 1, 0, 1], dtype=dtype),
    ]
    indptr_list = [
        np.array([0, 1, 4, 5, 7, 8, 10], dtype=dtype), np.array([0, 2, 4, 5, 6], dtype=dtype)
    ]

    arr_finch = finch.CSF((data, indices_list, indptr_list), shape=(3, 2, 4))

    assert_equal(arr_finch.todense(), arr)
    #assert_equal((arr_finch * scalar + arr_finch).todense(), arr * 2 + arr)

    assert arr_finch.todense().dtype == data.dtype


@pytest.mark.parametrize(
    "permutation", [(0, 1, 2), (2, 1, 0), (0, 2, 1), (1, 2, 0), (2, 0, 1)]
)
@pytest.mark.parametrize("order", ["C", "F"])
def test_permute_dims(arr3d, permutation, order):
    arr = arr3d

    levels = finch.Dense(finch.SparseList(finch.SparseList(finch.Element(0))))
    arr_finch = finch.Tensor(levels, arr, order=order)

    actual = finch.permute_dims(arr_finch, permutation)
    expected = np.transpose(arr, permutation)

    assert_equal(actual.todense(), expected)

    actual = finch.permute_dims(actual, permutation)
    expected = np.transpose(expected, permutation)

    assert_equal(actual.todense(), expected)
