import numpy as np
from numpy.testing import assert_equal
import pytest
import scipy.sparse as sp

import finch


@pytest.fixture
def arr2d():
    return np.array(
        [
            [0, 0, 3, 2, 0],
            [1, 0, 0, 1, 0],
            [0, 5, 0, 0, 0],
        ]
    )


def test_scipy_coo(arr2d):
    sp_arr = sp.coo_matrix(arr2d, dtype=np.int64)
    finch_arr = finch.Tensor(sp_arr)
    lvl = finch_arr._obj.body.lvl

    assert np.shares_memory(sp_arr.row, lvl.tbl[1].data)
    assert np.shares_memory(sp_arr.col, lvl.tbl[0].data)
    assert np.shares_memory(sp_arr.data, lvl.lvl.val)

    assert_equal(finch_arr.todense(), sp_arr.todense())
    new_arr = finch.permute_dims(finch_arr, (1, 0))
    assert_equal(new_arr.todense(), sp_arr.todense().transpose())


@pytest.mark.parametrize("cls", [sp.csc_matrix, sp.csr_matrix])
def test_scipy_compressed2d(arr2d, cls):
    sp_arr = cls(arr2d, dtype=np.int64)
    finch_arr = finch.Tensor(sp_arr)
    lvl = finch_arr._obj.body.lvl.lvl

    assert np.shares_memory(sp_arr.indices, lvl.idx.data)
    assert np.shares_memory(sp_arr.indptr, lvl.ptr.data)
    assert np.shares_memory(sp_arr.data, lvl.lvl.val)

    assert_equal(finch_arr.todense(), sp_arr.todense())
    new_arr = finch.permute_dims(finch_arr, (1, 0))
    assert_equal(new_arr.todense(), sp_arr.todense().transpose())
