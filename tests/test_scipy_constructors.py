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


@pytest.mark.parametrize(
    "classes",
    [(sp.coo_matrix, finch.COO), (sp.csc_matrix, finch.CSC), (sp.csr_matrix, finch.CSR)],
)
def test_scipy_constructor(arr2d, classes):
    scipy_class, finch_class = classes
    sp_arr = scipy_class(arr2d, dtype=np.int64)
    finch_arr = finch_class.from_scipy_sparse(sp_arr)

    assert_equal(finch_arr.todense(), sp_arr.todense())
