import numpy as np
from operation import Operation

def test_scale():
    xmatrix = np.array([[1, -2, 3], [3, 4, 5]])
    a = 2
    res = Operation.scale(xmatrix, a)
    expected = np.array([[0.6666667, -1.333333333, np.nan], [0.5, 0.66666667, 0.833333333]])
    assert np.allclose(res, expected)

