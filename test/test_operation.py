import pytest
import numpy as np
from operation import Operation

@pytest.fixture
def test_array():
    # Create two common numpy arrays for testing
    matrix_x = np.array([[1, 2, 3], [np.nan, 5, np.nan], [7, 8, 9], [10, 11, 12]])
    matrix_y = np.array([[np.nan, np.nan, np.nan], [4, np.nan, 6], [7, 8, 9], [10, 11, 12]])
    window_size_1 = 2
    window_size_2 = 3
    min_obs = 2
    return matrix_x, matrix_y, window_size_1, window_size_2, min_obs

@pytest.fixture
def op():
    # Create an instance of the Operation class
    return Operation()

def test_log(op):
    # Test case 1: Input array with small values and NaN
    x = np.array([[1e-99, np.nan], [np.nan, np.inf]])
    res_x = op.log(x)
    exp_x = np.array([[1e-99, np.nan], [np.nan, np.nan]])
    assert np.testing.assert_array_equal(res_x, exp_x) == None

def test_scale(op):
    x = np.array([[1, -2, np.nan], [3, 4, 5]])
    a = 2
    res = op.scale(x, a)
    exp = np.array([[0.6666667, -1.3333333, np.nan], [0.5, 0.6666667, 0.833333]])
    assert np.testing.assert_array_almost_equal(res, exp) == None

def test_signed_power(op):
    # Test case 1: Input matrix with positive and negative values, exponent = 2
    x = np.array([[1, -2, 3], [4, -5, -6]])
    exp = 2
    res = op.signed_power(x, exp)
    exp_res = np.array([[1, -4, 9], [16, -25, -36]])
    assert np.testing.assert_array_equal(res, exp_res) == None

    # Test case 2: Input matrix with positive and negative values, exponent = 0.5
    x = np.array([[1, -2, 3], [4, -5, -6]])
    exp = 0.5
    res = op.signed_power(x, exp)
    exp_res = np.array([[1.0, -1.41421356, 1.73205081], [2.0, -2.23606798, -2.44948974]])
    assert np.testing.assert_array_almost_equal(res, exp_res) == None

def test_shift(op):
    # Test case 1: Positive shift direction
    x = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    periods = 1
    res = op.shift(x, periods)
    exp_res = np.array([[np.nan, np.nan, np.nan], [1, 2, 3], [4, 5, 6]])
    assert np.testing.assert_array_equal(res, exp_res) == None

    # Test case 2: Negative shift direction
    x = np.array([[1, 2, 3], [np.nan, 5, 6], [7, 8, 9]])
    periods = -2
    fill_value = 0
    res = op.shift(x, periods, fill_value)
    exp_res = np.array([[7, 8, 9], [0, 0, 0], [0, 0, 0]])
    assert np.testing.assert_array_equal(res, exp_res) == None

def test_diff(op):
    # Test case 1: Default period = 1
    x = np.array([[1, 4, 7], [2, 5, 8], [3, 6, 9]])
    periods = 1
    res = op.diff(x, periods)
    exp_res = np.array([[np.nan, np.nan, np.nan], [1, 1, 1], [1, 1, 1]])
    assert np.testing.assert_array_equal(res, exp_res) == None

    # Test case 2: Period = -1
    periods = -1
    fill_value = 0
    res = op.diff(x, periods, fill_value)
    exp_res = np.array([[-1, -1, -1], [-1, -1, -1], [3, 6, 9]])
    assert np.testing.assert_array_equal(res, exp_res) == None

### Test cases for the rolling operations
def test_rolling_max(op, test_array):
    # Test case 1: Default window size = 2
    matrix_x, matrix_y, window_size_1, window_size_2, min_obs = test_array
    res = op.rolling_max(matrix_x, window_size_1)
    exp_res = np.array([[1, 2, 3], [1, 5, 3], [7, 8, 9], [10, 11, 12]])
    assert np.testing.assert_array_equal(res, exp_res) == None

    # Test case 2: Window size = 3, min_obs = 2
    res = op.rolling_max(matrix_y, window_size_2, min_obs)
    exp_res = np.array([[np.nan, np.nan, np.nan], [np.nan, np.nan, np.nan], [7, np.nan, 9], [10, 11, 12]])
    assert np.testing.assert_array_equal(res, exp_res) == None


def test_rolling_argmax(op, test_array):
    # Test case 1: Default window size = 2
    matrix_x, matrix_y, window_size_1, window_size_2, min_obs = test_array
    res = op.rolling_argmax(matrix_x, window_size_1)
    exp_res = np.array([[0, 0, 0], [1, 0, 1], [0, 0, 0], [0, 0, 0]])
    assert np.testing.assert_array_equal(res, exp_res) == None

    # Test case 2: Window size = 3, min_obs = 2
    res = op.rolling_argmax(matrix_y, window_size_2, min_obs)
    exp_res = np.array([[np.nan, np.nan, np.nan], [np.nan, np.nan, np.nan], [0, np.nan, 0], [0, 0, 0]])
    assert np.testing.assert_array_equal(res, exp_res) == None

def test_rolling_to_max(op, test_array):
    # Test case 1: Default window size = 2
    matrix_x, matrix_y, window_size_1, window_size_2, min_obs = test_array
    res = op.rolling_to_max(matrix_x, window_size_1)
    exp_res = np.array([[0, 0, 0], [np.nan, 0, np.nan], [0, 0, 0], [0, 0, 0]])
    assert np.testing.assert_array_equal(res, exp_res) == None

    # Test case 2: Window size = 3, min_obs = 2
    res = op.rolling_to_max(matrix_y, window_size_2, min_obs)
    exp_res = np.array([[np.nan, np.nan, np.nan], [np.nan, np.nan, np.nan], [0, np.nan, 0], [0, 0, 0]])
    assert np.testing.assert_array_equal(res, exp_res) == None

def test_rolling_to_min(op, test_array):
    # Test case 1: Default window size = 2
    matrix_x, matrix_y, window_size_1, window_size_2, min_obs = test_array
    res = op.rolling_to_min(matrix_x, window_size_1)
    exp_res = np.array([[0, 0, 0], [np.nan, 1, np.nan], [0, 1, 0], [1, 1, 1]])
    assert np.testing.assert_array_equal(res, exp_res) == None

    # Test case 2: Window size = 3, min_obs = 2
    res = op.rolling_to_min(matrix_y, window_size_2, min_obs)
    exp_res = np.array([[np.nan, np.nan, np.nan], [np.nan, np.nan, np.nan], [1, np.nan, 1], [2, 1, 2]])
    assert np.testing.assert_array_equal(res, exp_res) == None

def test_rolling_mean(op, test_array):
    # Test case 1: Default window size = 2
    matrix_x, matrix_y, window_size_1, window_size_2, min_obs = test_array
    res = op.rolling_mean(matrix_x, window_size_1)
    exp_res = np.array([[1, 2, 3], [1, 3.5, 3], [7, 6.5, 9], [8.5, 9.5, 10.5]])
    assert np.testing.assert_array_equal(res, exp_res) == None

    # Test case 2: Window size = 3, min_obs = 2
    res = op.rolling_mean(matrix_y, window_size_2, min_obs)
    exp_res = np.array([[np.nan, np.nan, np.nan], [np.nan, np.nan, np.nan], [5.5, np.nan, 7.5], [7, 9.5, 9]])
    assert np.testing.assert_array_equal(res, exp_res) == None
