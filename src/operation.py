from typing import Any, Optional, Tuple, Union, Literal
import bottleneck as bn # type: ignore
import numpy as np
import numpy.typing as npt
import pandas as pd
import xarray as xr

class Operation:
    """Basic operations for matrix

    The Operation class provides a series of basic operations for matrix, including:
    
    - `Element-wise` operations
    
    - `Pair-wise` operations
    
    - `Rolling window` operations
    
    - `Cross-sectional` operations
    
    - Other operations

    Examples
    --------
    Call methods from Operation class in ''FactorPool'' by:
    
    >>> from operation import Operation
    >>> res = Operation.method_name(matrix, args)
    """

    @classmethod
    def _mulvar_helper(
        cls, xmatrix: npt.NDArray[Any], ymatrix: npt.NDArray[Any]
    ) -> Tuple[npt.NDArray[Any], npt.NDArray[Any]]:
        """Helper function for multi-variable operations that handles 'NaN'
        values in 'xmatrix' and 'ymatrix'.

        This class mathod indentifies postions where either 'xmatrix' or 'ymatrix' contain 'NaN' values
        and returns copies of the input with 'NaNs' placed in these positions in both nd-arrays.

        Parameters
        ----------
        xmatrix : array_like
            The input matrix x. Must match the shape of 'ymatrix'.
        ymatrix : array_like
            The input matrix y. Must match the shape of 'xmatrix'.
        
        Returns
        ----------
        Tuple[ndarray, ndarray]
            A tuple of two ''NumPy'' arrays ('cp_x', 'cp_y') which are copies of original
            input arrays with 'NaNs' inserted where either 'xmatrix' or 'ymatrix' contain 'NaN'

        Examples
        ----------
        >>> import numpy as np
        >>> x = np.array([1,0, 2.0, np.nan, 4.0])
        >>> y = np.array([np.nan, 5.0, 3.0, 8.0])
        >>> cp_x, cp_y = Operation._mulvar_helper(x, y)

        .. note::
            Copies ('cp_x', 'cp_y') are created using 'np.copy' to avoid modifying the original input arrays.
            ''NaN'' values are identified using ''np.copyto()'' with the ''where'' argument.
        """
        nan_pos = np.isnan(xmatrix) | np.isnan(ymatrix)
        cp_x, cp_y = xmatrix.copy(), ymatrix.copy()
        np.copyto(cp_x, np.nan, where=nan_pos)
        np.copyto(cp_y, np.nan, where=nan_pos)
        return cp_x, cp_y
    
    @classmethod
    def log(cls, xmatrix: npt.NDArray[Any]) -> npt.NDArray[Any]:
        """Return the natural logarithm of one plus the input array, element-wise, with
        handling of ``Inf`` values.

        Calculates ``log(1 + x)`` based on: doc:`numpy.log1p`.

        .. note::
            For real-valued input, ``log1p`` is accurate also for ``x`` so small that ``1 + x == 1`` in floating-point accuracy.

            Logarithm is a multivalued function: for each ``x`` there is an infinite number of complex numbers ``z`` such that ``exp(z) = x``.

            For real-valued input data types, ``log1p`` always returns real output. For each value that cannot be expressed as a real number or infinity, it yields ``nan`` and sets the invalid floating point error flag (see ``numpy.seterr``).
            
        Parameters
        ----------
        xmatrix : array_like
            The input matrix x.

        Returns
        ----------
        ndarray
            The natural logarithm of ``1 + x``, element-wise.

        See Also
        ----------
        numpy.log1p : Equivalent function in NumPy.

        Example
        ----------
        >>> import numpy as np
        >>> xmatrix = np.array([1e-99, np.nan], [np.nan, np.inf])

        >>> Operation.log(xmatrix)
        array([[1e-99, nan],
               [nan, nan]])

        >>> np.log1p(xmatrix)
        array([[1e-99, nan],
               [nan, inf]])

        >>> np.log(1+xmatrix)
        array([[0., nan],
               [nan, inf]])  
        """
        xmatrix = np.log1p(np.abs(xmatrix)) * np.abs(xmatrix) / xmatrix
        xmatrix[np.isinf(xmatrix)] = np.nan
        return xmatrix
    
    @classmethod
    def add_log(
        cls, xmatrix: npt.NDArray[Any], ymatrix: npt.NDArray[Any]
    ) -> npt.NDArray[Any]:
        """Calculate the element-wise natural logarithm of the sum of two array-like objects.
        
        This method takes two arrays of the same shape and computes the natural logarithm of each element in both arrays, and returns their element-wise sum.

        Parameters
        ----------
        xmatrix : array_like
            The input matrix x.
        
        ymatrix : array_like
            The input matrix y.
        
        Returns
        ----------
        ndarray
            The element-wise sum of the natural logarithm of the input arrays.
        """
        res = Operation.log(xmatrix) + Operation.log(ymatrix)
        return res
    
    @classmethod
    def sub_log(
        cls, xmatrix: npt.NDArray[Any], ymatrix: npt.NDArray[Any]
    ) -> npt.NDArray[Any]:
        """Calculate the element-wise natural logarithm of the difference of two array-like objects.
        
        This method takes two arrays of the same shape and computes the natural logarithm of each element in both arrays, and returns their element-wise difference.

        Parameters
        ----------
        xmatrix : array_like
            The input matrix x.
        
        ymatrix : array_like
            The input matrix y.
        
        Returns
        ----------
        ndarray
            The element-wise difference of the natural logarithm of the input arrays.
        """
        res = Operation.log(xmatrix) - Operation.log(ymatrix)
        return res
    
    @classmethod
    def scale(cls, xmatrix: npt.NDArray[Any], a: Union[int, float]) -> npt.NDArray[Any]:
        """Calculate the element-wise scaling of input array by a given factor.

        Scale the input array by a given factor ``a``, such that the result is the element-wise division of the input array by the sum of the absolute values in each row, ignoring ``NaNs``.

        Parameters
        ----------
        xmatrix : array_like
            The input matrix x.
        
        a : int or float
            The scaling factor.
        
        Returns
        ----------
        ndarray
            The scaled matrix with the same dimensions as ``xmatrix``.
        
        Examples
        ----------
        >>> import numpy as np
        >>> xmatrix = np.array([[1, -2, np.nan], [3, 4, 5]])
        >>> a = 2
        >>> Operation.scale(xmatrix, a)
        array([[0.6666667, -1.3333333,      nan],
               [0.5      ,  0.6666667, 0.833333]])
        """
        # Transpose xmatrix to make division operate row-wise
        transposed_matrix = xmatrix.T
        # Calculate the sum of the absolute values in each row
        sum_abs = np.nansum(np.abs(xmatrix), axis=1)
        # Perform row-wise division and transpose back to original shape
        scaled_matrix = np.divide(transposed_matrix, sum_abs).T
        return a*scaled_matrix

    @classmethod
    def signed_power(
        cls, xmatrix: npt.NDArray[Any], exp: Union[int, float]
    ) -> npt.NDArray[Any]:
        """Get signed exponentiation power of array-like object, element-wise(binary operator pow).

        This function takes the sign of the element and multiplies it by the element raised to the power of ``exp``.

        Parameters
        ----------
        xmatrix : array_like
            The input matrix x.
        
        exp : int or float
            The exponent to which each element's absolute value should be raised.

        Returns
        ----------
        ndarray
            The signed exponentiation power of the input array.
        
        Examples
        ----------
        >>> import numpy as np
        >>> xmatrix = np.array([[1, -2, 3], [4, -5, -6]])
        >>> exp = 2
        >>> Operation.signed_power(xmatrix, exp)
        array([[ 1, -4,  9],
               [16, -25, -36]])
        
        >>> Operation.signed_power(xmatrix, 0.5)
        array([[ 1.        , -1.41421356,  1.73205081],
               [ 2.        , -2.23606798, -2.44948974]])
        """
        return np.sign(xmatrix) * np.power(np.abs(xmatrix), exp)
    
    @classmethod
    def shift(
        cls, xmatrix: npt.NDArray[Any], periods: int, fill_value=None
    ) -> npt.NDArray[Any]:
        """Shift index by desired number of periods with an optional `fill value`.

        Only the data is moved, while the index (axis labels) remains the same. This is consistent with the behavior of pandas.DataFrame.shift.

        Values shifted from beyond array bounds will appear at one end of the dimension, which are filled according to `fill_value`.

        Parameters
        ----------
        xmatrix : array_like
            The input matrix x.
        
        periods : int
            Number of periods to shift. Can be positive or negative.
        
        fill_value : scalar, optional
            The scalar value to use for newly introduced missing values. By default, `None` is used.

        Returns
        ----------
        ndarray
            copy of input object, shifted.

        Examples
        ----------
        >>> import numpy as np
        >>> xmatrix = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        >>> periods = 1
        >>> Operation.shift(xmatrix, periods)
        array([[nan, nan, nan],
               [1. , 2, , 3. ],
               [4. , 5. , 6. ]])

        >>> xmatrix = np.array([[1, 2, 3], [np.nan, 5, np.nan], [7, 8, 9]])
        >>> periods = -2
        >>> fill_value = 0
        >>> Operation.shift(xmatrix, periods, fill_value)
        array([[7., 8., 9.],
               [0., 0., 0.],
               [0., 0., 0.]]) 
        """
        return (
            pd.DataFrame(xmatrix).shift(periods=periods, fill_value=fill_value).values
        )
    
    @classmethod
    def diff(
        cls, xmatrix: npt.NDArray[Any], periods: int, fill_value=None
    ) -> npt.NDArray[Any]:
        """`N-th` discrete difference of element.

        Calculate the `N-th` discrete difference along the date dimension.

        .. warning::
            `fill_value` is to fill the missing values produced by the shift operation, not the ones in the final result.
        
        Parameters
        ----------
        xmatrix : array_like
            The input matrix x.
        
        periods : int
            Periods to shift for calculating the difference, accepts both positive and negative integers.
        
        fill_value : scalar, optional
            Value to use for newly missing values. By default None.
        
        Returns
        ----------
        ndarray
            The `N-th` discrete difference of the input array.
        
        Examples
        ----------
        >>> import numpy as np
        >>> xmatrix = np.array([[1, 4, 7], [2, 5, 8], [3, 6, 9]])
        >>> periods = 1
        >>> Operation.diff(xmatrix, periods)
        array([[nan, nan, nan],
               [1. , 1. , 1. ],
               [1. , 1. , 1. ]])
        
        >>> periods = -1
        >>> fill_value = 0
        >>> Operation.diff(xmatrix, periods, fill_value)
        array([[-1., -1., -1.],
               [-1., -1., -1.],
               [ 3.,  6.,  9]])
        """
        res = xmatrix - Operation.shift(xmatrix, periods, fill_value=fill_value)
        return res
    
    @classmethod
    def rolling_max(
        cls, xmatrix: npt.NDArray[Any], window_size: int, minobs: int = 1
    ) -> npt.NDArray[Any]:
        """Calculate the rolling maximum of the input array.

        This method utilizes the ``xarray`` library to handle the rolling window computation. It first converts the input array to an ``xarray.DataArray`` with dimensions ``date`` and ``ticker``. Then calculates the rolling maximum across the ``date`` dimension, requiring a minimum number of observations as specified by ``minobs``.

        Parameters
        ----------
        xmatrix : array_like
            The input matrix x.
        
        window_size : int
            The size of the rolling window.
        
        minobs : int, optional
            The minimum number of observations required for each window. By default 1.
        
        Returns
        ----------
        ndarray
            The rolling maximum of the input array.
        
        Examples
        ----------
        >>> import numpy as np
        >>> xmatrix = np.array([[1, 2, 3], [np.nan, 5, np.nan], [7, 8, 9], [10, 11, 12]])
        >>> window_size = 2
        >>> Operation.rolling_max(xmatrix, window_size)
        array([[ 1.,  2.,  3.],
               [ 1.,  5.,  3.],
               [ 7.,  8.,  9.],
               [10., 11., 12.]])
        
        >>> xmatrix = np.array([[np.nan, np.nan, np.nan], [4, np.nan, 6], [7, 8, 9], [10, 11, 12]])
        >>> window_size = 3
        >>> minobs = 2
        >>> Operation.rolling_max(xmatrix, window_size, minobs)
        array([[nan, nan, nan],
               [nan, nan, nan],
               [ 7., nan,  9.],
               [10., 11., 12.]])
        """
        dax = xr.DataArray(xmatrix, dims=["date", "ticker"])
        res = dax.rolling(date=window_size, min_periods=minobs).max().to_numpy()
        return res
    
    @classmethod
    def rolling_min(
        cls, xmatrix: npt.NDArray[Any], window_size: int, minobs: int = 1
    ) -> npt.NDArray[Any]:
        """Calculate the rolling minimum of the input array.

        This method utilizes the ``xarray`` library to handle the rolling window computation. It first converts the input array to an ``xarray.DataArray`` with dimensions ``date`` and ``ticker``. Then calculates the rolling maximum across the ``date`` dimension, requiring a minimum number of observations as specified by ``minobs``.

        Parameters
        ----------
        xmatrix : array_like
            The input matrix x.
        
        window_size : int
            The size of the rolling window.
        
        minobs : int, optional
            The minimum number of observations required for each window. By default 1.
        
        Returns
        ----------
        ndarray
            The rolling minimum of the input array.
        
        Examples
        ----------
        >>> import numpy as np
        >>> xmatrix = np.array([[1, 2, 3], [np.nan, 5, np.nan], [7, 8, 9], [10, 11, 12]])
        >>> window_size = 2
        >>> Operation.rolling_max(xmatrix, window_size)
        array([[ 1.,  2.,  3.],
               [ 1.,  2.,  3.],
               [ 7.,  5.,  9.],
               [ 7.,  8.,  9.]])
        
        >>> xmatrix = np.array([[np.nan, np.nan, np.nan], [4, np.nan, 6], [7, 8, 9], [10, 11, 12]])
        >>> window_size = 3
        >>> minobs = 2
        >>> Operation.rolling_max(xmatrix, window_size, minobs)
        array([[nan, nan, nan],
               [nan, nan, nan],
               [ 4., nan,  6.],
               [ 4.,  8.,  6.]])
        """
        dax = xr.DataArray(xmatrix, dims=["date", "ticker"])
        res = dax.rolling(date=window_size, min_periods=minobs).min().to_numpy()
        return res
    
    @classmethod
    def rolling_argmax(
        cls, xmatrix: npt.NDArray[Any], window_size: int, minobs: int = 1
    ) -> npt.NDArray[Any]:
        """Calculate the index of the maximum value whitin the window period.

        This method utilizes the ``xarray`` library to handle the rolling window computation. It first converts the input array to an ``xarray.DataArray`` with dimensions ``date`` and ``ticker``. Then calculates the rolling maximum across the ``date`` dimension, requiring a minimum number of observations as specified by ``minobs``.

        Parameters
        ----------
        xmatrix : array_like
            The input matrix x.
        window_size : int
            The size of the rolling window.
        minobs : int, optional
            The minimum number of observations required for each window. By default 1.
        
        Returns
        ----------
        ndarray
            The resulting numpy array with the rolling argmax values applied
        
        Examples
        ----------
        >>> import numpy as np
        >>> xmatrix = np.array([[1, 2, 3], [np.nan, 5, np.nan], [7, 8, 9], [10, 11, 12]])
        >>> window_size = 2
        >>> Operation.rolling_argmax(xmatrix, window_size)
        array([[ 0.,  0.,  0.],
               [ 1.,  0.,  1.],
               [ 0.,  0.,  0.],
               [ 0.,  0.,  0.]])
        
        >>> xmatrix = np.array([[np.nan, np.nan, np.nan], [4, np.nan, 6], [7, 8, 9], [10, 11, 12]])
        >>> window_size = 3
        >>> minobs = 2
        >>> Operation.rolling_argmax(xmatrix, window_size, minobs)
        array([[nan, nan, nan],
               [nan, nan, nan],
               [ 2., nan,  2.],
               [ 2.,  2.,  2.]])
        """

        dax = xr.DataArray(xmatrix, dims=["date", "ticker"])
        return dax.rolling(date=window_size, min_periods=minobs).argmax().to_numpy()
    
    @classmethod
    def rolling_argmin(
        cls, xmatrix: npt.NDArray[Any], window_size: int, minobs: int = 1
    ) -> npt.NDArray[Any]:
        """Calculate the index of the minimum value whitin the window period

        This method utilizes the ``xarray`` library to handle the rolling window computation. It first converts the input array to an ``xarray.DataArray`` with dimensions ``date`` and ``ticker``. Then calculates the rolling maximum across the ``date`` dimension, requiring a minimum number of observations as specified by ``minobs``.

        Parameters
        ----------
        xmatrix : array_like
            The input matrix x.
        window_size : int
            The size of the rolling window.
        minobs : int, optional
            The minimum number of observations required for each window. By default 1.
        
        Returns
        ----------
        ndarray
            The resulting numpy array with the rolling argmin values applied
        
        Examples
        ----------
        >>> import numpy as np
        >>> xmatrix = np.array([[1, 2, 3], [np.nan, 5, np.nan], [7, 8, 9], [10, 11, 12]])
        >>> window_size = 2
        >>> Operation.rolling_argmin(xmatrix, window_size)
        array([[ 1.,  1.,  1.],
               [ 0.,  0.,  0.],
               [ 1.,  0.,  1.],
               [ 0.,  0.,  0.]])
        
        >>> xmatrix = np.array([[np.nan, np.nan, np.nan], [4, np.nan, 6], [7, 8, 9], [10, 11, 12]])
        >>> window_size = 3
        >>> minobs = 2
        >>> Operation.rolling_argmin(xmatrix, window_size, minobs)
        array([[nan, nan, nan],
               [nan, nan, nan],
               [ 1., nan,  1.],
               [ 0.,  1.,  0.]])

        """
        dax = xr.DataArray(xmatrix, dims=["date", "ticker"])
        return dax.rolling(date=window_size, min_periods=minobs).argmin().to_numpy()
    