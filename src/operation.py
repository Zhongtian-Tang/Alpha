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
               [ 0., nan,  0.],
               [ 0.,  0.,  0.]])
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
        array([[ 0.,  0.,  0.],
               [ 1.,  1.,  1.],
               [ 0.,  1.,  0.],
               [ 1.,  1.,  1.]])
        
        >>> xmatrix = np.array([[np.nan, np.nan, np.nan], [4, np.nan, 6], [7, 8, 9], [10, 11, 12]])
        >>> window_size = 3
        >>> minobs = 2
        >>> Operation.rolling_argmin(xmatrix, window_size, minobs)
        array([[nan, nan, nan],
               [nan, nan, nan],
               [ 1., nan,  1.],
               [ 2.,  1.,  2.]])

        """
        dax = xr.DataArray(xmatrix, dims=["date", "ticker"])
        return dax.rolling(date=window_size, min_periods=minobs).argmin().to_numpy()
    
    @classmethod
    def rolling_to_max(
        cls, xmatrix: npt.NDArray[Any], window_size: int, minobs: int = 1
    ) -> npt.NDArray[Any]:
        """Calculate the number of intervals between the current value and the maximum value within the window period.

        Similar to: func:`rolling_argmax`, but ``rolling_argmax`` returns the ``index`` of the maximum value within the window period, while ``rolling_to_sum`` returns the number of intervals between the current value and the maximum value within the window period.

        .. note::
            According to the documentation of ``bottleneck``, index 0 is at the rightmost edge of the window.

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
            The number of intervals between the current value and the maximum value within the window period.
        
        Examples
        ----------
        >>> import numpy as np
        >>> xmatrix = np.array([[1, 2, 3], [np.nan, 5, np.nan], [7, 8, 9], [10, 11, 12]])
        >>> window_size = 2
        >>> Operation.rolling_to_max(xmatrix, window_size)
        array([[ 0.,  0.,  0.],
               [nan., 0., nan],
               [ 0.,  0.,  0.],
               [ 0.,  0.,  0.]])
        
        >>> xmatrix = np.array([[np.nan, np.nan, np.nan], [4, np.nan, 6], [7, 8, 9], [10, 11, 12]])
        >>> window_size = 3
        >>> minobs = 2
        >>> Operation.rolling_to_max(xmatrix, window_size, minobs)
        array([[nan, nan, nan],
               [nan, nan, nan],
               [ 0., nan,  0.],
               [ 0.,  0.,  0.]])
        """
        res = bn.move_argmax(
            a=xmatrix,
            window=np.min([window_size, xmatrix.shape[0]]),
            min_count=np.max([1, minobs]),
            axis=0
        )
        res[np.isnan(xmatrix)] = np.nan # Set NaN values to NaN
        return res
    
    @classmethod
    def rolling_to_min(
        cls, xmatrix: npt.NDArray[Any], window_size: int, minobs: int = 1
    ) -> npt.NDArray[Any]:
        """Calculate the number of intervals between the current value and the minimum value within the window period.

        Similar to: func:`rolling_argmin`, but ``rolling_argmin`` returns the ``index`` of the minimum value within the window period, while ``rolling_to_min`` returns the number of intervals between the current value and the minimum value within the window period.

        .. note::
            According to the documentation of ``bottleneck``, index 0 is at the rightmost edge of the window.
        
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
            The number of intervals between the current value and the minimum value within the window period.
        
        Examples
        ----------
        >>> import numpy as np
        >>> xmatrix = np.array([[1, 2, 3], [np.nan, 5, np.nan], [7, 8, 9], [10, 11, 12]])
        >>> window_size = 2
        >>> Operation.rolling_to_min(xmatrix, window_size)
        array([[ 0.,  0.,  0.],
               [nan., 1., nan],
               [ 0.,  1.,  0.],
               [ 1.,  1.,  1.]])
        
        >>> xmatrix = np.array([[np.nan, np.nan, np.nan], [4, np.nan, 6], [7, 8, 9], [10, 11, 12]])
        >>> window_size = 3
        >>> minobs = 2
        >>> Operation.rolling_to_min(xmatrix, window_size, minobs)
        array([[nan, nan, nan],
               [nan, nan, nan],
               [ 1., nan,  1.],
               [ 2.,  1.,  2.]])
        """
        res = bn.move_argmin(
            a=xmatrix,
            window=np.min([window_size, xmatrix.shape[0]]),
            min_count=np.max([1, minobs]),
            axis=0
        )
        res[np.isnan(xmatrix)] = np.nan
        return res
    
    @classmethod
    def rolling_mean(
        cls, xmatrix: npt.NDArray[Any], window_size: int, minobs: int = 1
    ) -> npt.NDArray[Any]:
        """Calculate the rolling mean.

        This method utilizes the ``xarray`` library to handle the rolling window computation. It first converts the input array into an ``xarray.DataArray`` with named dimensions ``date`` and ``ticker``. It then calculates the rolling mean across the ``date`` dimension, requiring a minimum number of observations as specified by ``minobs``.

        Parameters
        ----------
        xmatrix : array_like
            The input matrix.
        window_size : int
            Size of the moving window.
        minobs : int, optional
            Minimum number of observations in window required to have a value; otherwise, result is ``np.nan``. By default 1.

        Returns
        -------
        ndarray
            The resulting numpy array with the rolling mean values applied.

        See Also
        --------
        pandas.core.window.rolling.Rolling.mean : Pandas equivalent method.
        xarray.core.rolling.DataArrayRolling.mean : Reduce data windows by applying `mean` along specified dimension.
        xarray.core.rolling.DataArrayRolling : Xarray base class for rolling window operations.

        Examples
        --------
        >>> import numpy as np
        >>> xmatrix = np.array([[1, 2, 3], [np.nan, 5, np.nan], [7, 8, 9], [10, 11, 12]])
        >>> window_size = 2
        >>> Operation.rolling_mean(xmatrix, window_size)
        array([[ 1. ,  2. ,  3. ],
               [ 1. ,  3.5,  3. ],
               [ 7. ,  6.5,  9. ],
               [ 8.5,  9.5, 10.5]])

        >>> xmatrix = np.array([[np.nan, np.nan, np.nan], [4, np.nan, 6], [7, 8, 9], [10, 11, 12]])
        >>> window_size = 3
        >>> minobs = 2
        >>> Operation.rolling_mean(xmatrix, window_size, minobs)
        array([[nan, nan, nan],
               [nan, nan, nan],
               [5.5, nan, 7.5],
               [7. , 9.5, 9. ]])
        """
        # ? basic numpy method: cant define minobs(all rows less than window_size will be set to nan)
        """
        .. note::

            Method 1 - use numpy:

            .. code-block:: python

                res = np.full_like(xmatrix, np.nan)
                roll_mean = np.lib.stride_tricks.sliding_window_view(
                    xmatrix, window_size, axis=0).mean(axis=-1)
                res[window_size - 1 :] = roll_mean
                return res

            Method 2 - use bottleneck:
            Lower performance than xarray.

            .. code-block:: python

                bn.move_mean(xmatrix, window_size, minobs, axis=0)
        """
        dax = xr.DataArray(xmatrix, dims=("date", "ticker"))
        return dax.rolling(date=window_size, min_periods=minobs).mean().to_numpy()
    
    @classmethod
    def rolling_sum(
        cls, xmatrix: npt.NDArray[Any], window_size: int, minobs: int = 1
    ) -> npt.NDArray[Any]:
        """Calculate the rolling sum.

        This method utilizes the ``xarray`` library to handle the rolling window computation. It first converts the input array to an ``xarray.DataArray`` with dimensions ``date`` and ``ticker``. Then calculates the rolling sum across the ``date`` dimension, requiring a minimum number of observations as specified by ``minobs``.

        .. warning::
            Directly using pandas ``rolling.sum`` will sum all ``NaN`` values to 0
        
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
            The resulting numpy array with the rolling sum values applied
        
        Examples
        ----------
        >>> import numpy as np
        >>> xmatrix = np.array([[1, 2, 3], [np.nan, 5, np.nan], [7, 8, 9], [10, 11, 12]])
        >>> window_size = 2
        >>> Operation.rolling_sum(xmatrix, window_size)
        array([[ 2.,  4.,  6.],
               [ 2,   7.,  6.],
               [ 14., 13., 18.],
               [ 17., 19., 21.]])
        
        >>> xmatrix = np.array([[np.nan, np.nan, np.nan], [4, np.nan, 6], [7, 8, 9], [10, 11, 12]])
        >>> window_size = 3
        >>> minobs = 2
        >>> Operation.rolling_sum(xmatrix, window_size, minobs)
        array([[ nan,  nan,  nan],
               [ nan,  nan,  nan],
               [16.5., nan, 22.5],
               [21.  , 28.5, 27.]])

        .. note::
            Treat the nan as the mean value of the window period.
        """
        return window_size * Operation.rolling_mean(xmatrix, window_size, minobs)
    
    @classmethod
    def rolling_median(
        cls, xmatrix: npt.NDArray[Any], window_size: int, minobs: int = 1
    ) -> npt.NDArray[Any]:
        """Calculate the rolling median.

        This method utilizes the ``xarray`` library to handle the rolling window computation. It first converts the input array to an ``xarray.DataArray`` with dimensions ``date`` and ``ticker``. Then calculates the rolling sum across the ``date`` dimension, requiring a minimum number of observations as specified by ``minobs``.

        Paramters
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
            The resulting numpy array with the rolling median values applied
        
        Examples
        ----------
        >>> import numpy as np
        >>> xmatrix = np.array([[1, 2, 3], [np.nan, 5, np.nan], [7, 8, 9], [10, 11, 12]])
        >>> window_size = 2
        >>> Operation.roll_median(xmatrix, window_size)
        array([[ 1.,  2.,  3.],
               [ 1.,  3.5,  3. ],
               [ 7.,  6.5,  9. ],
               [ 8.5,  9.5, 10.5]])
        
        >>> xmatrix = np.array([[np.nan, np.nan, np.nan], [4, np.nan, 6], [7, 8, 9], [10, 11, 12]])
        >>> window_size = 3
        >>> minobs = 2
        >>> Operation.roll_median(xmatrix, window_size, minobs)
        array([[nan, nan, nan],
               [nan, nan, nan],
               [5.5, nan, 7.5],
               [7.,  9.5, 9.]])
        """
        dax = xr.DataArray(xmatrix, dims=["date", "ticker"])
        res = dax.rolling(date=window_size, min_periods=minobs).median().to_numpy()
        return res
    
    @classmethod
    def rolling_std(
        cls, xmatrix: npt.NDArray[Any], window_size: int, minobs: int = 1
    ) -> npt.NDArray[Any]:
        """Calculate the rolling standard deviation.

        This method utilizes the ``xarray`` library to handle the rolling window computation. It first converts the input array to an ``xarray.DataArray`` with dimensions ``date`` and ``ticker``. Then calculates the rolling sum across the ``date`` dimension, requiring a minimum number of observations as specified by ``minobs``.

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
            The resulting numpy array with the rolling standard deviation values applied
        
        Examples
        ----------
        >>> import numpy as np
        >>> xmatrix = np.array([[1, 2, 3], [np.nan, 5, np.nan], [7, 8, 9], [10, 11, 12]])
        >>> window_size = 2
        >>> Operation.rolling_std(xmatrix, window_size)
        array([[0. , 0. , 0. ],
               [0. , 1.5, 0. ],
               [0. , 1.5, 0. ],
               [1.5, 1.5, 1.5]])
        
        >>> xmatrix = np.array([[np.nan, np.nan, np.nan], [4, np.nan, 6], [7, 8, 9], [10, 11, 12]])
        >>> window_size = 3
        >>> minobs = 2
        >>> Operation.roll_std(xmatrix, window_size, minobs)
        array([[   nan,  nan,    nan],
               [   nan,  nan,    nan],
               [   1.5,  nan,    1.5],
               [2.4495,  1.5, 2.4495]])
        """
        dax = xr.DataArray(xmatrix, dims=["date", "ticker"])
        res = dax.rolling(date=window_size, min_periods=minobs).std().to_numpy()
        return res
    
    @classmethod
    def rolling_var(
        cls, xmatrix: npt.NDArray[Any], window_size: int, minobs: int = 1, ddof: Literal[0, 1] = 1
    ) -> npt.NDArray[Any]:
        """Calculate the rolling weighted window variance.

        This method utilizes the ``xarray`` library to handle the rolling window computation. It first converts the input array to an ``xarray.DataArray`` with dimensions ``date`` and ``ticker``. Then calculates the rolling sum across the ``date`` dimension, requiring a minimum number of observations as specified by ``minobs``.

        Parameters
        ----------
        xmatrix : array_like
            The input matrix x.
        window_size : int
            The size of the rolling window.
        minobs : int, optional
            The minimum number of observations required for each window. By default 1.
        ddof : int, optional
            The delta degrees of freedom. The divisor used in calculation is ``N - ddof``, where ``N`` represents the number of non-NaN elements. By default 1.
        
        Returns
        ----------
        ndarray
            The resulting numpy array with the rolling variance values applied
        
        Examples
        ----------
        >>> import numpy as np
        >>> xmatrix = np.array([[1, 2, 3], [np.nan, 5, np.nan], [7, 8, 9], [10, 11, 12]])
        >>> window_size = 2
        >>> Operation.rolling_var(xmatrix, window_size)
        array([[nan , nan , nan],
               [nan , 4.5 , nan],
               [nan , 4.5 , nan],
               [4.5 , 4.5 , 4.5]])

        >>> xmatrix = np.array([[np.nan, np.nan, np.nan], [4, np.nan, 6], [7, 8, 9], [10, 11, 12]])
        >>> window_size = 3
        >>> minobs = 2
        >>> Operation.rolling_var(xmatrix, window_size, minobs)
        array([[nan, nan, nan],
               [nan, nan, nan],
               [4.5, nan, 4.5],
               [9. , 4.5, 9. ]])
        """
        dax = xr.DataArray(xmatrix, dims=["date", "ticker"])
        res = dax.rolling(date=window_size, min_periods=minobs).var(ddof=ddof).to_numpy()
        return res
    
    @classmethod
    def rolling_prod(
        cls, xmatrix: npt.NDArray[Any], window_size: int, minobs: int = 1
    ) -> npt.NDArray[Any]:
        """Calculate the rolling product.

        This method utilizes the ``xarray`` library to handle the rolling window computation. It first converts the input array to an ``xarray.DataArray`` with dimensions ``date`` and ``ticker``. Then calculates the rolling sum across the ``date`` dimension, requiring a minimum number of observations as specified by ``minobs``.

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
            The resulting numpy array with the rolling product values applied
        
        Examples
        ----------
        >>> import numpy as np
        >>> xmatrix = np.array([[1, 2, 3], [np.nan, 5, np.nan], [7, 8, 9], [10, 11, 12]])
        >>> window_size = 2
        >>> Operation.rolling_prod(xmatrix, window_size)
        array([[  1.,   2.,   3.],
               [  1.,  10.,   3.],
               [  7.,  40.,   9.],
               [ 70.,  88., 108.]])

        >>> xmatrix = np.array([[np.nan, np.nan, np.nan], [4, np.nan, 6], [7, 8, 9], [10, 11, 12]])
        >>> window_size = 3
        >>> minobs = 2
        >>> Operation.rolling_prod(xmatrix, window_size, minobs)
        array([[ nan, nan,  nan],
               [ nan, nan,  nan],
               [ 28., nan,  54.],
               [280., 88., 648.]])
        """
        dax = xr.DataArray(xmatrix, dims=["date", "ticker"])
        res = dax.rolling(date=window_size, min_periods=minobs).prod().to_numpy()
        return res
    
    @classmethod
    def count_if(
        cls, condition: npt.NDArray[np.bool_], window_size: int, minobs: int = 1
    ) -> npt.NDArray[Any]:
        """Calculate the count that satisfies the condition within the window period.

        The condition is a boolean matrix, which is the same shape as the input matrix. This method calculates the number of ``True`` values in a rolling window of the specified size.

        Parameters
        ----------
        condition : array_like
            The boolean matrix that represents the condition.
        window_size : int
            The size of the rolling window.
        minobs : int, optional
            The minimum number of observations required for each window. By default 1.
        
        Returns
        ----------
        ndarray
            The resulting numpy array with the count values applied    
        """
        condition[np.isnan(condition)] = 0
        res = Operation.rolling_sum(
            xmatrix=condition, window_size=window_size, minobs=minobs
            )
        return res
    
    @classmethod
    def sum_if(
        cls, xmatrix: npt.NDArray[Any], condition: npt.NDArray[np.bool_], window_size: int, minobs: int = 1
    ) -> npt.NDArray[Any]:
        """Calculate the sum of values that satisfy the condition within the window period.

        The condition is a boolean matrix, which is the same shape as the input matrix. This method calculates the sum of values that satisfy the condition in a rolling window of the specified size.

        Parameters
        ----------
        xmatrix : array_like
            The input matrix x.
        condition : array_like
            The boolean matrix that represents the condition.
        window_size : int
            The size of the rolling window.
        minobs : int, optional
            The minimum number of observations required for each window. By default 1.
        
        Returns
        ----------
        ndarray
            The resulting numpy array with the sum values applied        
        """
        res = Operation.rolling_sum(
            xmatrix=xmatrix * condition, window_size=window_size, minobs=minobs
            )
        return res
    
    @classmethod
    def skew(
        cls, xmatrix: npt.NDArray[Any], window_size: int, minobs: int = 1
    ) -> npt.NDArray[Any]:
        """Calculate the rolling unbiased skewness.

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
            The resulting numpy array with the rolling skewness values applied
        
        See Also
        ----------
        pandas.core.window.rolling.Rolling.skew : Pandas equivalent method.
        """
        x = pd.DataFrame(xmatrix)
        res = x.rolling(window=window_size, min_periods=minobs).skew()
        return res.values
    
    @classmethod
    def kurt(
        cls, xmatrix: npt.NDArray[Any], window_size: int, minobs: int = 1
    ) -> npt.NDArray[Any]:
        """
        Calculate the rolling Fisher's definition of kurtosis with bias
        
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
            The resulting numpy array with the rolling kurtosis values applied
        
        See Also
        ----------
        pandas.core.window.rolling.Rolling.kurt : Pandas equivalent method.
        """
        x = pd.DataFrame(xmatrix)
        res = x.rolling(window=window_size, min_periods=minobs).kurt()
        return res.values
    
    @classmethod
    def co_skew(
        cls, xmatrix: npt.NDArray[Any], ymatrix: npt.NDArray[Any], window_size: int, minobs: int = 2
    ) -> npt.NDArray[Any]:
        """Calculate the rolling co-skewness.

        Parameters
        ----------
        xmatrix : array_like
            The input matrix x.
        ymatrix : array_like
            The input matrix y.
        window_size : int
            The size of the rolling window.
        minobs : int, optional
            The minimum number of observations required for each window. By default 2.
        
        Returns
        ----------
        ndarray
            The resulting numpy array with the rolling coskewness values applied
        
        See Also
        ----------
        pandas.core.window.rolling.Rolling.cov : Pandas equivalent method.
        """
        cp_x, cp_y = Operation._mulvar_helper(xmatrix, ymatrix)
        dm_x = cp_x - Operation.rolling_mean(cp_x, window_size)
        dm_y = cp_y - Operation.rolling_mean(cp_y, window_size)
        numerator = Operation.rolling_mean(
            xmatirx=dm_x * dm_y**2, window_size=window_size
        )
        denominator = Operation.rolling_std(
            cp_x, window_size, minobs=minobs
        ) * Operation.rolling_var(cp_y, window_size, minobs=minobs, ddof=0)
        res = numerator / denominator
        return res
    
    @classmethod
    def co_kurt(
        cls,
        xmatrix: npt.NDArray[Any],
        ymatrix: npt.NDArray[Any],
        window_size: int,
        minobs: int = 2,
    ) -> npt.NDArray[Any]:
        """Calculate the rolling co-kurtosis.

        Parameters
        ----------
        xmatrix : array_like
            The input matrix x.
        ymatrix : array_like
            The input matrix y.
        window_size : int
            The size of the rolling window.
        minobs : int, optional
            The minimum number of observations required for each window. By default 2.
        
        Returns
        ----------
        ndarray
            The cokurtosis of the input arrays. 
        """
        cp_x, cp_y = Operation._mulvar_helper(xmatrix=xmatrix, ymatrix=ymatrix)
        dm_x = cp_x - Operation.rolling_mean(cp_x, window_size)
        dm_y = cp_y - Operation.rolling_mean(cp_y, window_size)
        numerator = Operation.rolling_mean(
            xmatirx=dm_x * dm_y**3, window_size=window_size
        )
        denominator = Operation.rolling_std(
            cp_x, window_size, minobs=minobs
        ) * Operation.rolling_var(cp_y, window_size, minobs=minobs, ddof=0) ** (3 / 2)
        denominator[denominator == 0] = np.nan
        return numerator / denominator
    
    @classmethod
    def cs_rank(cls, xmatrix: npt.NDArray[Any], pct: bool = True) -> npt.NDArray[Any]:
        """Compute numerical data ranks along section axis.

        Equal values are assigned a rank that is the average of the ranks that would have been otherwise assigned to all of the values within that set. Ranks begin at 1, not 0. If `pct`, computes percentage ranks.

        ``NaNs`` in the input array are returned as ``NaNs``

        Parameters
        ----------
        xmatrix : array_like
            The input matrix x.
        pct : bool, optional
            Whether or not to display the returned rankings in percentile form. By default ``True``.
        
        Returns
        ----------
        ndarray
            The ranked value.
        """
        dax = xr.DataArray(xmatrix, dims=["date", "ticker"])
        res = dax.rank(dim="date", pct=pct).to_numpy()
        return res
    
    @classmethod
    def rolling_rank(
        cls, xmatrix: npt.NDArray[Any], window_size: int, minobs: int = 1, pct: bool = True
    ) -> npt.NDArray[Any]:
        """Calculate the rolling rank along date axis

        Returns the rank of the value within the specified window period. If `pct`, returns the percentile rank.

        Parameters
        ----------
        xmatrix : array_like
            The input matrix x.
        window_size : int
            The size of the rolling window.
        minobs : int, optional
            The minimum number of observations required for each window. By default 1.
        pct : bool, optional
            Whether or not to display the returned rankings in percentile form. By default ``True``.
        
        Returns
        ----------
        ndarray
            The resulting numpy array with the rolling rank values applied
        """
        nan_idx = np.isnan(xmatrix)
        res = np.full(xmatrix.shape, np.nan)
        for dumi in range(minobs, xmatrix.shape[0]):
            head = max(dumi - window_size + 1, 0)
            # Ensure the current value is also cut.
            tail = dumi + 1
            val = xmatrix[head:tail, :]
            vrk = np.sum(val <= val[-1, :], axis=0)
            res[dumi, :] = vrk
        if pct:
            valids = 1 - nan_idx.astype(int)
            val_cnt = bn.move_sum(
                a=valids,
                window=np.min([window_size, valids.shape[0]]),
                min_count=np.max([1, minobs]),
                axis=0
            )
            # Avoid division by zero
            val_cnt[val_cnt == 0] = np.nan
            res = res / val_cnt
        # Keep the nan in the original data
        res[nan_idx] = np.nan
        return res

    @classmethod
    def cov(
        cls, xmatrix: npt.NDArray[Any], ymatrix: npt.NDArray[Any], window_size: int, minobs: int = 2
    ) -> npt.NDArray[Any]:
        """Calculate the rolling sample covariance.

        Parameters
        ----------
        xmatrix : array_like
            The input matrix x.
        ymatrix : array_like
            The input matrix y.
        window_size : int
            The size of the rolling window.
        minobs : int, optional
            The minimum number of observations required for each window. By default 2.
        
        Returns
        ----------
        ndarray
            The resulting numpy array with the rolling covariance values applied
        
        See Also
        ----------
        pandas.core.window.rolling.Rolling.cov : Pandas equivalent method.
        """
        cp_x, cp_y = Operation._mulvar_helper(xmatrix, ymatrix)
        res = Operation.rolling_mean(
            xmatrix=cp_x * cp_y, window_size=window_size, minobs=minobs
        ) - Operation.rolling_mean(
            cp_x, window_size, minobs
        ) * Operation.rolling_mean(
            cp_y, window_size, minobs
        )
        return res
    
    @classmethod
    def corr(
        cls, xmatrix: npt.NDArray[Any], ymatrix: npt.NDArray[Any], window_size: int, minobs: int = 2
    ) -> npt.NDArray[Any]:
        """Calculate the rolling correlation

        .. note::
            This function uses the Pearson correlation coefficient. (https://en.wikipedia.org/wiki/Pearson_correlation_coefficient)
        
        Parameters
        ----------
        xmatrix : array_like
            The input matrix x.
        ymatrix : array_like
            The input matrix y.
        window_size : int
            The size of the rolling window.
        minobs : int, optional
            The minimum number of observations required for each window. By default 2.
        
        Returns
        ----------
        ndarray
            The resulting numpy array with the rolling correlation values applied
        
        See Also
        ----------
        pandas.core.window.rolling.Rolling.corr : Pandas equivalent method.
        """
        cp_x, cp_y = Operation._mulvar_helper(xmatrix, ymatrix)
        numerator = Operation.cov(
            xmatrix, ymatrix, window_size, minobs
        )
        denominator = Operation.rolling_std(
            cp_x, window_size, minobs
        ) * Operation.rolling_std(
            cp_y, window_size, minobs
        )
        denominator[denominator == 0] = np.nan
        return numerator / denominator
    
    @classmethod
    def cs_mean_spilt(cls, xmatrix: npt.NDArray[Any]) -> npt.NDArray[np.int_]:
        """Divide into two groups based on the mean value of the input matrix.

        Compare the value of each element with the mean value of the cross section. If the value is greater than the mean value, the value is 1; otherwise, it is -1.
        
        Parameters
        ----------
        xmatrix : array_like
            The input matrix x.
        
        Returns
        ----------
        ndarray
            The resulting numpy array with the split values applied
        """
        mean_x = np.nanmean(xmatrix, axis=1, keepdims=True)
        res = np.where(xmatrix > mean_x, 1, -1)
        return res
    
    @classmethod
    def cs_reg_resi(
        cls, xmatrix: npt.NDArray[Any], ymatrix: npt.NDArray[Any]
    ) -> npt.NDArray[Any]:
        """Calculate the residual of cross section regression(including intercept).

        Parameters
        ----------
        xmatrix : array_like
            The input matrix x.
        ymatrix : array_like
            The input matrix y.
        
        Returns
        ----------
        ndarray
            The resulting numpy array with the residual values applied
        """
        cp_x, cp_y = Operation._mulvar_helper(xmatrix, ymatrix)
        mean_x, mean_y = np.nanmean(cp_x, axis=1), np.nanmean(cp_y, axis=1)
        var_x = np.nanvar(cp_x, axis=1)
        mean_xy = np.nanmean(cp_x * cp_y, axis=1)
        beta = (mean_xy - mean_x * mean_y) / var_x
        res = cp_y - beta * cp_x.reshape([-1, 1])
        return res
    
    @classmethod
    def linear_decay(
        cls, xmatrix: npt.NDArray[Any], window_size: int
    ) -> npt.NDArray[Any]:
        """Calculate the wma (weighted moving average) of the input matrix.

        The weighted average of the time series is ```d, d-1, ..., 1`` (the sum of the weights is 1, which needs to be normalized), where the closer the day is, the greater the weight.
        
        Parameters
        ----------
        xmatrix : array_like
            The input matrix x.
        window_size : int
            The size of the rolling window.
        
        Returns
        ----------
        ndarray
            The resulting numpy array with the linear decay values applied
        """
        cp_x = xmatrix.copy()
        nan_idx = np.isnan(cp_x)
        cp_x[nan_idx] = 0
        adj = (~nan_idx).astype(int)

        res = np.zeros_like(cp_x)
        adj_fct = np.zeros_like(xmatrix, dtype=float)

        for dumi in range(window_size, 0, -1):
            tail = window_size - dumi
            x_shift = np.roll(cp_x, tail, axis=0)
            adj_shift = np.roll(adj, tail, axis=0)
            x_shift[:tail, :] = 0
            adj_shift[:tail, :] = 0
            res += x_shift * dumi
            adj_fct += adj_shift * dumi
        
        adj_fct[adj_fct == 0] = np.nan
        res = res / adj_fct
        res[nan_idx] = np.nan
        return res
    
    # TBD
    """@classmethod
    def exponential_decay()
    """

    @classmethod
    def linear_regression(
        cls,
        xmatrix: npt.NDArray[Any],
        ymatrix: npt.NDArray[Any],
        window_size: int,
        calc_alpha: bool = False,
        calc_sigma: bool = False,
        minobs: int = 2,
    ) -> Union[
        npt.NDArray[Any],
        Tuple[npt.NDArray[Any], ...],
    ]:
        """Calculate the uni-variate linear regression of feature and target matrix. The calculation method is based on the following formula:

        .. math:: y = \\alpha + \\beta * x

        Fit a linear model with coefficients : math: `\omega = (\omega_{1}, ..., \omega_{p})` to minimize the residual sum of squares between the observed targets in the dataset, and the targets predicted by the linear approximation.

        Parameters
        ----------
        xmatrix : array_like
            The input matrix x.
        ymatrix : array_like
            The input matrix y.
        window_size : int
            The size of the rolling window.
        calc_alpha : bool, optional
            Whether to calculate the intercept. By default ``False``.
        calc_sigma : bool, optional
            Whether to calculate the residual sum of squares. By default ``False``.
        minobs : int, optional
            The minimum number of observations required for each window. By default 2.
        
        Returns
        ----------
        Union[ndarray, Tuple[ndarray, ...]]
            The resulting numpy array with the linear regression values applied
        """
        cp_x, cp_y = Operation._mulvar_helper(xmatrix, ymatrix)
        mean_x = Operation.rolling_mean(
            xmatrix=cp_x, window_size=window_size, minobs=minobs
        )
        mean_y = Operation.rolling_mean(
            xmatrix=cp_y, window_size=window_size, minobs=minobs
        )
        numerator = (
            Operation.rolling_mean(cp_x * cp_y, window_size, minobs) - mean_x * mean_y
        )
        denominator = Operation.rolling_var(cp_x, window_size, minobs=minobs, ddof=0)
        denominator[denominator == 0] = np.nan
        beta = numerator / denominator
        
        if calc_alpha or calc_sigma:
            """Need to calculate intercept."""
            alpha = mean_y - mean_x * beta
            if calc_sigma:
                """Need to calculate residual sum of squares / valid data"""
                ss_x = Operation.rolling_mean(
                    xmatrix=cp_x**2, window_size=window_size, minobs=minobs
                )
                ss_y = Operation.rolling_mean(
                    xmatrix=cp_y**2, window_size=window_size, minobs=minobs
                )
                s_xy = Operation.rolling_mean(
                    xmatrix=cp_x * cp_y, window_size=window_size, minobs=minobs 
                )
                sigma = (
                    ss_y 
                    + ss_x * beta **2
                    + alpha**2
                    - 2 * beta * s_xy
                    - 2 * alpha * mean_y
                    + 2 * alpha * beta * mean_x
                )
                return alpha, beta, sigma
            else:
                return alpha, beta
        else:
            return beta
        
        
    