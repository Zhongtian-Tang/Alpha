.. currentmodule:: operation
=========
Operation
=========
Operations form the cornerstone of the Alphas library, serving to define the mathematical manipulations applied to the data.

Helper functions
------------------------

.. autosummary::
   :toctree: api/

   Operation._mulvar_helper

Element-wise operations
------------------------

.. autosummary::
   :toctree: api/

   Operation.log
   Operation.scale
   Operation.signed_power
   Operation.shift
   Operation.diff

Pair-wise operations
------------------------
.. autosummary::
   :toctree: api/

   Operation.add_log
   Operation.sub_log

Rolling window operations
--------------------------
.. note::
      See `Windowing Operations <https://pandas.pydata.org/docs/reference/window.html#api-functions-rolling>`__ for further usage details and examples.


.. autosummary::
   :toctree: api/

   Operation.rolling_max
   Operation.rolling_min
   Operation.rolling_argmax
   Operation.rolling_argmin
   Operation.rolling_to_max
   Operation.rolling_to_min
   Operation.rolling_mean
   Operation.rolling_sum
   Operation.rolling_median
   Operation.rolling_std
   Operation.rolling_var
   Operation.rolling_prod
   Operation.rolling_rank

Statistics operations
------------------------
.. autosummary::
   :toctree: api/

   Operation.count_if
   Operation.sum_if
   Operation.skew
   Operation.kurt
   Operation.co_skew
   Operation.co_kurt
   Operation.cov
   Operation.corr
   Operation.linear_decay
   Operation.linear_regression

Cross-sectional operations
------------------------
.. autosummary:: 
   :toctree: api/

   Operation.cs_rank
   Operation.cs_mean_spilt
   Operation.cs_reg_resi

"""


