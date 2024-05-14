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
