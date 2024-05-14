.. Alpha documentation master file, created by Zhongtian Tang
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to Alpha documentation
*************************
Alpha documentation 📈
*************************

.. image:: https://img.shields.io/badge/python-3.12-%231183c4
   :target: https://www.python.org/downloads/release/python-3111/

.. image:: https://img.shields.io/badge/style-black-black
   :target: https://github.com/psf/black

.. image:: https://img.shields.io/badge/docstyle-numpy-%234e9eb7
   :target: https://numpydoc.readthedocs.io/en/latest/format.html

----------

**Date**: |today|

**Download documentation**: `Zipped HTML <../alphadev.tar.gz>`__

**Useful links**:
`Pandas <https://pandas.pydata.org/docs/index.html>`__ |
`Numpy <https://numpy.org/doc/stable/>`__ |
`Xarray <https://docs.xarray.dev/en/stable/>`__ |

   
   
   The reference guide contains a detailed description of
   the ``Alpha`` API. The reference describes how the methods work and which parameters can
   be used. It assumes that you have an understanding of the key concepts in commonly-used python-based data scientific libraries.


.. grid:: 1 2 2 2
   :gutter: 3
   :padding: 2 2 0 0
   :class-container: sd-text-center

   .. grid-item-card::  Framework Brief
      :columns: 12
      :class-card: intro-card
      :shadow: md

      :mod:`Alpha`, based on `Python <https://www.python.org/>`__ programming language and other scientific computing libraries such as `Pandas <https://pandas.pydata.org/docs/index.html>`__, `Numpy <https://numpy.org/doc/stable/>`__, `Xarray <https://docs.xarray.dev/en/stable/>`__. It is designed to be a flexible and easy-to-use framework for quantitative research. 🚀



   .. grid-item-card:: Operations
      :img-top: _static/operation.svg
      :class-card: intro-card
      :shadow: md

      The Operator Library provides a comprehensive suite of pre-built functionalities for matrix manipulation, including element-wise, rolling window, cross-sectional operations, and more, enhancing efficiency and versatility in data processing for quantitative analysis.

      +++

      .. button-ref:: reference/operation
            :ref-type: doc
            :click-parent:
            :color: secondary
            :expand:

            To the Operation API reference

.. toctree::
   :maxdepth: 3
   :hidden:
   :titlesonly:
   
   reference/operation

Fast Indexing
==================
* :ref:`genindex`

Recent Changes
==================
.. git_changelog::
