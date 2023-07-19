========
Overview
========

An example package. Generated with cookiecutter-pylibrary.

Installation
============

::

    pip install chinese-verdict-sheet-nlp

Usage
=============

To use the project:

.. code-block:: python

    import chinese_verdict_sheet_nlp
    chinese_verdict_sheet_nlp()

Development
===========

To run all the steps, you can execute the "tox" command. This command will perform the following steps::

    tox

.. list-table::
    :widths: 10 90
    :stub-columns: 1

    - - clean
      - remove build artifacts
    - - check
      - check coding style with flake8, isort
    - - docs
      - generate Sphinx HTML documentation, including API docs
    - - py38, py39, py310
      - run tests with the specified Python version
    - - report
      - generate coverage report with the specified Python version



You can also execute specific steps individually.

unit test
----------

To run tests, execute the following command::

    tox -e py38

To run tests and generate coverage report, execute the following command::

    tox -e report

A code coverage report will be generated and saved as *htmlcov/index.html.*

Note, to combine the coverage data from all the tox environments run:

.. list-table::
    :widths: 10 90
    :stub-columns: 1

    - - Windows
      - ::

            set PYTEST_ADDOPTS=--cov-append
            tox

    - - Other
      - ::

            PYTEST_ADDOPTS=--cov-append tox


check coding style
-------------------
To run check coding style, execute the following command::

    tox -e check

build documentation
---------------------
To run build documentation, execute the following command::

    tox -e docs

A documentation will be generated and saved as dist/docs directory.
