Getting Started
===============

Installation
------------
To install Pyphotonics, run the following commands:

.. code-block:: console

  git clone git@github.com:rohanku/pyphotonics.git
  cd pyphotonics
  pip install -e .

Certain functions may not work as intended if the necessary external programs are not correctly installed. Pyphotonics requires the Klayout binary to be added to PATH for layout viewing purposes, and Lumerical should be installed at the default location or have its ``lumapi`` Python library added to PYTHONPATH for simulations.

Modules
-------

Pyphotonics has two main modules:

:ref:`Layout <Layout>`
  Functions for automatically generating GDS layouts.

:ref:`Simulation <Simulation>`
  Functions for automated device characterization with Lumerical simulations.

