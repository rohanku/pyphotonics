# Pyphotonics

Tools for automated photonic circuit design.

## Setup

To install Pyphotonics, run the following commands:
```Bash
git clone git@github.com:rohanku/pyphotonics.git
cd pyphotonics
pip install -e .
```

## Requirements

Certain functions may not work as intended if the necessary external programs are not correctly installed. Pyphotonics requires the Klayout binary to be added to PATH for layout viewing purposes, and Lumerical should be installed at the default location or have its `lumapi` Python library added to PYTHONPATH for simulations.
