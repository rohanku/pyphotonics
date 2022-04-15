# Pyphotonics

Tools for automated photonic circuit design. For more information, check out the [documentation](https://pyphotonics.readthedocs.io/en/latest/index.html).

## Setup

To install Pyphotonics, run the following commands:
```Bash
git clone https://github.com/rohanku/pyphotonics.git
cd pyphotonics
pip install -e .
```

Certain functions may not work as intended if the necessary external programs are not correctly installed. Pyphotonics requires the Klayout binary to be added to PATH for layout viewing purposes, and Lumerical should be installed at the default location or have its `lumapi` Python library added to PYTHONPATH for simulations.

## Features to add

- Ridge waveguides and tapers
- Saving and opening .route files
- Fix cursor line bug (start point does not change at the appropriate time)
