# Digital_representation
Code for numerical checks of digital representations of quantum observables.

The first note's energy–error asymptotics are checked in `numerical-check/`:

- `check.py`, `validate.py`: one-bit quadrature, Fourier minimization, spectral convergence and derivative checks.
- `check_two_digits.py`, `verify_two_digits.py`: two-bit construction and Fourier/Parseval checks.
- `summarize.py`, `export_tex.py`: regenerate the [detailed report](numerical-check/README.md) and PGFPlots figure/table in `figures/`.
- JSON files contain the numerical results. These finite checks support the analytical results; they do not replace proofs.

## Run

Python 3.11+ and NumPy are required (reference version: 2.3.5; no SciPy).

```sh
python3 -m pip install -r numerical-check/requirements.txt
cd numerical-check
OPENBLAS_NUM_THREADS=1 python3 check.py
python3 validate.py
python3 check_two_digits.py
python3 verify_two_digits.py
python3 summarize.py
python3 export_tex.py
```

Scripts overwrite their result files. The exported `.tex` files are snippets for a LaTeX document using PGFPlots (including its `groupplots` library).
