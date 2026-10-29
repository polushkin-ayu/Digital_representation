# Digital_representation
Code for numerical checks of digital representations of quantum observables.

The first note's energy–error asymptotics are checked in `numerical-check/`:

- `check.py`, `validate.py`: one-bit quadrature, Fourier minimization, spectral convergence and derivative checks.
- `check_two_digits.py`, `verify_two_digits.py`: two-bit construction and Fourier/Parseval checks.
- `check_joint_optimization.py`, `validate_joint_optimization.py`: independent four-channel joint optimization, free mixed-coefficient fits, resolution controls and FFT/gradient checks.
- `summarize.py`, `export_tex.py`: regenerate the [detailed report](numerical-check/README.md) and PGFPlots figure/table in `figures/`.
- JSON files contain the numerical results. These finite checks support the analytical results; they do not replace proofs.

The global one-digit frontier, without assuming the analytic optimisers, is computed in `sdp-frontier/`:

- `sdp_frontier.py`: semidefinite programme over all instruments with Fourier-truncated Kraus functions (any number of unrecorded channels, complex phases); global frontier at mean momentum 1/2 and 0, and the circulation crossover near balance.
- `figure_data.py`: PGFPlots figures in `figures/` from the CSV results in `sdp-frontier/data/`.
- See [`sdp-frontier/README.md`](sdp-frontier/README.md) for the method, output format and run instructions. These scripts need SciPy, CVXPY and SCS in addition to NumPy.

## Run `numerical-check/`

Python 3.11+ and NumPy are required (reference version: 2.3.5; no SciPy).

```sh
python3 -m pip install -r numerical-check/requirements.txt
cd numerical-check
OPENBLAS_NUM_THREADS=1 python3 check.py
python3 validate.py
python3 check_two_digits.py
python3 verify_two_digits.py
python3 check_joint_optimization.py
python3 validate_joint_optimization.py
python3 summarize.py
python3 export_tex.py
```

Scripts overwrite their result files. The exported `.tex` files are snippets for a LaTeX document using PGFPlots (including its `groupplots` library).

## License and citation

The code is released under the MIT License (see `LICENSE`). Citation metadata are in `CITATION.cff`.
