# Binary position readout on a ring: computational materials

Companion to *Momentum Transfer and Energy Balance in a Binary Position Measurement on a Ring*.
Package v1, 23 November 2026. Distributed with the repository release
[coord-digit-v03](https://github.com/polushkin-ayu/Digital_representation/releases/tag/coord-digit-v03).
The versioned Zenodo DOI is listed in the repository release after archival.

## Contents and scope

- `data/`: original solver logs, input grids and batch jobs, root-search iterations,
  and saved diagnostics. These are preserved results of the original calculation.
- `simulate.py`: finite-Fourier-cutoff SDP solver (CVXPY, Clarabel or SCS).
- `roots.py`: direct SDP root searches for the local zero-energy boundary.
- `reproduce.py`: reconstructs the selected grid and all 35 numerical points used
  in Figures 6–9, checks them against the literal coordinates of those figures,
  and writes CSV files and `build/verification.json`. It performs no optimization.
- `figures/`: standalone TikZ/PGFPlots copies of the current English article's
  Figures 6–9, including their analytic curves and the exact threshold segment.
- `check_solver.py`: recomputes one N=24 grid point and compares it to the stored run.
- `build/`: included results of the reproduction and solver checks.
- `SHA256SUMS`: integrity manifest for the archive contents.

All energies are in E_kappa units and momenta in Delta p units. The dimensionless
input mean on the fundamental interval is a in [0, 1/2]. For arbitrary mu,
E(epsilon,mu)/E_kappa = E(epsilon,a)/E_kappa + a^2 - mu^2,
a = distance(mu, Z). epsilon_star = 1/2 - 1/pi; B = 1/2 - 4/pi^2.

## Quick reproduction from saved data

Python 3.12 was used for the checked environment; exact package versions are in
`requirements.txt`. Create an isolated environment:

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python reproduce.py
.venv/bin/python check_solver.py
```

The first command rebuilds derived data without invoking a solver. The second
runs one small independent numerical check; it is not a repeat of the full grid.
The saved numerical libraries may produce slightly different residuals on other
platforms. Requested solver tolerances are not certified accuracies.

To compile the four figures, run, for example:

```sh
cd figures
tectonic 06-threshold.tex
tectonic 07-detuning.tex
tectonic 08-boundary.tex
tectonic 09-global-threshold.tex
```

Tectonic (or a LaTeX installation with TikZ and PGFPlots 1.18) is needed only for
figure PDFs. Their numerical coordinates are validated by `reproduce.py`.
Reference formula numbers and manuscript captions are intentionally not embedded
in standalone figures; see the article for definitions and interpretation.

## Full recomputation (potentially long)

Preserve `data/`. Create a fresh work directory with grid and job specifications:

```sh
mkdir rerun
cp data/grid.json data/*-jobs.json rerun/
RING_DATA_DIR=rerun .venv/bin/python simulate.py --mode map --workers 4
RING_DATA_DIR=rerun .venv/bin/python simulate.py --mode repair --workers 1
RING_DATA_DIR=rerun .venv/bin/python simulate.py --mode refine --workers 2
RING_DATA_DIR=rerun .venv/bin/python roots.py
```

Runners append results and resume by skipping existing (N,mu,epsilon) entries.
Remove a failed entry from the *rerun* log before retrying that job. Raw logs in
`data/` are not overwritten. Original repair jobs target the original run;
different machines may require additional repairs to satisfy selection criteria.

`reproduce.py --data-dir rerun --out rerun-derived` deliberately compares newly
computed numbers with the archived baseline. To enable that comparison, copy
`data/map-data.npz` and `data/audit-summary.json` into `rerun/`. Its strict baseline
assertions can fail for a legitimate new solve: inspect residuals and differences
rather than editing solver output to force agreement. `reproduce.py` is primarily
an exact archival reproduction check, not an acceptance test for a new solver.

## Selection rules and limitations

Results are read in order map, repair, refine; the last record passing the
residual checks is selected for each (N,a,epsilon). The local-law points use map
and repair only, matching the article's plotting history. Thresholds: objective
gap < 2e-5, primal and dual minimum eigenvalues > -2e-7, normalization residual
and error violation < 2e-7. The historical map applies max(0,energy) after
selection; **no selected energy is negative**, so no clipping changes its data.

The N=24 surface contains 435 computed values and 33 exact zero entries. Of the
435, 371 retain `optimal_inaccurate` status. All statuses and residuals are
preserved. The N=16 comparison and N=32,40 refinements are diagnostics, not
rigorous infinite-dimensional error certificates. Root-search residuals are
retained even where the iteration limit prevented reaching the nominal target.

The global threshold is obtained from PCHIP interpolation of the N=24 surface.
The epsilon=0.04 estimate violates a rigorous upper bound and is **excluded from
Figure 9**; `build/excluded-threshold.csv` records both values. It is not silently
clipped or replaced. The local boundary in Figure 8 uses direct root solves.
No numerical claims are made for epsilon < 0.04; the high-accuracy law is proved
analytically in Appendix H, not fitted from these simulations.

## Provenance

Raw data originate in the project's `output/measurement-map/`. The SDP and root
algorithms are preserved; only configurable data-directory handling was added.
Old plotting scripts that modified manuscript sources were not included. The
new reconstruction script is read-only with respect to the raw data. This is a
separate supplement for the physical article, not a replacement for the earlier
JPA/JMP computational deposit.
