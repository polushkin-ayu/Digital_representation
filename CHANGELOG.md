# Changelog

## coord-digit-v02 — 2026-11-02

The figure-generation pipeline now incorporates the final audit already
present in `sdp-frontier/data/audit_final.csv`.

- `sdp-frontier/figure_data.py` substitutes final recomputed energies and
  phase currents at matching sweep points. Unmatched points retain their
  original sweep values. If the audit file is absent, the generator uses
  the sweep files alone.
- `figures/numerics_frontier.tex` and `figures/numerics_crossover.tex` have
  been regenerated using this selection rule.
- The numerical README now points to the final audit for achieved solver
  diagnostics and explains the figure-data selection. It distinguishes
  observed shifts under further iteration from certified solver errors.

No solver code, numerical CSV data, or optimisation results were changed,
and no new solver runs are included in this snapshot. The generator
reproduces the committed figure snippets from the stored data.

The final audit contains 33 entries: 24 with status `optimal` and nine
with status `optimal_inaccurate`. Original sweep points outside the audit
do not have individual solver statuses in the sweep CSVs. These finite
SDP calculations are numerical illustrations, not certified bounds for
the infinite-dimensional optimum. The crossover data still use only
the Fourier cutoff N = 16.

## coord-digit-v01

Previous archived snapshot, DOI
[10.5281/zenodo.23807777](https://doi.org/10.5281/zenodo.23807777).
It contains the solver code, the original sweeps, and the final audit.
Its figure generator uses the original sweeps without substituting the
final audit values. The data are unchanged in `coord-digit-v02`.
