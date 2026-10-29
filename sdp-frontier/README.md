# Global energy–error frontier by semidefinite programming

These scripts compute the one-digit energy–error frontier of a coordinate readout on a ring
without assuming the analytic optimisers. They support Section "Numerical frontier" of
A. Yu. Polushkin, *Reading coordinate digits on a ring: multiscale energy cost and circulation detuning*.

## Method

An instrument with outcomes a = 0, 1 and any number of unrecorded channels is represented by the
positive operators T_a = Σ_r |k_ar⟩⟨k_ar| acting on the Fourier coefficients of its Kraus functions.
In these variables:

- Kraus normalisation Σ_{a,r} |k_ar|² = 1 fixes the sum of every Fourier diagonal of T_0 + T_1;
- the first-digit error and the kinetic-energy change ΔE/E_κ = Σ l(l + 2μ)|c_l|² are linear in T_a.

The minimal energy at a prescribed error is therefore a semidefinite programme. Restricting the
Kraus functions to the modes |l| ≤ N gives a finite programme, solved with SCS through CVXPY.
Each truncated optimum is attained by an admissible instrument, so it is an **upper bound** on
the exact infimum. Comparing N = 16 with N = 24 measures the truncation error. These finite
calculations illustrate and test the analytical results; they do not replace the proofs.

Two sweeps are computed:

| Part | Content | Truncation | SCS tolerance |
|---|---|---|---|
| `global` | E(ε) for ε = 0.04, 0.06, …, 0.50 at μ = 1/2 and μ = 0 | N = 16 and 24 | 1e-8 |
| `crossover` | M(δ, ν) + ν and the optimiser's momentum transfer for δ = ζ\|ν\|^(2/3), ζ = −1.5, …, 2, at ν = ±10⁻³, 4·10⁻³, 1.6·10⁻² | N = 16 | 1e-10 |

The crossover values are of order 10⁻⁵–10⁻³, hence the tighter tolerance. The interior-point
solver Clarabel returned values below the proved lower bound on this problem, so it is not used.

## Output

`data/frontier_global.csv` has the columns `N, mu, eps, dE_over_Ek`.

`data/crossover.csv` has the columns `omega, nu, zeta, delta, M_plus_nu, G, two_p, minus_sgn_z`.
Here `G` is the analytic leading value 𝒢(δ, |ν|), `two_p` is twice the phase current of the
numerical optimiser in the antiperiodic gauge, and `minus_sgn_z` is its leading prediction −sgn(ν) z.

## Run

Python 3.9 or later; the reference run used Python 3.9.6 and the versions in `requirements.txt`.

```sh
python3 -m pip install -r sdp-frontier/requirements.txt
python3 sdp-frontier/sdp_frontier.py                 # all points, 6 worker processes
python3 sdp-frontier/sdp_frontier.py crossover workers=4
python3 sdp-frontier/figure_data.py                   # writes figures/numerics_*.tex
```

Results are appended and flushed point by point, and a rerun skips the points already present,
so an interrupted sweep can simply be restarted. Points with an active error constraint take
minutes each with SCS at tolerance 1e-10.

`figure_data.py` writes `figures/numerics_frontier.tex` and `figures/numerics_crossover.tex`
as `tikzpicture` snippets with the numerical points embedded as inline tables; the analytic
curves Φ(ζ), σ*(ζ), δ²/B and 1 + c₃δ are drawn by PGFPlots. The snippets need `pgfplots`
with the `groupplots` library, `amsmath` and a `\sgn` operator. With `--standalone` the
script writes complete documents for the manuscript build instead.
