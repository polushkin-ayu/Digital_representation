#!/usr/bin/env python3
"""Convex (SDP) evaluation of the one-digit energy--error frontier on a ring.

Each outcome a of a coordinate instrument is represented by the positive operator
T_a = sum_r |k_ar><k_ar| on Fourier coefficients k_ar = sum_l c_{ar,l} e^{i l phi},
truncated to |l| <= N.  Kraus normalisation sum_{a,r}|k_ar|^2 = 1 is the set of
linear constraints sum_{l-l'=m} (T_0+T_1)[l,l'] = delta_{m0}; the error and the
energy change (spectral formula, Lemma 2.1 of the paper) are linear in T.
Truncation restricts the admissible class, so the exact optimum of the truncated
programme is an upper bound on the infimum over the full class.  SCS returns an
approximation to that optimum, not a certified bound: at the requested tolerances
it may stop with an inaccurate status (e.g. after the iteration limit), and the
requested tolerance is not the achieved accuracy (see audit_points.py).  The
difference between N = 16 and N = 24 is a stability test, not a guaranteed
bound on the truncation error.

Usage (from any directory):
    python3 sdp_frontier.py [outdir] [global] [crossover] [N=16 N=24] [workers=6]
Defaults: outdir = data/ next to this script, both parts, N = 16 and 24.
Points run in parallel and are appended to outdir/frontier_global.csv and
outdir/crossover.csv as soon as they finish; a rerun skips points already there.
"""
import sys, csv, time, warnings
from pathlib import Path
import numpy as np, cvxpy as cp
from scipy.optimize import brentq

# SCS often reports "optimal_inaccurate" at the requested 1e-10 tolerance;
# the values are checked against the analytic laws and between truncations.
warnings.filterwarnings("ignore", message="Solution may be inaccurate")

PI = np.pi
B = 0.5 - 4 / PI**2
EPS = 0.5 - 1 / PI


def s1_coeff(m):
    """<s_1 e^{i m phi}> with s_1 = sign cos phi."""
    if m % 2 == 0:
        return 0.0
    return 2 / (PI * abs(m)) * (-1) ** ((abs(m) - 1) // 2)


class Frontier:
    def __init__(self, N):
        self.N = N
        self.l = np.arange(-N, N + 1)
        n = len(self.l)
        diff = self.l[:, None] - self.l[None, :]
        self.S = np.vectorize(s1_coeff)(diff).astype(float)
        self.T = [cp.Variable((n, n), hermitian=True) for _ in range(2)]
        self.mu = cp.Parameter()
        self.target = cp.Parameter()
        Tsum = self.T[0] + self.T[1]
        cons = [t >> 0 for t in self.T]
        # Hermiticity makes the m and -m conditions conjugate; keep m >= 0 only.
        for m in range(0, 2 * N + 1):
            diag_sum = cp.trace((diff == -m).astype(float) @ Tsum)
            cons.append(cp.real(diag_sum) == (1.0 if m == 0 else 0.0))
            if m:
                cons.append(cp.imag(diag_sum) == 0)
        self.err = 0.5 + cp.real(cp.trace(self.S.T @ self.T[1]))
        cons.append(self.err <= self.target)
        L = self.l.astype(float)
        energy = cp.real(cp.trace(np.diag(L * L) @ Tsum)) + 2 * self.mu * cp.real(cp.trace(np.diag(L) @ Tsum))
        self.prob = cp.Problem(cp.Minimize(energy), cons)

    def solve(self, eps, mu, tol=1e-10):
        """Return (Delta E / E_kappa, phase current p = transfer + 1/2)."""
        self.target.value, self.mu.value = eps, mu
        self.prob.solve(solver=cp.SCS, eps_abs=tol, eps_rel=tol, max_iters=400000)
        if self.prob.status not in ("optimal", "optimal_inaccurate"):
            raise RuntimeError(self.prob.status)
        diag = sum(np.real(np.diag(t.value)) for t in self.T)
        return self.prob.value, float(np.dot(self.l + 0.5, diag))


def z_root(delta, omega):
    if omega == 0:
        return 0.0 if delta >= 0 else np.sqrt(-2 * PI * delta)
    return brentq(lambda z: z**3 + 2 * PI * delta * z - PI**2 * B * omega, 0, 10)


def G(delta, omega):
    z = z_root(delta, omega)
    return (delta + z * z / (2 * PI)) ** 2 / B - omega * z


_CACHE = {}


def _point(job):
    """Solve one point in a worker process; Frontier objects are cached per N."""
    kind, N, eps, mu, tol, meta = job
    if N not in _CACHE:
        _CACHE[N] = Frontier(N)
    val, p = _CACHE[N].solve(eps, mu, tol)
    return kind, meta, val, p


def jobs_global(Ns):
    # values are O(1e-2..1) here, so a looser tolerance suffices
    for N in Ns:
        for mu in (0.5, 0.0):
            for eps in np.round(np.arange(0.04, 0.5001, 0.02), 4):
                yield ("global", N, float(eps), mu, 1e-8, (N, mu, float(eps)))


def jobs_crossover():
    for omega, signs in ((1e-3, (1, -1)), (4e-3, (1,)), (1.6e-2, (1,))):
        for sign in signs:
            for zeta in np.round(np.arange(-1.5, 2.01, 0.5), 3):
                nu = sign * omega
                delta = zeta * omega ** (2 / 3)
                yield ("crossover", 16, EPS - delta, 0.5 + nu, 1e-10, (omega, nu, float(zeta), delta))


def main(out, parts=("global", "crossover"), workers=6, global_N=(16, 24)):
    """Run all points in parallel; each result is appended and flushed at once."""
    from multiprocessing import Pool
    out.mkdir(parents=True, exist_ok=True)
    files = {"global": (out / "frontier_global.csv", ["N", "mu", "eps", "dE_over_Ek"]),
             "crossover": (out / "crossover.csv", ["omega", "nu", "zeta", "delta", "M_plus_nu", "G", "two_p", "minus_sgn_z"])}
    handles = {}
    for kind in parts:
        path, header = files[kind]
        new = not path.exists() or path.stat().st_size == 0
        handles[kind] = open(path, "a", newline="")
        if new:
            csv.writer(handles[kind]).writerow(header)
    jobs = (list(jobs_crossover()) if "crossover" in parts else []) + \
           (list(jobs_global(global_N)) if "global" in parts else [])
    # resume: skip points already present in the output files
    done = set()
    for kind in parts:
        path, _ = files[kind]
        with open(path) as f:
            for r in csv.DictReader(f):
                done.add(("global", int(r["N"]), float(r["mu"]), round(float(r["eps"]), 6)) if kind == "global"
                         else ("crossover", float(r["omega"]), float(r["nu"]), round(float(r["zeta"]), 6)))
    def key(job):
        kind, N, eps, mu, tol, meta = job
        return (kind, N, meta[1], round(meta[2], 6)) if kind == "global" else (kind, meta[0], meta[1], round(meta[2], 6))
    jobs = [j for j in jobs if key(j) not in done]
    t0 = time.time()
    with Pool(workers) as pool:
        for i, (kind, meta, val, p) in enumerate(pool.imap_unordered(_point, jobs), 1):
            w = csv.writer(handles[kind])
            if kind == "global":
                w.writerow([*meta, f"{val:.10e}"])
            else:
                omega, nu, zeta, delta = meta
                w.writerow([omega, nu, zeta, f"{delta:.8e}", f"{val+nu:.10e}", f"{G(delta, omega):.10e}",
                            f"{2*p:.8e}", f"{-np.sign(nu)*z_root(delta, omega):.8e}"])
            handles[kind].flush()
            print(f"{i}/{len(jobs)} {kind} {meta} {time.time()-t0:.0f}s", flush=True)
    for h in handles.values():
        h.close()


if __name__ == "__main__":
    args = sys.argv[1:]
    parts = tuple(a for a in args if a in ("global", "crossover")) or ("global", "crossover")
    Ns = tuple(int(a[2:]) for a in args if a.startswith("N=")) or (16, 24)
    dirs = [a for a in args if a not in ("global", "crossover") and not a.startswith("N=")]
    workers = next((int(a[8:]) for a in args if a.startswith("workers=")), 6)
    dirs = [a for a in dirs if not a.startswith("workers=")]
    out = Path(dirs[0]) if dirs else Path(__file__).resolve().parent / "data"
    main(out, parts, workers=workers, global_N=Ns)
