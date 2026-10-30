#!/usr/bin/env python3
"""Audit of the reference points quoted in the numerical section.

Each point is re-solved with the production settings of sdp_frontier.py and the
following are stored: SCS status, iterations, primal and dual residuals, duality
gap and objectives; and, recomputed independently from the returned operators
T_0, T_1: the smallest eigenvalues (PSD violation), the largest violation of the
Kraus-normalisation equalities, the violation of the error constraint, the energy
Tr(D T) and the phase current.  Requested tolerances are not treated as achieved
accuracy; the gap and the constraint violations are reported instead.

Usage: python3 audit_points.py [outfile.csv] [workers=6]   (default: data/audit.csv next to this script)
"""
import sys, csv, time
from pathlib import Path
from multiprocessing import Pool
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))  # python -I omits the script directory
import sdp_frontier as S

PI, B, EPS = S.PI, S.B, S.EPS

def points():
    P = []
    for N in (16, 24):
        for eps in (0.04, 0.12, 0.16, 0.18, 0.20, 0.30):
            P.append(("global", N, 0.5, eps, 1e-8, None))
        for eps in (0.04, 0.18, 0.44, 0.48):
            P.append(("global", N, 0.0, eps, 1e-8, None))
    for omega, sign, zetas in ((1e-3, 1, (-1.5, 0.0, 1.0, 2.0)), (1e-3, -1, (-1.5, 0.0, 1.0, 2.0)),
                               (4e-3, 1, (0.0,)), (1.6e-2, 1, (-1.5, 0.0, 0.5, 1.0))):
        for z in zetas:
            P.append(("crossover", 16, 0.5 + sign * omega, EPS - z * omega ** (2 / 3), 1e-10, (omega, sign * omega, z)))
    return P

_C = {}
def run(pt):
    kind, N, mu, eps, tol, meta = pt
    if N not in _C:
        _C[N] = S.Frontier(N)
    fr = _C[N]
    t0 = time.time()
    val, p = fr.solve(eps, mu, tol)
    info = fr.prob.solver_stats.extra_stats["info"]
    T0, T1 = (np.asarray(t.value) for t in fr.T)
    T0, T1 = (T0 + T0.conj().T) / 2, (T1 + T1.conj().T) / 2
    l = fr.l.astype(float); n = len(l)
    diff = fr.l[:, None] - fr.l[None, :]
    Ts = T0 + T1
    norm_res = max(abs(Ts[diff == -m].sum() - (1.0 if m == 0 else 0.0)) for m in range(-2 * N, 2 * N + 1))
    err = 0.5 + float(np.real(np.trace(fr.S.T @ T1)))
    energy = float(np.real(np.trace(np.diag(l * l + 2 * mu * l) @ Ts)))
    cur = float(np.dot(l + 0.5, np.real(np.diag(Ts))))
    row = dict(kind=kind, N=N, mu=mu, eps=eps, tol=tol,
               omega=meta[0] if meta else "", nu=meta[1] if meta else "", zeta=meta[2] if meta else "",
               status=fr.prob.status, scs_status=info["status"], iters=info["iter"],
               res_pri=info["res_pri"], res_dual=info["res_dual"], gap=info["gap"],
               pobj=info["pobj"], dobj=info["dobj"],
               min_eig_T0=float(np.linalg.eigvalsh(T0).min()), min_eig_T1=float(np.linalg.eigvalsh(T1).min()),
               norm_violation=float(norm_res), err=err, err_violation=max(0.0, err - eps),
               energy=val, energy_recomputed=energy, two_p=2 * cur, seconds=round(time.time() - t0, 1))
    return row

if __name__ == "__main__":
    out = Path(sys.argv[1]) if len(sys.argv) > 1 and not sys.argv[1].startswith("workers=") else Path(__file__).resolve().parent / "data" / "audit.csv"
    workers = next((int(a[8:]) for a in sys.argv[1:] if a.startswith("workers=")), 6)
    pts = points()
    with Pool(workers) as pool, open(out, "w", newline="") as f:
        w = None
        for i, row in enumerate(pool.imap_unordered(run, pts), 1):
            if w is None:
                w = csv.DictWriter(f, fieldnames=list(row)); w.writeheader()
            w.writerow(row); f.flush()
            print(f"{i}/{len(pts)} {row['kind']} N={row['N']} mu={row['mu']} eps={row['eps']:.5f} {row['scs_status']} gap={row['gap']:.1e} {row['seconds']}s", flush=True)
