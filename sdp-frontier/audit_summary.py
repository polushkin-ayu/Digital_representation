#!/usr/bin/env python3
"""Summarise data/audit.csv: accuracy indicators relative to the quantities quoted in the paper."""
import csv, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
import sdp_frontier as S
from scipy.optimize import brentq

here = Path(__file__).resolve().parent
data = Path(sys.argv[1]) if len(sys.argv) > 1 else here / "data"
rows = list(csv.DictReader(open(data / "audit.csv")))
prod_g = {(int(r["N"]), float(r["mu"]), round(float(r["eps"]), 4)): float(r["dE_over_Ek"]) for r in csv.DictReader(open(data / "frontier_global.csv"))}
prod_c = {(float(r["omega"]), float(r["nu"]), float(r["zeta"])): r for r in csv.DictReader(open(data / "crossover.csv"))}

def Phi(z):
    s = brentq(lambda s: s**3 + S.PI * z * s / 2 - S.B * S.PI**2 / 8, 0, 10)
    return (z + 2 * s * s / S.PI) ** 2 / S.B - 2 * s, s

print("| kind | N | mu | eps / (omega, nu, zeta) | SCS status | gap | res_pri | res_dual | min eig | norm viol. | err viol. | value | scale | gap/scale | rerun vs production |")
print("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
for r in sorted(rows, key=lambda r: (r["kind"], int(r["N"]), -float(r["mu"]) if r["kind"] == "global" else float(r["omega"]), float(r["eps"]))):
    g = float(r["gap"]); val = float(r["energy"])
    if r["kind"] == "global":
        scale = abs(val); key = (int(r["N"]), float(r["mu"]), round(float(r["eps"]), 4))
        prod = prod_g.get(key); label = f"{float(r['eps']):.2f}"
        cmp = f"{abs(val - prod):.1e}" if prod is not None else "n/a"
    else:
        om, nu, z = float(r["omega"]), float(r["nu"]), float(r["zeta"])
        centred = val + nu; scale = abs(centred); label = f"({om:g}, {nu:+g}, {z:g})"
        pr = prod_c.get((om, nu, z))
        cmp = f"{abs(centred - float(pr['M_plus_nu'])):.1e}" if pr else "n/a"
    mineig = min(float(r["min_eig_T0"]), float(r["min_eig_T1"]))
    print(f"| {r['kind']} | {r['N']} | {float(r['mu']):g} | {label} | {r['scs_status']} | {g:.1e} | {float(r['res_pri']):.1e} | {float(r['res_dual']):.1e} | {mineig:.1e} | {float(r['norm_violation']):.1e} | {float(r['err_violation']):.1e} | {val:.6e} | {scale:.2e} | {g/scale if scale > 0 else float('nan'):.1e} | {cmp} |")
