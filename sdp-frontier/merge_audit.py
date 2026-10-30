#!/usr/bin/env python3
"""Assemble data/audit_final.csv: the first audit with re-solved rows substituted.

Usage: python3 merge_audit.py [first.csv] [rerun.csv] [out.csv]
Defaults: data/audit.csv, data/audit_rerun_2M.csv, data/audit_final.csv next to this script.
Rows are matched by (kind, N, mu, eps); a column "run" records which audit each row comes from.
"""
import csv, sys
from pathlib import Path

d = Path(__file__).resolve().parent / "data"
first, rerun, out = (Path(a) for a in (sys.argv[1:4] + [d / "audit.csv", d / "audit_rerun_2M.csv", d / "audit_final.csv"][len(sys.argv) - 1:]))
key = lambda r: (r["kind"], r["N"], r["mu"], r["eps"])
new = {key(r): r for r in csv.DictReader(open(rerun))}
fields = list(csv.DictReader(open(rerun)).fieldnames) + ["run"]
rows = []
for r in csv.DictReader(open(first)):
    k = key(r)
    rows.append(dict(new[k], run="rerun_2M") if k in new else dict(r, max_iters=400000, run="first"))
with open(out, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)
print(f"{out}: {len(rows)} rows, {len(new)} re-solved")
