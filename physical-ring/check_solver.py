"""Recompute one published grid point independently of the stored batch logs."""
from pathlib import Path
import importlib.metadata as md
import json,sys
from simulate import Model
from reproduce import acceptable, records
P=Path(__file__).resolve().parent
r=Model(24).solve(.17,.5,tol=1e-10,solver='CLARABEL')
old=[x for x in records(P/'data',['map','repair']) if x.get('N')==24 and x.get('mu')==.5 and x.get('eps')==.17 and acceptable(x)][-1]
r['stored_energy']=old['energy'];r['absolute_difference']=abs(r['energy']-old['energy'])
r['python']=sys.version;r['versions']={x:md.version(x) for x in ['numpy','scipy','matplotlib','cvxpy','clarabel','scs']}
(P/'build').mkdir(exist_ok=True)
(P/'build/solver-check.json').write_text(json.dumps(r,indent=2)+'\n')
assert acceptable(r),r
assert r['absolute_difference']<2e-7,r
print(json.dumps(r,indent=2))
