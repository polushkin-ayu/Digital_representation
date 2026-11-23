"""Direct root solves, not roots of the interpolated heatmap."""
import json,time,os
from pathlib import Path
from simulate import Model,EPS
import numpy as np
P=Path(os.environ.get('RING_DATA_DIR',Path(__file__).resolve().parent/'data'));P.mkdir(parents=True,exist_ok=True);B=.5-4/np.pi**2
out=P/'roots.jsonl'
done={(r['N'],r['nu']) for r in map(json.loads,out.read_text().splitlines())} if out.exists() else set()
for N in (16,24):
 m=Model(N)
 for nu in [.0001,.0003,.001,.003,.01,.03,.06]:
  if (N,nu) in done:continue
  eps=EPS-np.sqrt(B*nu);hist=[]
  lo=.04;hi=EPS
  for it in range(8):
   r=m.solve(eps,.5-nu,1e-10,'CLARABEL');f=r['energy']-2*nu
   hist.append(r)
   if abs(f)<2e-8:break
   if f>0:lo=eps
   else:hi=eps
   candidate=eps+f/r['dual_error_multiplier']
   eps=candidate if lo<candidate<hi else (lo+hi)/2
  eps=hist[-1]['eps']
  row=dict(N=N,nu=nu,eps=eps,delta=EPS-eps,asymptotic_delta=float(np.sqrt(B*nu)),residual=f,history=hist)
  with open(out,'a') as file:file.write(json.dumps(row)+'\n')
  print(N,nu,eps,f,len(hist),flush=True)
