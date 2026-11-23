"""One-digit SDP. Exact finite-cutoff symmetry reduction; not a phase restriction.
Real-part averaging of Fourier Gram matrices preserves PSD, objective and even
coordinate test. Averaging outcome exchange + phi translation by pi gives
T0=A/2,T1=D A D/2, D_ll=(-1)^l. A is real PSD, trace A=1;
even off-diagonal sums vanish. Energy=Tr(diag(l²+2mu*l) A),
error=1/2-Tr(S A)/2. Any PSD A decomposes into hidden Kraus channels.
"""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('OMP_NUM_THREADS','1')
import argparse,csv,json,time,warnings
from pathlib import Path
import numpy as np
import cvxpy as cp
from concurrent.futures import ProcessPoolExecutor,as_completed
warnings.filterwarnings('ignore',message='Solution may be inaccurate')
ROOT=Path(os.environ.get('RING_DATA_DIR', Path(__file__).resolve().parent/'data'))
ROOT.mkdir(parents=True, exist_ok=True)
EPS=0.5-1/np.pi
class Model:
 def __init__(self,N):
  self.N=N; self.l=np.arange(-N,N+1);n=len(self.l);d=self.l[:,None]-self.l[None,:]
  self.S=np.zeros((n,n));mask=d%2!=0
  self.S[mask]=2/(np.pi*np.abs(d[mask]))*(-1.)**((np.abs(d[mask])-1)//2)
  self.A=cp.Variable((n,n),symmetric=True);self.mu=cp.Parameter();self.eps=cp.Parameter()
  cons=[self.A>>0,cp.trace(self.A)==1]
  cons += [cp.sum(cp.diag(self.A,k))==0 for k in range(2,2*N+1,2)]
  self.error=.5-.5*cp.sum(cp.multiply(self.S,self.A))
  cons += [self.error<=self.eps]
  self.scale=100.0
  self.prob=cp.Problem(cp.Minimize(cp.sum(cp.multiply(self.l**2+2*self.mu*self.l,cp.diag(self.A)))/self.scale),cons)
 def solve(self,eps,mu,tol=1e-8,solver='SCS'):
  self.eps.value=eps; self.mu.value=mu;t=time.time()
  if solver=='SCS':self.prob.solve(solver='SCS',eps_abs=tol,eps_rel=tol,max_iters=200000,warm_start=True)
  else:self.prob.solve(solver='CLARABEL',tol_gap_abs=tol,tol_gap_rel=tol,tol_feas=tol,max_iter=200,direct_solve_method='qdldl')
  if self.A.value is None: return dict(N=self.N,mu=mu,eps=eps,status=self.prob.status,seconds=time.time()-t)
  a=np.asarray(self.A.value);q=np.diag(a);info=(self.prob.solver_stats.extra_stats or {}).get('info',{})
  error=float(.5-.5*np.sum(self.S*a));energy=float(np.dot(self.l**2+2*mu*self.l,q))
  lam=self.prob.constraints[1].dual_value;tau=self.prob.constraints[-1].dual_value
  z=np.diag((self.l**2+2*mu*self.l)/self.scale)+lam*np.eye(len(self.l))-.5*tau*self.S
  for k,c in zip(range(2,2*self.N+1,2),self.prob.constraints[2:-1]):
   z+=float(c.dual_value)*.5*(np.eye(len(self.l),k=k)+np.eye(len(self.l),k=-k))
  dual=float((-lam+tau*(.5-eps))*self.scale)
  gap=abs(energy-dual)
  row=dict(N=self.N,mu=mu,eps=eps,solver=solver,tol=tol,status=self.prob.status,energy=energy,error=error,
   transfer=float(self.l@q),variance=float(self.l**2@q-(self.l@q)**2),
   min_eig=float(np.linalg.eigvalsh(a).min()),norm_res=max([abs(np.trace(a)-1)]+[abs(np.diag(a,k).sum()) for k in range(2,2*self.N+1,2)]),
   error_violation=max(0.,error-eps),gap=gap,dual_energy=dual,dual_min_eig=float(np.linalg.eigvalsh(z).min()),dual_error_multiplier=float(tau*self.scale),res_pri=info.get('res_pri'),res_dual=info.get('res_dual'),
   iterations=self.prob.solver_stats.num_iters,seconds=round(time.time()-t,3))
  return row
CACHE={}
def task(job):
 N,mu,eps,solver,tol=job
 if N not in CACHE:CACHE[N]=Model(N)
 try:return CACHE[N].solve(eps,mu,tol,solver)
 except Exception as e:return dict(N=N,mu=mu,eps=eps,status='failed',message=str(e))
def main():
 p=argparse.ArgumentParser();p.add_argument('--mode',default='pilot');p.add_argument('--workers',type=int,default=4);a=p.parse_args()
 if a.mode=='pilot':jobs=[(N,mu,eps,'SCS',1e-8) for N in (12,20) for mu,eps in [(0,.04),(.25,.12),(.49,.18),(.5,EPS),(.25,.4),(.5,.08)]]
 else:jobs=json.loads((ROOT/(a.mode+'-jobs.json')).read_text())
 out=ROOT/(a.mode+'.jsonl');done=set()
 if out.exists():
  for line in out.read_text().splitlines():
   r=json.loads(line);done.add((r['N'],r['mu'],r['eps']))
 jobs=[j for j in jobs if tuple(j[:3]) not in done]
 t=time.time()
 with ProcessPoolExecutor(max_workers=a.workers) as pool,open(out,'a',buffering=1) as f:
  fs=[pool.submit(task,j) for j in jobs]
  for i,fu in enumerate(as_completed(fs),1):
   r=fu.result();f.write(json.dumps(r)+'\n');print(i,len(jobs),r,'elapsed',round(time.time()-t),flush=True)
if __name__=='__main__':main()
