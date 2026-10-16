"""Finite-difference gradients and independent FFT energy for the joint solver."""
from pathlib import Path
import json
import numpy as np
from check_joint_optimization import Problem
p=Problem(8,192);rng=np.random.default_rng(1729)
z=rng.normal(size=np.prod(p.shape))*.004
value,g,obs=p.evaluate(z,.03,.02)
errs=[]
for _ in range(8):
 d=rng.normal(size=z.size);d/=np.linalg.norm(d)
 for h in [1e-5,3e-6,1e-6]:
  fd=(p.evaluate(z+h*d,.03,.02)[0]-p.evaluate(z-h*d,.03,.02)[0])/(2*h)
  errs.append(abs(fd-g@d))
assert max(errs)<1e-8
# Build the normalised instrument at equidistant points; energy via FFT,
# without using the analytic derivative or the stable energy expression.
a=z.reshape(p.shape);N=8192;x=2*np.pi*np.arange(N)/N;n=np.arange(1,9)
b=np.column_stack([np.ones(N),np.cos(x[:,None]*n)/n,np.sin(x[:,None]*n)/n]);f=a@b.T
C=np.cos(x/2)/np.sqrt(2);S=np.sin(x/2)/np.sqrt(2)
u0=np.array([C,C,S,S]);E=np.stack([np.array([-S,-S,C,C]),np.array([C,-C,S,-S]),np.array([-S,S,C,-C])],axis=1)
u=(u0+np.einsum('ikx,kx->ix',E,f))/np.sqrt(1+np.sum(f*f,axis=0))
k=u*np.exp(-.5j*x);coeff=np.fft.fft(k,axis=1)/N;ell=np.fft.fftfreq(N,d=1/N)
Ffft=float(np.sum(abs(coeff)**2*ell*(ell+1)))
assert abs(Ffft-obs['F'])<1e-12
r=json.loads(Path(__file__).with_name('joint-optimization-results.json').read_text())
controls=r['controls'];ref=next(q for q in r['rows'] if q['modes']==128 and q['t']==.02 and q['v']==.02)
assert abs(controls[0]['dual']-ref['dual'])<1e-13
random_values=[q['dual'] for q in controls[1:]]
assert max(random_values)-min(random_values)<1e-12
report=dict(max_gradient_difference=max(errs),fft_energy=Ffft,analytic_energy=obs['F'],fft_difference=Ffft-obs['F'],quadrature_dual_difference=controls[0]['dual']-ref['dual'],random_start_dual_spread=max(random_values)-min(random_values))
Path(__file__).with_name('joint-optimization-validation.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
