"""Check analytic derivatives, repeatability and saved convergence evidence."""
import json
from pathlib import Path
import numpy as np
from check import mesh, minimize, quadrant
p=Path(__file__).resolve().parent
phi,w=mesh(128);n=np.arange(1,9);S=np.sin(2*phi[:,None]*n)
s=np.sign(np.cos(phi));t=.07
a=np.random.default_rng(123).normal(size=8)*.01/n**2
v=np.random.default_rng(456).normal(size=8);v/=np.linalg.norm(v)
def objective(a):
    f=S@a
    return np.dot(2*n*n,a*a)+t*np.dot(w*s,np.sin(phi+f)*np.sin(f))
def gradient(a):return 4*n*n*a+t*S.T@(w*s*np.sin(phi+2*S@a))
hess=np.diag(4.*n*n)+2*t*S.T@((w*s*np.cos(phi+2*S@a))[:,None]*S)
h=1e-6
ge=abs((objective(a+h*v)-objective(a-h*v))/(2*h)-gradient(a)@v)
he=np.max(np.abs((gradient(a+h*v)-gradient(a-h*v))/(2*h)-hess@v))
assert ge<1e-9 and he<1e-7,(ge,he)
r=json.loads((p/'results.json').read_text())
assert max(x['gradient_inf'] for x in r['fourier'])<1e-11
q=quadrant(.02,256)
errors=[abs(x['ratio']-q['ratio']) for x in r['fourier'] if x['t']==.02 and x['seed']==0 and x['order']==256]
assert all(a>b for a,b in zip(errors,errors[1:]))
repeat,_=minimize(.02,64,256)
saved=next(x for x in r['fourier'] if x['t']==.02 and x['modes']==64 and x['order']==256)
assert abs(repeat['ratio']-saved['ratio'])<1e-10
report={'directional_gradient_error':float(ge),'hessian_vector_error':float(he),'maximum_gradient_residual':max(x['gradient_inf'] for x in r['fourier']),'repeat_ratio_difference':abs(repeat['ratio']-saved['ratio']),'fourier_error_decreases':True}
(p/'validation.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
