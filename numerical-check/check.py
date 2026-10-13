"""Independent numerical checks, using NumPy only. Run from any directory."""
from pathlib import Path
import json
import numpy as np
from numpy.polynomial.legendre import leggauss

OUT = Path(__file__).resolve().parent
B = .5 - 4 / np.pi**2
TARGET = 1 / B

def quadrant(t, order):
    x, w = leggauss(order)
    z, w = (x + 1) * np.pi / 4, w / 2  # normalized quarter average
    cz = np.cos(z)
    lo, hi = max(2*t, 0.) + 1e-14, 1 + 2*t
    for _ in range(80):
        c = (lo + hi) / 2
        if np.dot(w, 1 / np.sqrt(c - 2*t*cz)) > 1:
            lo = c
        else:
            hi = c
    c = (lo + hi) / 2
    d = c - 2*t*cz
    speed = np.sqrt(d)
    # Stable formulas avoid subtracting two order-one kinetic energies.
    change = (d - 1) / (speed + 1)
    F = np.dot(w, change**2 / speed) / 4
    delta = -.5 * np.dot(w, cz * change / speed)
    residual = float(np.dot(w, 1/speed)-1)
    if abs(residual) > 1e-12 or not (F > 0 and delta > 0):
        raise RuntimeError("Quadrature normalization or positivity failed")
    return dict(t=t, order=order, c=c, delta=delta, F=F,
                ratio=F/delta**2,
                normalization_residual=float(np.dot(w, 1/speed)-1))

def mesh(order):
    x, w = leggauss(order)
    edges = [-np.pi, -np.pi/2, 0, np.pi/2, np.pi]
    phi = np.concatenate([(a+b)/2+(b-a)*x/2 for a,b in zip(edges[:-1],edges[1:])])
    weights = np.tile(w/8, 4)  # normalized full-circle integral
    return phi, weights

def minimize(t, modes, order, seed=0):
    phi,w = mesh(order)
    n = np.arange(1,modes+1)
    basis = np.sin(2*phi[:,None]*n)
    s = np.sign(np.cos(phi))
    a = np.zeros(modes) if seed == 0 else np.random.default_rng(seed).normal(size=modes)*.03/n**2
    def evaluate(a):
        f = basis@a
        F = float(np.dot(2*n*n, a*a))
        # Exact known baseline integrated analytically, small difference numerically.
        dc = -2*np.sin(phi+f)*np.sin(f)
        delta = float(.5*np.dot(w*s, dc))
        return F-t*delta, F, delta, f
    for iteration in range(60):
        value,F,delta,f = evaluate(a)
        grad = 4*n*n*a + t*(basis.T@(w*s*np.sin(phi+2*f)))
        if np.max(np.abs(grad)) < 1e-13:
            break
        hessian = np.diag(4.*n*n) + 2*t*(basis.T@((w*s*np.cos(phi+2*f))[:,None]*basis))
        step = np.linalg.solve(hessian, grad)
        if np.max(np.abs(step)) < 3e-16:
            break
        rate = 1.
        while evaluate(a-rate*step)[0] > value - 1e-4*rate*np.dot(grad,step) + 32*np.finfo(float).eps*max(abs(value), 1e-30):
            rate *= .5
            if rate < 2**-25:
                raise RuntimeError(f"Newton line search failed: t={t}, N={modes}, grad={np.max(np.abs(grad))}, step={np.max(np.abs(step))}, value={value}")
        a -= rate*step
    value,F,delta,f = evaluate(a)
    grad = 4*n*n*a + t*(basis.T@(w*s*np.sin(phi+2*f)))
    if not np.all(np.isfinite(a)) or np.max(np.abs(grad)) > 1e-11:
        raise RuntimeError("Newton stationarity tolerance not met")
    if not (F > 0 and delta > 0):
        raise RuntimeError("Nonpositive branch energy or improvement")
    return dict(t=t, modes=modes, order=order, seed=seed, iterations=iteration+1,
                F=F, delta=delta, ratio=F/delta**2, dual=value,
                gradient_inf=float(np.max(np.abs(grad)))), a

def spectrum(t, a, cutoff, order):
    phi,w = mesh(order)
    n=np.arange(1,len(a)+1)
    f=np.sin(2*phi[:,None]*n)@a
    fp=(2*n*np.cos(2*phi[:,None]*n))@a
    s=np.sign(np.cos(phi))
    q=(.5+fp)**2-t*s*np.cos(phi+2*f)/2
    # Complex antiperiodic basis exp(i*(l+1/2)*phi), both signs symmetric.
    freq=np.arange(-cutoff,cutoff)+.5
    basis=np.exp(1j*phi[:,None]*freq)
    ans=[]
    for sign in [-1,1]:
        potential=-q+sign*t*s/2
        matrix=np.diag(freq**2)+basis.conj().T@((w*potential)[:,None]*basis)
        matrix=(matrix+matrix.conj().T)/2
        vals=np.linalg.eigvalsh(matrix)
        ans.append([float(v) for v in vals[:4]])
    return dict(t=t, cutoff=cutoff, profile_modes=len(a), order=order, L0=ans[0], L1=ans[1])

def main():
    quadratures=[quadrant(t,order) for order in [64,128,256]
                 for t in [.2,.1,.05,.02,.01,.005,.002,.001]]
    fourier=[]; spectra=[]
    for modes in [8,16,32,64]:
        for t in [.1,.02,.005,.001]:
            result,a=minimize(t,modes,256)
            fourier.append(result)
        result,a=minimize(.02,modes,256)
        spectra.append(spectrum(.02,a,modes,256))
    for seed in [11,29,47]:
        result,_=minimize(.02,32,256,seed)
        fourier.append(result)
    # Independent quadrature-resolution check of Fourier gradients and spectra.
    result,a=minimize(.02,64,384)
    fourier.append(result)
    spectra.append(spectrum(.02,a,64,384))
    data=dict(numpy_version=np.__version__, B=B, target=TARGET,
              quadrature=quadratures, fourier=fourier, spectra=spectra)
    (OUT/'results.json').write_text(json.dumps(data,indent=2)+'\n')
    print('Target',TARGET)
    print('Quadrature (order 256): t, delta, F/delta^2')
    for r in quadratures:
        if r['order']==256: print(r['t'],r['delta'],r['ratio'])
    print('Fourier: modes, t, ratio, gradient infinity norm')
    for r in fourier: print(r['modes'],r['t'],r['ratio'],r['gradient_inf'])
    print('Spectra at t=.02: cutoff, smallest four L0 eigenvalues')
    for r in spectra:print(r['cutoff'],r['L0'])

if __name__=='__main__': main()
