"""Separate Fourier/Parseval checks of the two-digit proof; NumPy only.

Finite tests support, but do not prove, the operator inequalities.
"""
from pathlib import Path
import json
import numpy as np
from numpy.polynomial.legendre import leggauss
from check_two_digits import response
OUT=Path(__file__).resolve().parent
CHI=4/np.pi-1
B=.5-4/np.pi**2

def sigma(n):
    n=np.asarray(n,dtype=int)
    valid=(n%2==0)&((n//2)%2!=0)
    out=np.zeros(n.shape,dtype=complex)
    out[valid]=2j/(np.pi*(n[valid]//2))
    return out

def spectral_check(M,phi,w):
    m=np.arange(-M,M);e=m*(m+1)
    keep=e>0;m=m[keep];e=e[keep]
    # Columns: S exp(+i phi/2), S exp(-i phi/2).
    A=np.column_stack([sigma(m),sigma(m+1)])
    matrix=A.conj().T@(A/e[:,None])
    vc_coeff=(A[:,0]+A[:,1])/(2*e)
    vd_coeff=(A[:,0]-A[:,1])/(2j*e)
    basis=np.exp(1j*phi[:,None]*(m+.5))
    vc=basis@vc_coeff;vd=basis@vd_coeff
    directc,_=response(phi);directd,_=response(phi+np.pi);directd=-directd
    errors=[np.sqrt(np.dot(w,np.abs(vc-directc)**2)),np.sqrt(np.dot(w,np.abs(vd-directd)**2))]
    assert np.max(np.abs(matrix-np.diag(np.diag(matrix))))<1e-14
    return {'cutoff':M,'matrix_diagonal':np.diag(matrix).real.tolist(),
            'matrix_error':float(np.max(np.abs(matrix-CHI*np.eye(2)))),
            'response_l2_errors':[float(x) for x in errors]}

def fields(phi,lam,eta):
    vc,dvc=response(phi);vs,dvs=response(phi+np.pi);vs=-vs;dvs=-dvs
    y=(phi+np.pi/2)%np.pi-np.pi/2
    h=np.sin(y)-2*y/np.pi;hp=np.abs(np.cos(phi))-2/np.pi
    alpha=phi/2-eta*h;ap=.5-eta*hp
    c,s=np.cos(alpha),np.sin(alpha)
    V=vc**2+vs**2;z=np.sqrt(1+lam**2*V)
    zp=lam**2*(vc*dvc+vs*dvs)/z
    u=[];up=[]
    for a,ad,v,vd in [(c,-s*ap,vc,dvc),(s,c*ap,vs,dvs)]:
        for sign in [1,-1]:
            raw=(a+sign*lam*v)/np.sqrt(2)
            u.append(raw/z)
            up.append((ad+sign*lam*vd)/np.sqrt(2)/z-raw*zp/z**2)
    return np.array(u),np.array(up)

def errors(phi,w,u):
    # Direct error probabilities, without the d1/d2 formulas of the builder.
    b1=(np.cos(phi)<0);b2=(np.sin(2*phi)>0)
    W1=np.sum(u[2:]**2,axis=0);W2=np.sum(u[[1,3]]**2,axis=0)
    return float(np.dot(w,np.where(b1,1-W1,W1))),float(np.dot(w,np.where(b2,1-W2,W2)))

def main():
    x,w0=leggauss(384);edges=np.linspace(-np.pi,np.pi,5)
    phi=np.concatenate([(a+b)/2+(b-a)*x/2 for a,b in zip(edges[:-1],edges[1:])]);w=np.tile(w0/8,4)
    spec=[spectral_check(M,phi,w) for M in [16,32,64,128,256]]
    assert all(a['matrix_error']>b['matrix_error'] for a,b in zip(spec,spec[1:]))
    epsstar=.5-1/np.pi
    # IFT derivative evaluated from the direct error, not from its rearrangement.
    hstep=1e-5
    ep=errors(phi,w,fields(phi,0,hstep)[0])[0]
    em=errors(phi,w,fields(phi,0,-hstep)[0])[0]
    deriv=-(ep-em)/(2*hstep)
    assert abs(deriv-B)<1e-10
    checks=[]
    for lam in [.1,.02,.005]:
        lo,hi=-5*lam**2,5*lam**2
        for _ in range(60):
            eta=(lo+hi)/2
            e1,_=errors(phi,w,fields(phi,lam,eta)[0])
            if e1<epsstar:hi=eta
            else:lo=eta
        eta=(lo+hi)/2;u,up=fields(phi,lam,eta)
        e1,e2=errors(phi,w,u);delta=.5-e2
        # Direct kinetic energy (no subtraction of the kernel vector).
        F=float(np.dot(w,np.sum(up**2,axis=0))-.25)
        assert abs(e1-epsstar)<1e-13
        assert F>=4*(np.sqrt(CHI+delta)-np.sqrt(CHI))**2
        coeff_energy=[]
        for N in [2048,4096,8192]:
            grid=2*np.pi*np.arange(N)/N
            ug,_=fields(grid,lam,eta)
            k=ug*np.exp(-.5j*grid)
            coeff=np.fft.fft(k,axis=1)/N
            freq=np.fft.fftfreq(N,d=1/N)
            Ffft=float(np.sum(np.abs(coeff)**2*freq*(freq+1)))
            # Also use a different input with mean momentum 1/2:
            # r_{-1}=1/4, r_1=3/4, E_in/E_kappa=1.
            Eout=float(np.sum(np.abs(coeff)**2*(.25*(freq-1)**2+.75*(freq+1)**2)))
            assert abs(Eout-1-Ffft)<1e-12
            coeff_energy.append({'grid':N,'F':Ffft,'difference_from_direct':Ffft-F})
        assert abs(coeff_energy[-1]['difference_from_direct'])<1e-10
        ua,upa=fields(np.array([-np.pi,np.pi]),lam,eta)
        trace_err=max(np.max(np.abs(ua[:,0]+ua[:,1])),np.max(np.abs(upa[:,0]+upa[:,1])))
        assert trace_err<1e-13
        checks.append({'lambda':lam,'eta':eta,'e1_minus_threshold':e1-epsstar,'delta2':delta,'F_direct':F,'ratio':F/delta**2,'boundary_trace_error':float(trace_err),'fourier_energy':coeff_energy})
    # Complex normalized instruments unrelated to the attaining branch.
    rng=np.random.default_rng(812)
    N=4096;grid=2*np.pi*np.arange(N)/N;s2=-np.sign(np.sin(2*grid));s2[np.abs(np.sin(2*grid))<1e-12]=0
    freq=np.fft.fftfreq(N,d=1/N)
    margins=[]
    for trial in range(24):
        noise=np.zeros((6,N),dtype=complex)
        for j in range(6):
            for ell in [-4,-3,-2,1,2,3,4]:
                noise[j]+=(rng.normal()+1j*rng.normal())*np.exp(1j*ell*grid)/np.sqrt(42)
        base=np.ones((6,N),dtype=complex)/np.sqrt(6)
        k=base+(10**rng.uniform(-3,-.5))*noise
        k/=np.sqrt(np.sum(np.abs(k)**2,axis=0))
        coeff=np.fft.fft(k,axis=1)/N
        F=float(np.sum(np.abs(coeff)**2*freq*(freq+1)))
        W=np.sum(np.abs(k[3:])**2,axis=0)
        delta=abs(float(np.mean(s2*(.5-W))))
        lower=4*(np.sqrt(CHI+delta)-np.sqrt(CHI))**2
        assert F+1e-12>=lower
        margins.append(F-lower)
    report={'chi':CHI,'target':1/CHI,'spectral_checks':spec,'ift_derivative':deriv,'ift_derivative_exact':B,'construction_checks':checks,'complex_instruments_tested':len(margins),'minimum_complex_lower_bound_margin':min(margins)}
    (OUT/'two-digits-verification.json').write_text(json.dumps(report,indent=2)+'\n')
    print('Resolvent matrix max error (cutoff 256):',spec[-1]['matrix_error'])
    print('Response L2 errors:',spec[-1]['response_l2_errors'])
    print('IFT derivative:',deriv,'exact:',B)
    for x in checks:print('lambda',x['lambda'],'error1',x['e1_minus_threshold'],'ratio',x['ratio'],'FFT error',x['fourier_energy'][-1]['difference_from_direct'])
    print('Complex tests passed:',len(margins))
if __name__=='__main__':main()
