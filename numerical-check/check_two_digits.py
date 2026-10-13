"""Sanity check of the explicit two-digit asymptotic construction (not a proof)."""
from pathlib import Path
import numpy as np
import json
from numpy.polynomial.legendre import leggauss
OUT=Path(__file__).resolve().parent

def response(x):
    turns=np.floor((x+np.pi)/(2*np.pi)).astype(int)
    y=x-2*np.pi*turns
    z=np.abs(y); C=np.cos(z/2); S=np.sin(z/2)
    b=-1-np.pi/4
    left=z<=np.pi/2
    v=np.where(left,(z+b)*S, -z*S-2*C+(np.pi+2+b)*S)
    dv=np.where(left,S+(z+b)*C/2,(-z+np.pi+2+b)*C/2)
    factor=np.where(turns%2==0,1.,-1.)
    return factor*np.sign(y)*v, factor*dv

def run(order):
    x,w=leggauss(order)
    edges=[-np.pi,-np.pi/2,0,np.pi/2,np.pi]
    phi=np.concatenate([(a+b)/2+(b-a)*x/2 for a,b in zip(edges[:-1],edges[1:])])
    w=np.tile(w/8,4)
    s1=np.sign(np.cos(phi));s2=-np.sign(np.sin(2*phi))
    C=np.cos(phi/2);S=np.sin(phi/2)
    vc,dvc=response(phi);tmp,dtmp=response(phi+np.pi);vs,dvs=-tmp,-dtmp
    hphi=(phi+np.pi/2)%np.pi-np.pi/2
    h=np.sin(hphi)-2*hphi/np.pi
    dh=np.abs(np.cos(phi))-2/np.pi
    V=vc**2+vs**2
    chi=float(np.dot(w,s2*(C*vc+S*vs)))
    rows=[]
    for lam in [.1,.05,.02,.01,.005,.002]:
        den=1+lam**2*V;Z=np.sqrt(den)
        def d1(eta):
            dcos=2*np.sin(phi-eta*h)*np.sin(eta*h)
            return .5*np.dot(w*s1,(dcos+lam**2*(vc**2-vs**2-V*np.cos(phi)))/den)
        lo,hi=-5*lam**2,5*lam**2
        assert d1(lo)*d1(hi)<0
        for _ in range(70):
            mid=(lo+hi)/2
            if d1(mid)>0:hi=mid
            else:lo=mid
        eta=(lo+hi)/2
        alpha=phi/2-eta*h;da=.5-eta*dh
        ca,sa=np.cos(alpha),np.sin(alpha)
        delta=lam*np.dot(w*s2,(ca*vc+sa*vs)/den)
        dz=lam**2*(vc*dvc+vs*dvs)/Z
        us=[];dus=[];base=[];dbase=[]
        for a,ad,v,vd,z0,dz0 in [(ca,-sa*da,vc,dvc,C,-S/2),(sa,ca*da,vs,dvs,S,C/2)]:
            for sign in [1,-1]:
                numer=(a+sign*lam*v)/np.sqrt(2)
                u=numer/Z
                du=(ad+sign*lam*vd)/np.sqrt(2)/Z-numer*dz/den
                us.append(u);dus.append(du);base.append(z0/np.sqrt(2));dbase.append(dz0/np.sqrt(2))
        us=np.array(us);dus=np.array(dus)
        diff=us-np.array(base);ddiff=dus-np.array(dbase)
        F=float(np.dot(w,np.sum(ddiff**2-.25*diff**2,axis=0)))
        directd2=.5*np.dot(w*s2,np.sum(us[[0,2]]**2-us[[1,3]]**2,axis=0))
        assert abs(delta-directd2)<1e-14
        normerr=float(np.max(np.abs(np.sum(us**2,axis=0)-1)))
        assert normerr<1e-13 and abs(d1(eta))<1e-13
        rows.append(dict(parameter=lam,eta=eta,delta2=float(delta),delta1=float(d1(eta)),F=F,ratio=F/delta**2,norm_error=normerr))
    return dict(order=order,chi_quadrature=chi,orthogonality=float(np.dot(w,vc*S)),rows=rows)
def main():
    results=dict(target=np.pi/(4-np.pi),chi=4/np.pi-1,checks=[run(64),run(128),run(256)])
    (OUT/'two-digits-results.json').write_text(json.dumps(results,indent=2)+'\n')
    print('Target:',results['target'])
    for a in results['checks']:
     print('order',a['order'],'chi',a['chi_quadrature'],'orthogonality',a['orthogonality'])
     for x in a['rows']:print(x['parameter'],x['eta'],x['delta2'],x['ratio'])

if __name__ == "__main__":
    main()
