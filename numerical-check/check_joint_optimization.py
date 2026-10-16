"""Independent real four-channel variational check; NumPy only.

Optimises F-t*d1-v*d2 with unrestricted real periodic Fourier tangent
corrections to an antiperiodic frame, followed by pointwise normalisation. No analytic attaining
profile, B, chi, or separability is used by the optimiser. Local numerical
minima in this finite parametrisation are not global certificates.
"""
from pathlib import Path
import json
import numpy as np
from numpy.polynomial.legendre import leggauss
OUT=Path(__file__).resolve().parent

class Problem:
    def __init__(self,modes,order):
        x,w=leggauss(order);edges=np.linspace(-np.pi,np.pi,5)
        self.x=np.concatenate([(a+b)/2+(b-a)*x/2 for a,b in zip(edges[:-1],edges[1:])])
        self.w=np.tile(w/8,4); x=self.x
        freq=np.arange(1,modes+1)
        self.b=np.concatenate([np.ones((len(x),1)),np.cos(x[:,None]*freq)/freq,np.sin(x[:,None]*freq)/freq],axis=1)
        self.bp=np.concatenate([np.zeros((len(x),1)),-np.sin(x[:,None]*freq),np.cos(x[:,None]*freq)],axis=1)
        self.u0=np.array([np.cos(x/2),np.cos(x/2),np.sin(x/2),np.sin(x/2)])/np.sqrt(2)
        self.p0=np.array([-np.sin(x/2),-np.sin(x/2),np.cos(x/2),np.cos(x/2)])/(2*np.sqrt(2))
        self.s1=np.sign(np.cos(x));self.s2=-np.sign(np.sin(2*x))
        self.j1=np.array([1,1,-1,-1])[:,None];self.j2=np.array([1,-1,1,-1])[:,None]
        C=np.cos(x/2)/np.sqrt(2);S=np.sin(x/2)/np.sqrt(2)
        e1=np.array([-S,-S,C,C]);e2=np.array([C,-C,S,-S]);e3=np.array([-S,S,C,-C])
        self.E=np.stack([e1,e2,e3],axis=1)
        self.Ep=np.stack([-self.u0/2,e3/2,-e2/2],axis=1)
        self.shape=(3,1+2*modes)
    def evaluate(self,z,t,v):
        a=z.reshape(self.shape);f=a@self.b.T;fp=a@self.bp.T
        q=self.u0+np.einsum('ikx,kx->ix',self.E,f)
        qp=self.p0+np.einsum('ikx,kx->ix',self.Ep,f)+np.einsum('ikx,kx->ix',self.E,fp)
        n=np.sqrt(np.sum(q*q,axis=0));u=q/n
        uq=np.sum(u*qp,axis=0);up=(qp-u*uq)/n
        d=u-self.u0;dp=up-self.p0
        F=float(np.dot(self.w,np.sum(dp*dp-.25*d*d,axis=0)))
        # Subtract the exact threshold baseline before integration.
        sq=d*(u+self.u0)
        d1=float(.5*np.dot(self.w*self.s1,np.sum(self.j1*sq,axis=0)))
        d2=float(.5*np.dot(self.w*self.s2,np.sum(self.j2*sq,axis=0)))
        au=-.5*d-t*self.s1*self.j1*u-v*self.s2*self.j2*u;ap=2*dp
        ua=np.sum(u*au,axis=0);ub=np.sum(u*ap,axis=0)
        temp=uq*ap+ub*qp
        gq=(au-u*ua)/n-(temp-u*np.sum(u*temp,axis=0))/n**2-u*np.sum(ap*up,axis=0)/n
        gqp=(ap-u*ub)/n
        gf=np.einsum('ikx,ix->kx',self.E,gq)+np.einsum('ikx,ix->kx',self.Ep,gqp)
        gfp=np.einsum('ikx,ix->kx',self.E,gqp)
        grad=(gf*self.w)@self.b+(gfp*self.w)@self.bp
        return F-t*d1-v*d2,grad.ravel(),dict(F=F,d1=d1,d2=d2,norm_error=float(np.max(abs(np.sum(u*u,axis=0)-1))),direct_energy_difference=float(np.dot(self.w,np.sum(up*up,axis=0))-.25-F))
    def solve(self,t,v,seed=0):
        z=np.zeros(np.prod(self.shape))
        if seed:z=np.random.default_rng(seed).normal(size=z.size)*.001
        value,g,obs=self.evaluate(z,t,v);history=[]
        for it in range(2000):
            if np.max(abs(g))<2e-9:break
            q=g.copy();alpha=[]
            for s,y,rho in reversed(history):
                aa=rho*np.dot(s,q);alpha.append(aa);q-=aa*y
            scale=np.dot(history[-1][0],history[-1][1])/np.dot(history[-1][1],history[-1][1]) if history else 1.
            direction=scale*q
            for (s,y,rho),aa in zip(history,reversed(alpha)):
                direction+=s*(aa-rho*np.dot(y,direction))
            direction=-direction
            if np.dot(direction,g)>=0:direction=-g;history=[]
            rate=1.
            for trial in range(50):
                nz=z+rate*direction;nv,ng,no=self.evaluate(nz,t,v)
                if nv<=value+1e-4*rate*np.dot(g,direction)+64*np.finfo(float).eps*max(abs(value),1e-12):break
                rate*=.5
            else:raise RuntimeError(f'line search: t={t} v={v} it={it} g={np.max(abs(g))} value={value} diff={nv-value} descent={np.dot(g,direction)}')
            ss=nz-z;yy=ng-g;sy=np.dot(ss,yy)
            if sy>1e-25:history.append((ss,yy,1/sy));history=history[-15:]
            z,value,g,obs=nz,nv,ng,no
        if obs['norm_error']>1e-13 or abs(obs['direct_energy_difference'])>1e-12:raise RuntimeError('normalisation or direct energy check')
        if np.max(abs(g))>1e-8:raise RuntimeError(f'gradient {np.max(abs(g))} t={t} v={v}')
        return dict(t=t,v=v,seed=seed,dual=value,gradient_inf=float(np.max(abs(g))),iterations=it,**obs)

def main():
    p=Problem(8,64);rng=np.random.default_rng(46);z=rng.normal(size=np.prod(p.shape))*.003;d=rng.normal(size=z.size);d/=np.linalg.norm(d)
    _,g,_=p.evaluate(z,.03,.02);e=1e-6
    fd=(p.evaluate(z+e*d,.03,.02)[0]-p.evaluate(z-e*d,.03,.02)[0])/(2*e)
    graderr=abs(fd-np.dot(g,d));assert graderr<1e-9,graderr
    rows=[]
    for M in [16,32,64,128]:
        p=Problem(M,192)
        for r in [.04,.02,.01,.005,.0025,.00125]:
            for a,b in [(1,0),(0,1),(1,.5),(1,1),(1,2)]:
                row=p.solve(r*a,r*b);row.update(modes=M,order=192,scale=r,direction=[a,b]);rows.append(row)
                print(M,r,a,b,row['iterations'],row['gradient_inf'],flush=True)
    controls=[]
    for M,order,seed in [(128,256,0),(32,192,1),(32,192,2),(32,192,3)]:
        row=Problem(M,order).solve(.02,.02,seed);row.update(modes=M,order=order);controls.append(row)
    B=.5-4/np.pi**2;chi=4/np.pi-1
    fits=[]
    for M in [16,32,64,128]:
        for r in [.04,.02,.01,.005,.0025,.00125]:
            rr=[x for x in rows if x['modes']==M and x['scale']==r];joint=[x for x in rr if x['t']>0 and x['v']>0]
            design=np.array([[x['d1']**2,2*x['d1']*x['d2'],x['d2']**2] for x in joint]);energy=np.array([x['F'] for x in joint]);co=np.linalg.solve(design,energy)
            both=next(x for x in joint if x['t']==x['v']);one=next(x for x in rr if x['v']==0);two=next(x for x in rr if x['t']==0)
            fits.append(dict(modes=M,scale=r,A=float(co[0]),cross=float(co[1]),C=float(co[2]),mixed_dual=(both['dual']-one['dual']-two['dual'])/r**2))
    report=dict(method='real normalised four-channel tangent Fourier optimisation',gradient_directional_error=graderr,targets=dict(A=1/B,cross=0,C=1/chi),rows=rows,controls=controls,fits=fits)
    report['extrapolations']=[]
    for M in [16,32,64,128]:
        subset=[x for x in fits if x['modes']==M and x['scale']<=.01]
        report['extrapolations'].append(dict(modes=M,**{k:float(np.polyfit([x['scale'] for x in subset],[x[k] for x in subset],2)[-1]) for k in ['A','cross','C']}))
    (OUT/'joint-optimization-results.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(fits[-4:],indent=2))
if __name__=='__main__':main()
