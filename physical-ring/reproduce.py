"""Rebuild figure data from saved solver records; validate all displayed points.
No optimization is run. Numerical agreement is not a rigorous error certificate.
"""
from pathlib import Path
import argparse, collections, csv, json, re
import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.optimize import brentq

HERE = Path(__file__).resolve().parent
EPS = .5 - 1 / np.pi
B = .5 - 4 / np.pi**2
C = (2 - np.sqrt(2))**2 / np.pi**2

def acceptable(r):
    return ('energy' in r and r['gap'] < 2e-5 and r['min_eig'] > -2e-7
            and r['dual_min_eig'] > -2e-7 and r['norm_res'] < 2e-7
            and r['error_violation'] < 2e-7)

def records(folder, names):
    return [json.loads(line) for name in names
            for line in (folder / (name + '.jsonl')).read_text().splitlines()]

def write_csv(path, rows):
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=HERE/'data')
    parser.add_argument('--out', type=Path, default=HERE/'build')
    args = parser.parse_args(); args.out.mkdir(parents=True, exist_ok=True)
    grid = json.loads((args.data_dir/'grid.json').read_text())
    aa, ee = np.array(grid['a']), np.array(grid['eps'])
    raw = records(args.data_dir, ['map', 'repair', 'refine'])
    selected = {}
    for r in raw:
        if acceptable(r): selected[r['N'], r['mu'], r['eps']] = r
    cost = np.zeros((len(aa), len(ee))); difference = cost.copy()
    rows = []; chosen = []; clipped = []
    for i, a in enumerate(aa):
        for j, e in enumerate(ee):
            analytic = e == .5 or (a == .5 and e >= EPS)
            if not analytic:
                r = selected[24, a, e]; chosen.append(r)
                if r['energy'] < 0: clipped.append(r)
                cost[i,j] = max(0, r['energy'])  # historical map selection rule
                difference[i,j] = abs(r['energy']-selected[16,a,e]['energy'])
            rows.append(dict(a=a, epsilon=e, cost=cost[i,j],
                             source='analytic_zero' if analytic else 'SDP_N24',
                             cutoff_difference=difference[i,j]))
    saved = np.load(args.data_dir/'map-data.npz')
    np.testing.assert_allclose(cost, saved['cost'], atol=1e-14, rtol=0)
    np.testing.assert_allclose(difference, saved['cutoff_difference'], atol=1e-14, rtol=0)
    assert len(chosen)==435
    write_csv(args.out/'map.csv', rows)
    np.savez(args.out/'map-data.npz', a=aa, eps=ee, cost=cost, cutoff_difference=difference)
    laws = {k:[] for k in ['accuracy','energy','momentum','boundary']}
    # The published local points used map+repair, before refinement diagnostics.
    local = {}
    for r in records(args.data_dir, ['map','repair']):
        if acceptable(r): local[r['N'],r['mu'],r['eps']] = r
    for (n, a, e), r in local.items():
        if n != 24: continue
        delta, omega = EPS-e, .5-a
        if a == .5 and 0 < delta < .065:
            laws['accuracy'].append((delta,r['energy']))
        if abs(e-EPS)<1e-12 and 0<omega<=.100001:
            laws['energy'].append((omega,omega-r['energy']))
            laws['momentum'].append((omega,r['transfer']+.5))
    root_rows=records(args.data_dir,['roots'])
    for r in root_rows:
        if r['N']==24: laws['boundary'].append((r['nu'],r['delta']))
    interp=RegularGridInterpolator((aa,ee),cost,method='pchip')
    def full(mu,e):
        a=abs(mu-np.floor(mu+.5))
        return float(interp([[a,e]])[0])+a*a-mu*mu
    def root(e):
        return .5 if e>=EPS else brentq(lambda m: full(m,e),.5,1.5,xtol=1e-11)
    laws['global']=[(e,root(e)) for e in ee if .04<e<EPS]
    excluded=root(.04); upper=(1+C/.04)/2
    assert excluded>upper
    write_csv(args.out/'excluded-threshold.csv',[dict(epsilon=.04,interpolated=excluded,
              rigorous_upper=upper,excess=excluded-upper,displayed=False)])
    leading={'accuracy':lambda x:x*x/B,'energy':lambda x:.75*(np.pi**2*B)**(1/3)*x**(4/3),
             'momentum':lambda x:.5*(np.pi**2*B*x)**(1/3),'boundary':lambda x:np.sqrt(B*x)}
    for key, points in laws.items():
        laws[key]=sorted(points)
        write_csv(args.out/(key+'.csv'), [dict(x=x,value=y,leading=leading[key](x) if key in leading else '') for x,y in laws[key]])
    # Verify against literal coordinates in standalone copies of the CURRENT figures.
    mapping={'06-threshold':['accuracy'],'07-detuning':['energy','momentum'],
             '08-boundary':['boundary'],'09-global-threshold':['global']}
    verified={}
    for name, keys in mapping.items():
        text=(HERE/'figures'/(name+'.tex')).read_text()
        blocks=re.findall(r'\\addplot\[only marks,mark=\*,mark size=1\.[36]pt,digitZero\] coordinates \{([^}]+)\}',text)
        # Figure 6 has an additional exact marker before its numerical panel.
        if name=='06-threshold': blocks=blocks[-1:]
        assert len(blocks)==len(keys),(name,len(blocks))
        for block,key in zip(blocks,keys):
            points=sorted((float(x),float(y)) for x,y in re.findall(r'\(([^,]+),([^\)]+)\)',block))
            np.testing.assert_allclose(points,laws[key],atol=6e-11,rtol=0)
            verified[key]=len(points)
    audit=dict(selected_N24=len(chosen),status_counts=dict(collections.Counter(r['status'] for r in chosen)),
       max_gap=max(r['gap'] for r in chosen),max_norm_res=max(r['norm_res'] for r in chosen),
       max_error_violation=max(r['error_violation'] for r in chosen),
       min_primal_eigenvalue=min(r['min_eig'] for r in chosen),
       max_cutoff_difference=float(difference.max()),
       max_cutoff_difference_eps_ge_008=float(difference[:,ee>=.08].max()),
       negative_costs_clipped=len(clipped),verified_figure_points=verified,
       excluded_epsilon_004=dict(interpolated=excluded,rigorous_upper=upper),
       max_root_energy_residual_N24=max(abs(r['residual']) for r in root_rows if r['N']==24))
    reference=json.loads((args.data_dir/'audit-summary.json').read_text())
    for key in ['selected_N24','status_counts','max_gap','max_norm_res','max_error_violation',
                'min_primal_eigenvalue','max_cutoff_difference','max_cutoff_difference_eps_ge_008']:
        assert audit[key]==reference[key],key
    (args.out/'verification.json').write_text(json.dumps(audit,indent=2)+'\n')
    print(json.dumps(audit,indent=2))

if __name__=='__main__': main()
