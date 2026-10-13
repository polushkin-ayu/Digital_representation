"""Generate PGFPlots figure and table directly from results.json."""
from pathlib import Path
import json
p=Path(__file__).resolve().parent
r=json.loads((p/'results.json').read_text());out=p.parent/'figures'
out.mkdir(parents=True, exist_ok=True)
q=sorted([x for x in r['quadrature'] if x['order']==256 and x['t']<=.1],key=lambda x:x['delta'])
f=sorted([x for x in r['fourier'] if x['modes']==64 and x['order']==256 and x['seed']==0],key=lambda x:x['delta'])
c=[x for x in r['fourier'] if x['t']==.02 and x['seed']==0 and x['order']==256]
ref=next(x['ratio'] for x in q if x['t']==.02)
def coords(rows,x,y):return ' '.join(f'({x(a):.14g},{y(a):.14g})' for a in rows)
s=r'''\begin{tikzpicture}
\begin{groupplot}[group style={group size=2 by 1,horizontal sep=1.45cm},
 width=0.45\textwidth,height=5.5cm, tick label style={font=\footnotesize},
 label style={font=\small},title style={font=\small},grid=major,
 legend style={font=\scriptsize,draw=none,fill=white},scaled ticks=false]
\nextgroupplot[title={(а) Асимптотический коэффициент},xlabel={$\delta$},ylabel={$F/\delta^2$},
 xmin=0,xmax=.005,ymin=10.55,ymax=10.74,xtick={0,.002,.004},ytick={10.56,10.64,10.72},
 tick label style={/pgf/number format/fixed,/pgf/number format/precision=3},legend pos=north west]
\addplot[blue!70!black,mark=*,mark size=1.7pt,thin] coordinates {QUAD};
\addlegendentry{Квадратура}
\addplot[orange!80!black,only marks,mark=square,mark size=2.5pt] coordinates {FOURIER};
\addlegendentry{Фурье, $N=64$}
\addplot[black,dashed,domain=0:.005] {TARGET};
\addlegendentry{$1/B$}
\nextgroupplot[title={(б) Сходимость по гармоникам},xlabel={$N$},ylabel={$|R_N-R_{\rm quad}|$},
 xmode=log,ymode=log,log basis x=2,xtick={8,16,32,64},xticklabels={8,16,32,64},
 xmin=7,xmax=72,ymin=3e-6,ymax=.006]
\addplot[blue!70!black,mark=*,thick] coordinates {ERROR};
\end{groupplot}
\end{tikzpicture}
'''
s=s.replace('QUAD',coords(q,lambda a:a['delta'],lambda a:a['ratio'])).replace('FOURIER',coords(f,lambda a:a['delta'],lambda a:a['ratio'])).replace('TARGET',str(r['target'])).replace('ERROR',coords(c,lambda a:a['modes'],lambda a:abs(a['ratio']-ref)))
(out/'convergence.tex').write_text(s)
rows=[]
for a in c:
 err=abs(a['ratio']-ref);mant,exp=f'{err:.2e}'.split('e')
 rows.append(f"{a['modes']} & {a['ratio']:.9f} & ${mant}\\times10^{{{int(exp)}}}$ \\\\")
(out/'convergence-rows.tex').write_text(r'\begin{tabular}{rcc}\hline $N$ & $R_N=F_N/\delta_N^2$ & $|R_N-R_{\rm quad}|$\\\hline'+'\n'+'\n'.join(rows)+'\n'+r'\hline\end{tabular}'+'\n')
print('Figure and table exported')
