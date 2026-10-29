import numpy as np
from gol import reactions as R, gliders as G, search as S, patterns as P
from gol.compiler import layout as LY
kb=LY.KB; A=LY._KB_ROW; C=LY._KB_COL; NEo=kb.outputs[0]
T0=-160
TR=(0,1,1,0)  # transpose
cands=[]
for (dk,par),r in LY.KILL.items():
    rt=r.transformed(TR)
    se_t=next(t for t in rt.inputs if t.dir=='SE'); ne_t=next(t for t in rt.inputs if t.dir=='NE')
    for dx in range(-60,60):
        # shift so that ne_t lies on NEo: NE sx=1, sy=-1: lane eq: ne.lane + sy*dx - sx*dy = NEo.lane
        dy = (ne_t.lane - dx - NEo.lane)
        dt = NEo.tref - ne_t.tref + 4*dx
        r2=rt.shifted(dx,dy,dt)
        ne2=next(t for t in r2.inputs if t.dir=='NE')
        if ne2!=NEo: continue
        g2=next(t for t in r2.inputs if t.dir=='SE')
        cands.append((dk,par,dx,g2))
cands=[c for c in cands if np.abs(c[3].cells_at(T0)).max()<150 and np.abs(c[3].cells_at(T0+500)).max()<150]
print(len(cands))
cfg1=[np.concatenate([A.cells_at(T0),C.cells_at(T0),g.cells_at(T0)]) for *_,g in cands]
cfg0=[np.concatenate([C.cells_at(T0),g.cells_at(T0)]) for *_,g in cands]
res1=S.simulate_many(cfg1,500); res0=S.simulate_many(cfg0,500)
good=[]
for (dk,par,dx,g),c1,c0 in zip(cands,res1,res0):
    t1,r1=G.identify(c1,T0+500); t0,r0=G.identify(c0,T0+500)
    if not t1 and not len(r1) and t0==sorted([C,g]) and not len(r0):
        good.append((dk,par,dx,g)); 
for x in good: print("OK",x, "theta_g2-theta_A",R.theta(x[3])-R.theta(A),"lane",x[3].lane)
