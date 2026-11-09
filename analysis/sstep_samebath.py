"""S step with the S bath identical to the phi bath (set8), m_phi = 1 TeV.
mode shared: V0 = 2.5e35, eps = 1e-12, scan m_S.   mode dti: eps -> 1, scan V0 at given m_S list."""
import numpy as np, math, json, sys
from analysis.portalEpsScan import build_pot, SET8_BOSON_PHI, SET8_FERMION_PHI
from analysis.portalBounce import scan, analyze
from analysis.twoStepBounce import percolation_T
mode=sys.argv[1]; out=[]
def run(V0, eps, mS):
    v=math.sqrt(2*V0)/1000.0; s0=math.sqrt(2*eps*V0)/mS
    pot=build_pot(V0, v, s0, eps, 0.1, SET8_BOSON_PHI, SET8_FERMION_PHI, [(6.0,1.0)], 100.0, phi_thermal="set8", S_thermal="set8")
    Ts=np.geomspace(1.5*max(pot.T0_S(),pot.T0_phi()), 0.8*min(pot.T0_S(),pot.T0_phi()), 28)
    res=scan(pot,Ts,refine=12,verbose=False,workers=4,do_2d=False); info=analyze(pot,res,b4=None)
    s1=info.get('S step (0->a)',{}); d=info.get('phi direct (0->v, single-step reference)',{})
    Tn=s1.get('T_n',np.nan); Tp=percolation_T(pot,res,"0->a",V0) if np.isfinite(Tn) else np.nan
    r=dict(V0=V0,eps=eps,m_S=mS,T0_S=pot.T0_S(),Tn1=Tn,b1=s1.get('beta/H',np.nan),Tperc1=Tp,Tdir=d.get('T_n',np.nan),bdir=d.get('beta/H',np.nan))
    print({k:(float(f"{x:.4g}") if isinstance(x,float) else x) for k,x in r.items()}, flush=True); return r
if mode=='shared':
    for mS in (1000,1050,1100,1150,1200,1300,1400,1600,1800,2000,2500): out.append(run(2.5e35,1e-12,float(mS)))
else:
    for mS in (1100.,1200.,1300.):
        for V0 in (5e25,5e27,5e29,2.5e35): out.append(run(V0,1-1e-6,mS))
json.dump(out,open(f'data/portal/sstep_samebath_{mode}.json','w'),indent=1)
