import numpy as np, math, json, sys
from analysis.portalEpsScan import build_pot, SET8_BOSON_PHI, SET8_FERMION_PHI, SET8_V0
from analysis.portalBounce import scan, analyze
from analysis.twoStepBounce import percolation_T
V0=SET8_V0; v=math.sqrt(2*V0)/1000.0   # m_phi = 1 TeV
out=[]
for mS in [float(x) for x in sys.argv[1].split(',')]:
    s0=math.sqrt(2*1e-12*V0)/mS
    pot=build_pot(V0, v, s0, 1e-12, 0.1, SET8_BOSON_PHI, SET8_FERMION_PHI, [(20.0,1.09)], 100.0, phi_thermal="set8")
    Ts=np.geomspace(2.0*max(pot.T0_S(),pot.T0_phi()), 0.6*min(pot.T0_S(),pot.T0_phi()), 24)
    res=scan(pot,Ts,refine=10,verbose=False,workers=4,do_2d=False); info=analyze(pot,res,b4=None)
    s1=info.get('S step (0->a)',{}); d=info.get('phi direct (0->v, single-step reference)',{})
    Tn=s1.get('T_n',np.nan); Tp=percolation_T(pot,res,"0->a",V0) if np.isfinite(Tn) else np.nan
    r=dict(m_S=mS,T0_S=pot.T0_S(),T0_phi=pot.T0_phi(),Tn1=Tn,b1=s1.get('beta/H',np.nan),Tperc1=Tp,Tdir=d.get('T_n',np.nan),bdir=d.get('beta/H',np.nan))
    out.append(r); print({k:(round(x,1) if isinstance(x,float) else x) for k,x in r.items()}, flush=True)
json.dump(out,open('data/portal/sstep_newbath_mphi1000.json','w'),indent=1)
