import numpy as np, pandas as pd, time, sys
from multiprocessing import Pool
from model import *
PT, XT = 800.0, 0.2e-3
def rc(t, T): return 0.5*(1-np.cos(np.pi*min(max(t/T,0),1)))
def paths(path, Tr, Pt=PT, xt=XT):
    """returns Ps(t), x0(t), and t_full (time at which target values are reached)"""
    xp = -0.05e-3; r = 1e-3
    if path=='H0':   # step pressure in 1 ms, hold pressed for Tr, release in 1 ms (interpretation)
        return (lambda t: Pt*min(t/r,1)), (lambda t: xp+(xt-xp)*min(max((t-Tr)/r,0),1)), Tr+r
    if path=='H1':   # linear ramp over Tr, pressed until ramp end, 1 ms release
        return (lambda t: Pt*min(t/Tr,1)), (lambda t: xp+(xt-xp)*min(max((t-Tr)/r,0),1)), Tr+r
    if path=='E1':   # raised-cosine pressure, gap closes 0.6 -> 0.2 mm with raised cosine
        return (lambda t: Pt*rc(t,Tr)), (lambda t: 0.6e-3+(xt-0.6e-3)*rc(t,Tr)), Tr
    if path=='E2':
        return (lambda t: Pt*rc(t,Tr)), (lambda t: xt), Tr
def run(cfg):
    path, Tr, Q, fac, tag = cfg
    kw = dict(Q=Q)
    if fac is not None:
        for k,f in zip(['k1','k2','kc','m1','m2'], fac[:5]): kw[k]=BASE[k]*f
    p = params(**kw)
    Pf, Xf, tf = paths(path, Tr, PT*(fac[6] if fac is not None else 1), XT*(fac[5] if fac is not None else 1))
    T = Tr + 1.0
    t,y,U,Ps = simulate(p, Pf, T, x0_fun=Xf)
    succ, rms = classify(t, U)
    tc, r = window_rms(U); above = r > 1e-6
    if above[-1]:
        k = len(above)-1
        while k>0 and above[k-1]: k -= 1
        t_on = tc[k]-0.005  # start of first window of the final above-threshold stretch
        P_on = Pf(t_on)
    else: t_on, P_on = np.nan, np.nan
    return dict(tag=tag, path=path, Tr_ms=Tr*1e3, Q=Q, success=succ, rms_cm3s=rms*1e6, t_onset_ms=t_on*1e3,
                t_after_target_ms=(t_on-tf)*1e3, Ps_at_onset=P_on, Ps_peak=float(Ps.max()))
if __name__=='__main__':
    t0=time.time(); which=sys.argv[1]
    P4=['H0','H1','E1','E2']
    if which=='2':
        cfgs=[(pa,Tr,Q,None,'exp2') for pa in P4 for Tr in [0.001,0.025,0.05,0.1,0.2,0.4] for Q in [0.8,1.0,1.4]]
        with Pool(8) as pool: d=pd.DataFrame(pool.map(run,cfgs,chunksize=1))
        d.to_csv('exp2_paths.csv',index=False)
    else:
        N=int(sys.argv[2]); cfgs=[]
        for pa in P4:
            for Tr in [0.05,0.2]:
                for i in range(N):
                    rng=np.random.default_rng(1000+i)   # same seed i -> same parameter draw across conditions (paired)
                    cfgs.append((pa,Tr,1.0,rng.uniform(0.9,1.1,7),f'draw{i}'))
        with Pool(8) as pool: d=pd.DataFrame(pool.map(run,cfgs,chunksize=4))
        d.to_csv('exp3_robustness.csv',index=False)
    print(d.to_string()); print(time.time()-t0)
