import numpy as np, pandas as pd, time
from multiprocessing import Pool
from model import *
def conv(cfg):
    name, Ps, kw = cfg; on, rms, f0 = onset_const(Ps, **dict(kw)); return dict(check=name, Ps=Ps, on=on, rms_cm3s=rms*1e6, F0=f0)
def thr(cfg):
    name, kw = cfg; v, n = threshold_bisect(tol=4.0, **dict(kw)); return dict(check=name, Ps_th=v)
def rk4_f0(Ps=800, T=0.6):
    p = params(); f = make_rhs(p, lambda t: Ps*min(1,t/0.005), lambda t: p['x0'])
    h = 1/(44100*4); y = np.array([1e-6,0,1e-6,0]); n = int(T/h); U=[]
    for i in range(n):
        t=i*h; k1=np.array(f(t,y)); k2=np.array(f(t+h/2,y+h/2*k1)); k3=np.array(f(t+h/2,y+h/2*k2)); k4=np.array(f(t+h,y+h*k3))
        y=y+h/6*(k1+2*k2+2*k3+k4)
        if i%4==0: U.append(max(0,2*p['lg']*(p['x0']+min(y[0],y[2])))*np.sqrt(2*Ps/1.2))
    U=np.array(U); t=np.arange(len(U))/44100; return dict(check='RK4 fixed-step h=1/176400', Ps=Ps, on=classify(t,U)[0], rms_cm3s=classify(t,U)[1]*1e6, F0=f0_of(t,U))
if __name__=='__main__':
    t0=time.time()
    cs=[]
    for Ps in (300,800):
        for nm,kw in [('baseline',{}),('rtol1e-6',dict(rtol=1e-6,atol=1e-9)),('rtol1e-10',dict(rtol=1e-10,atol=1e-13)),
                      ('max_step4e-5',dict(max_step=4e-5)),('max_step1e-5',dict(max_step=1e-5)),
                      ('EPS0.5um',dict(EPS=0.5e-6)),('EPS0.25um',dict(EPS=0.25e-6)),('Radau',dict(method='Radau')),('T=2s',dict(T=2.0))]:
            cs.append((nm,Ps,kw))
    ts=[(n,k) for n,k in [('baseline',{}),('rtol1e-6',dict(rtol=1e-6,atol=1e-9)),('max_step4e-5',dict(max_step=4e-5)),('max_step1e-5',dict(max_step=1e-5)),('EPS0.5um',dict(EPS=0.5e-6)),('Radau',dict(method='Radau')),('thr0.5',dict(thr=0.5e-6)),('thr2',dict(thr=2e-6))]]
    with Pool(8) as pool:
        a = pool.map_async(conv, cs); b = pool.map_async(thr, ts); c = pool.apply_async(rk4_f0)
        C = pd.DataFrame(a.get()+[c.get()]); Tt = pd.DataFrame(b.get())
    C.to_csv('part2_convergence_runs.csv',index=False); Tt.to_csv('part2_threshold_convergence.csv',index=False)
    print(C.to_string()); print(Tt.to_string()); print(time.time()-t0)
