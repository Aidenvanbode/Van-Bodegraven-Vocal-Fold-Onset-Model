import sys; sys.path.insert(0,'../step4c'); import numpy as np, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from model_c import *
XT,XP=0.2e-3,-0.05e-3
fig,axs=plt.subplots(4,2,figsize=(11,9),sharex='col')
for col,(st,Pth) in enumerate((('A',259.8),('B',195.3))):
    Pt=1.2*Pth; Tr=0.025; p=params(stage=st)
    Pf=lambda t:Pt*min(t/Tr,1); Xf=lambda t:XP+(XT-XP)*min(max((t-Tr)/1e-3,0),1)
    t,y,U,Ps=simulate(p,Pf,0.4,x0_fun=Xf,seed=5000)
    tc,r=window_rms(U); 
    axs[0,col].plot(t*1e3,Ps); axs[0,col].set_ylabel('Ps'); axs[0,col].set_title(f'Stage {st} hard, Tr 25 ms')
    axs[1,col].plot(t*1e3,(y[0]+np.array([Xf(x) for x in t]))*1e3,label='x0+x1'); axs[1,col].plot(t*1e3,(y[2]+np.array([Xf(x) for x in t]))*1e3,label='x0+x2'); axs[1,col].legend(); axs[1,col].set_ylabel('half-width (mm)')
    axs[2,col].plot(t*1e3,U*1e6); axs[2,col].set_ylabel('U cm3/s')
    axs[3,col].semilogy(tc*1e3,r*1e6,'.-'); axs[3,col].axhline(1); axs[3,col].set_ylabel('win RMS')
    for tt in np.arange(0,0.06,0.005): i=int(tt*FS); print(st, f'{tt*1e3:4.0f}ms Ps={Ps[i]:6.1f} x1+x0={1e3*(y[0][i]+Xf(t[i])):.4f} x2+x0={1e3*(y[2][i]+Xf(t[i])):.4f} U={U[i]*1e6:.2f}')
    print(st,'win rms first 8:',np.round(r[:8]*1e6,2))
axs[3,0].set_xlim(0,150); axs[3,1].set_xlim(0,150); fig.tight_layout(); fig.savefig('diag_hard_trace.png',dpi=110)
