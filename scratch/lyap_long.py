"""Long-averaged Benettin lambda on converged attractors (check the 18-yr estimate's
plateau). Saves running-lambda arrays + final values to lyap_long_results.npz."""
import os, numpy as np, jax, jax.numpy as jnp
from error_discovery.models import TwinModels
from qgs.params.params import QgParams
from qgs_jax.spec import (DEFAULT_MAOOAM_PARAMS,DEFAULT_ATEMP_PARAMS,DEFAULT_GOTEMP_PARAMS,DEFAULT_A_INSOLATION,DEFAULT_O_INSOLATION)
from qgs_jax import model as jm
from error_discovery.config import DT, steps_to_days
HERE=os.path.dirname(os.path.abspath(__file__)); BLOCK=200; N_BLOCKS=25000; D0=1e-8
def bp(ax,ay):
    p=QgParams(); p.set_atmospheric_channel_fourier_modes(ax,ay); p.set_oceanic_basin_fourier_modes(2,4)
    p.set_params(DEFAULT_MAOOAM_PARAMS); p.atemperature_params.set_params(DEFAULT_ATEMP_PARAMS); p.gotemperature_params.set_params(DEFAULT_GOTEMP_PARAMS)
    p.atemperature_params.set_insolation(*DEFAULT_A_INSOLATION); p.gotemperature_params.set_insolation(*DEFAULT_O_INSOLATION); return p
def benettin(T,x0,ndim,seed):
    step=jax.jit(lambda y: jm.integrate_final(T,y,DT,BLOCK))
    rng=np.random.default_rng(seed); p=rng.standard_normal(ndim); p=D0*p/np.linalg.norm(p)
    x=jnp.asarray(np.asarray(x0)); xp=x+jnp.asarray(p); logs=np.empty(N_BLOCKS); run=np.empty(N_BLOCKS)
    for b in range(N_BLOCKS):
        x=step(x); xp=step(xp); d=float(jnp.linalg.norm(xp-x)); logs[b]=np.log(d/D0); xp=x+(D0/d)*(xp-x)
        run[b]=logs[:b+1].mean()/(BLOCK*DT)
    return run
print("JAX",jax.devices(),flush=True); out={}
for ax,ay in [(2,2),(2,4),(3,3)]:
    tw=TwinModels(0.1,(0.,0.,1.,0.),params=bp(ax,ay))
    x0=jnp.asarray(np.load(os.path.join(HERE,"converged_ndim%d.npz"%tw.ndim))["x"])
    run=benettin(tw.T_base,x0,tw.ndim,1); lam=float(run[-1]); h=steps_to_days((1/lam)/DT) if lam>0 else np.inf
    out["run%d"%tw.ndim]=run; out["lam%d"%tw.ndim]=lam
    print("ndim %d: lambda=%.3e horizon=%.1f d (avg over %.0f yr)"%(tw.ndim,lam,h,N_BLOCKS*BLOCK/89.165/365),flush=True)
out["yrs"]=np.arange(1,N_BLOCKS+1)*BLOCK/89.165/365
np.savez(os.path.join(HERE,"lyap_long_results.npz"),**out); print("saved",flush=True)
