import os, numpy as np, jax, jax.numpy as jnp
from error_discovery.models import TwinModels
from qgs.params.params import QgParams
from qgs_jax.spec import (DEFAULT_MAOOAM_PARAMS,DEFAULT_ATEMP_PARAMS,DEFAULT_GOTEMP_PARAMS,DEFAULT_A_INSOLATION,DEFAULT_O_INSOLATION)
from error_discovery.discovery import rollout, make_residual, lm_fit, recover, obs_indices
from error_discovery.config import DT, steps_to_days
HERE=os.path.dirname(os.path.abspath(__file__))
p=QgParams(); p.set_atmospheric_channel_fourier_modes(3,3); p.set_oceanic_basin_fourier_modes(2,4)
p.set_params(DEFAULT_MAOOAM_PARAMS); p.atemperature_params.set_params(DEFAULT_ATEMP_PARAMS); p.gotemperature_params.set_params(DEFAULT_GOTEMP_PARAMS)
p.atemperature_params.set_insolation(*DEFAULT_A_INSOLATION); p.gotemperature_params.set_insolation(*DEFAULT_O_INSOLATION)
tw=TwinModels(0.1,(0.,0.,1.,0.),params=p)
xa=jnp.asarray(np.load(os.path.join(HERE,"attractor_ic_default-ndim58_sp910000000.npz"))["x"]); ctrue=np.asarray(tw.c_true)
mask=jnp.ones(tw.ndim); oidx=obs_indices(mask)
WINDOWS=[80,160,320,480,640,960,1280,1920,2560,3840,5120,8000]
days=[];rel=[];ms=[]
for W in WINDOWS:
    Y=np.asarray(rollout(lambda x: tw.f_true(x), xa, DT, W))
    r=recover(tw,xa,Y,DT,mask,eta=0.0,n_iters=60)
    resid=make_residual(tw,xa,Y,DT,oidx); rng=np.random.default_rng(0); hits=0
    for s in range(10):
        chat,_,_=lm_fit(resid, jnp.asarray(rng.normal(0,0.05,4)), n_iters=60)
        hits+=(np.linalg.norm(np.asarray(chat)-ctrue)/np.linalg.norm(ctrue)<0.05)
    days.append(steps_to_days(W)); rel.append(r["rel_err"]); ms.append(hits/10.0)
    print("ndim58 W=%5d (%5.1fd) rel_err=%.2e multistart %d/10"%(W,steps_to_days(W),r["rel_err"],hits),flush=True)
    np.savez(os.path.join(HERE,"chaotic_win_58.npz"), days=np.array(days),rel=np.array(rel),ms=np.array(ms),horizon=11.0)
print("DONE",flush=True)
