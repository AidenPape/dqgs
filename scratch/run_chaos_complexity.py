"""Chaos-decoupling test for a HARDER error: window sweep + multistart for a given
g_coeffs, on 36-dim (weakly chaotic) and 58-dim (truly chaotic) converged attractors.
Tests whether 'find the track past the horizon' survives a multi-coefficient,
conditioning-challenged error. Incremental save."""
import os, sys, numpy as np, jax, jax.numpy as jnp
from error_discovery.models import TwinModels
from qgs.params.params import QgParams
from qgs_jax.spec import (DEFAULT_MAOOAM_PARAMS,DEFAULT_ATEMP_PARAMS,DEFAULT_GOTEMP_PARAMS,DEFAULT_A_INSOLATION,DEFAULT_O_INSOLATION)
from error_discovery.discovery import rollout, make_residual, lm_fit, recover, obs_indices
from error_discovery.config import DT, steps_to_days
HERE=os.path.dirname(os.path.abspath(__file__))
# The ~30k-yr states; were converged_ndim<N>.npz, renamed 2026-09-24 (notes/rename_map.md).
CONVERGED = {36: "attractor_ic_default_sp1030000000.npz",
             56: "attractor_ic_default-ndim56_sp980000000.npz",
             58: "attractor_ic_default-ndim58_sp910000000.npz"}
GC=tuple(float(c) for c in sys.argv[1].split(","))   # e.g. 0,1,1,1
LABEL=sys.argv[1].replace(",","")
WINDOWS=[160,320,480,640,960,1280,1920,2560,3840,5120,8000]
def bp(ax,ay):
    p=QgParams(); p.set_atmospheric_channel_fourier_modes(ax,ay); p.set_oceanic_basin_fourier_modes(2,4)
    p.set_params(DEFAULT_MAOOAM_PARAMS); p.atemperature_params.set_params(DEFAULT_ATEMP_PARAMS); p.gotemperature_params.set_params(DEFAULT_GOTEMP_PARAMS)
    p.atemperature_params.set_insolation(*DEFAULT_A_INSOLATION); p.gotemperature_params.set_insolation(*DEFAULT_O_INSOLATION); return p
for ax,ay in [(2,2),(3,3)]:
    tw=TwinModels(0.1,GC,params=bp(ax,ay)); ndim=tw.ndim
    xa=jnp.asarray(np.load(os.path.join(HERE,CONVERGED[ndim]))["x"]); ctrue=np.asarray(tw.c_true)
    mask=jnp.ones(ndim); oidx=obs_indices(mask); days=[];rel=[];ms=[]
    for W in WINDOWS:
        Y=np.asarray(rollout(lambda x: tw.f_true(x), xa, DT, W))
        r=recover(tw,xa,Y,DT,mask,eta=0.0,n_iters=80)
        resid=make_residual(tw,xa,Y,DT,oidx); rng=np.random.default_rng(0); hits=0
        for s in range(10):
            chat,_,_=lm_fit(resid, jnp.asarray(rng.normal(0,0.05,4)), n_iters=80)
            hits+=(np.linalg.norm(np.asarray(chat)-ctrue)/np.linalg.norm(ctrue)<0.05)
        days.append(steps_to_days(W)); rel.append(r["rel_err"]); ms.append(hits/10.0)
        print("g=%s ndim%d W=%5d (%5.1fd) rel_err=%.2e multistart %d/10"%(LABEL,ndim,W,steps_to_days(W),r["rel_err"],hits),flush=True)
        np.savez(os.path.join(HERE,"chaos_%s_%d.npz"%(LABEL,ndim)), days=np.array(days),rel=np.array(rel),ms=np.array(ms),gc=np.array(GC))
print("DONE",flush=True)
