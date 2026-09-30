"""Out-of-class (nonlinear gate) chaos test. TRUE error gate is g=psi_a1^2 (well-excited, NOT aliasable) (a
cross-product, NOT in the discovery model's linear class {1,theta_a1,T_o1,psi_a1}), so
the truth is unrepresentable -- a genuine model-form error. We fit the linear discovery
model anyway and ask, via multistart, whether the loss surface stays UNIMODAL (all
starts converge to the same best linear fit = decoupling survives model error) or goes
MULTIMODAL past the predictability horizon (chaos+misspecification breaks it).
Metric per window: best residual (out-of-class floor), and consistency = fraction of 10
random starts landing within 5% of the lowest-cost solution. Runs 36 and 58. Incremental save."""
import os, numpy as np, jax, jax.numpy as jnp
from error_discovery.models import TwinModels
from qgs.params.params import QgParams
from qgs_jax.spec import (DEFAULT_MAOOAM_PARAMS,DEFAULT_ATEMP_PARAMS,DEFAULT_GOTEMP_PARAMS,DEFAULT_A_INSOLATION,DEFAULT_O_INSOLATION)
from error_discovery.discovery import rollout, make_residual, lm_fit, obs_indices
from error_discovery.config import DT, steps_to_days
HERE=os.path.dirname(os.path.abspath(__file__)); EPS=0.1
# The ~30k-yr states; were converged_ndim<N>.npz, renamed 2026-09-24 (notes/rename_map.md).
CONVERGED = {36: "attractor_ic_default_sp1030000000.npz",
             56: "attractor_ic_default-ndim56_sp980000000.npz",
             58: "attractor_ic_default-ndim58_sp910000000.npz"}
WINDOWS=[160,320,480,640,960,1280,1920,2560,3840,5120,8000]
def bp(ax,ay):
    p=QgParams(); p.set_atmospheric_channel_fourier_modes(ax,ay); p.set_oceanic_basin_fourier_modes(2,4)
    p.set_params(DEFAULT_MAOOAM_PARAMS); p.atemperature_params.set_params(DEFAULT_ATEMP_PARAMS); p.gotemperature_params.set_params(DEFAULT_GOTEMP_PARAMS)
    p.atemperature_params.set_insolation(*DEFAULT_A_INSOLATION); p.gotemperature_params.set_insolation(*DEFAULT_O_INSOLATION); return p
print("JAX",jax.devices(),flush=True)
for ax,ay in [(2,2),(3,3)]:
    tw=TwinModels(0.1,(0.,0.,1.,0.),params=bp(ax,ay)); ndim=tw.ndim
    ipsi=tw.idx_psia1; E=tw.E_full
    def f_true_nl(x):                       # out-of-class nonlinear gate g=psi_a1^2 (well-excited, NOT aliasable)
        return tw.f_base(x) + EPS*(x[ipsi]*x[ipsi])*(E@x)
    xa=jnp.asarray(np.load(os.path.join(HERE,CONVERGED[ndim]))["x"])
    mask=jnp.ones(ndim); oidx=obs_indices(mask); days=[];floor=[];cons=[]
    for W in WINDOWS:
        Y=np.asarray(rollout(f_true_nl, xa, DT, W))
        resid=make_residual(tw,xa,Y,DT,oidx); rng=np.random.default_rng(0)
        chats=[]; costs=[]
        for s in range(10):
            chat,_,hist=lm_fit(resid, jnp.asarray(rng.normal(0,0.05,4)), n_iters=80)
            chats.append(np.asarray(chat)); costs.append(float(hist[-1]))
        chats=np.array(chats); best=chats[int(np.argmin(costs))]
        consistent=np.mean([np.linalg.norm(c-best)/(np.linalg.norm(best)+1e-12)<0.05 for c in chats])
        days.append(steps_to_days(W)); floor.append(np.sqrt(2*min(costs))); cons.append(consistent)
        print("psi2 ndim%d W=%5d (%5.1fd) best_resid=%.2e consistency=%.0f/10"%(ndim,W,steps_to_days(W),np.sqrt(2*min(costs)),consistent*10),flush=True)
        np.savez(os.path.join(HERE,"chaos_psi2_%d.npz"%ndim), days=np.array(days),floor=np.array(floor),cons=np.array(cons))
print("DONE",flush=True)
