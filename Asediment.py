import numpy as np
import jax
import jax.numpy as jnp
from jax import jit
import Model
from Type import Array, DVS
import Scheme
from jax import Array
from typing import TypedDict
from typing import TypeAlias
from typing import Any
lattice: DVS = Model.DefineDVS("D2Q16")
import jax
import os
Array: TypeAlias = jax.Array
from Type import Lattice, DVS, Array
import sys
import matplotlib.pyplot as plt



@jit
def Compute_geq(rho: Array,u: Array,c:Array,e,lamdaxx,lamdayy,lamdaxy):
    
    ux=u.at[:,:,0:1].get()
    uy=u.at[:,:,1:2].get()
    cx=c.at[0:1,5:9].get()
    cy=c.at[1:2,5:9].get()
    
    feq=jnp.broadcast_to(rho,(rho.shape[0],rho.shape[1],9))
    feq=feq.at[:,:,0:1].set( (1-(lamdayy + lamdaxx))*rho)

    feq=feq.at[:,:,1:2].set( (0.5*lamdaxx + ux/( 4 * c.at[0:1,1:2].get())) *rho )
    feq=feq.at[:,:,3:4].set( (0.5*lamdaxx + ux/( 4 * c.at[0:1,3:4].get())) *rho )

    feq=feq.at[:,:,2:3].set( (0.5*lamdayy + uy/( 4 * c.at[1:2,2:3].get())) *rho )
    feq=feq.at[:,:,4:5].set( (0.5*lamdayy + uy/( 4 * c.at[1:2,4:5].get())) *rho )

    feq=feq.at[:,:,5:9].set( (0.25*e*e*lamdaxy/( cx*cy) + ux/(8*cx) + uy/(8*cy)) * rho )

    return feq

@jit
def CalcMacroVars_sedment(f: Array,h:Array) -> tuple[Array, Array]:
    hc = jnp.sum(f, axis=-1, keepdims=True)
    return hc/h

@jit
def CalcMacroVars_sedment00(f: Array,si,dt) -> tuple[Array, Array]:
    c = jnp.sum(f, axis=-1, keepdims=True)
    SI=jnp.sum(si, axis=-1, keepdims=True)
    return c

@jit
def Compute_geq_normal(W,rho,XI,u,e,h):
    Cs2=(e*e)/3
    return h*W*rho*(1 + ( XI.at[0:1,:].get()*u.at[:,:,0:1].get() + XI.at[1:2,:].get()*u.at[:,:,1:2].get() )/ Cs2 )

