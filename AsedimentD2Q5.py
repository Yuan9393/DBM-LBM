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
def Compute_geq(rho: Array,u: Array,c:Array,e,lamdaxx,lamdayy):
    
    ux=u.at[:,:,0:1].get()
    uy=u.at[:,:,1:2].get()

    feq=jnp.broadcast_to(rho,(rho.shape[0],rho.shape[1],5))
    feq=feq.at[:,:,0:1].set( (1-(lamdayy + lamdaxx))*rho )
    feq=feq.at[:,:,1:2].set( (0.5*lamdaxx + (c.at[0:1,1:2].get()*ux+c.at[1:2,1:2].get()*uy)/( 2*e*e ) ) *rho)
    feq=feq.at[:,:,3:4].set( (0.5*lamdaxx + (c.at[0:1,3:4].get()*ux+c.at[1:2,3:4].get()*uy)/( 2*e*e ) ) *rho )

    feq=feq.at[:,:,2:3].set( (0.5*lamdayy + (c.at[0:1,2:3].get()*ux+c.at[1:2,2:3].get()*uy)/( 2*e*e )) *rho )
    feq=feq.at[:,:,4:5].set( (0.5*lamdayy + (c.at[0:1,4:5].get()*ux+c.at[1:2,4:5].get()*uy)/( 2*e*e )) *rho)

    return feq

@jit
def CalcMacroVars_sedment(f: Array) -> tuple[Array, Array]:
    c = jnp.sum(f, axis=-1, keepdims=True)
    return c