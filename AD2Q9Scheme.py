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
import jax
import os
Array: TypeAlias = jax.Array
from Type import Lattice, DVS, Array
import sys
import matplotlib.pyplot as plt

def D2Q9Lattice(e) -> Lattice:
   
    name = "D2Q9"
    cx = jnp.asarray([0, 1, 0, -1, 0, 1, -1, -1, 1], dtype="int32")
    cy = jnp.asarray([0, 0, 1, 0, -1, 1, 1, -1, -1], dtype="int32")
    c: Array = jnp.ones((2, 9), dtype="int32")
    c = c.at[0, :].set(cx)
    c = c.at[1, :].set(cy)
    ctranspose = c.transpose()
    w: Array = 1.0 / 36.0 * jnp.ones(9)
    w = w.at[jnp.asarray([jnp.linalg.norm(ci) < 1.1 for ci in ctranspose])].set(1.0 / 9.0)
    w = w.at[0].set(4.0 / 9.0)
    opp = jnp.asarray(
        [ctranspose.tolist().index((-ctranspose[i]).tolist()) for i in range(9)]
    )
    
    return {
        "Name": name,
        "XI": c *e,
        "XIT": ctranspose ,
        "C": c *e,
        "W": w,
        "OPP": opp,
    }

def DefineLattice(name: str,e) -> Lattice:
    
    if name =="D2Q9":
        lattice = D2Q9Lattice(e)
    return lattice


@jit
def collision(f:Array,tau:float,feq:Array,dt:float,force:Array):
    return f -(f-feq )/tau+dt*force

@jit
def collision00(f:Array,tau:float,feq:Array,dt:float,force:Array):
    return f -(f-feq )/tau+dt*(1-1/(2*tau))*force

@jit
def StreamMap(fStage: Array, c: Array,e):
    def streaming_i(f: Array, c: Array):
        return jnp.roll(f, (c[0]//e, c[1]//e), axis=(0, 1))
    return jax.vmap(streaming_i, in_axes=(-1), out_axes=-1)(fStage, c)

@jit
def BGKcollision0(f: Array, feq: Array, tau: Array | float):
    return (feq-f)/tau


@jit
def Euler1ST0(
    f: Array, convection: Array, collision: Array, dt: float, force: Array | float = 0
) -> Array:
   
    return f + dt * (collision - convection + force)


