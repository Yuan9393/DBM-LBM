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

def D2Q5Lattice(e) -> Lattice:
    """Define D2Q5 lattice"""
    name = "D2Q5"
    cx = jnp.asarray([0, 1, 0, -1, 0], dtype="int32")
    cy = jnp.asarray([0, 0, 1, 0, -1], dtype="int32")
    c: Array = jnp.ones((2, 5), dtype="int32")
    c = c.at[0, :].set(cx)
    c = c.at[1, :].set(cy)
    ctranspose = c.transpose()
    w: Array = 1.0 / 6.0 * jnp.ones(5)
    w = w.at[0].set(1.0 / 3.0)
    opp = jnp.asarray(
        [ctranspose.tolist().index((-ctranspose[i]).tolist()) for i in range(5)]
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
    """Return a predefined lattice

    Args:
        name (str): the name for the needed lattice

    Returns:
        Lattice:
    """
    if name =="D2Q5":
        lattice = D2Q5Lattice(e)
    return lattice


@jit
def collision(f:Array,tau:float,feq:Array,dt:float,force:Array):
    return f -(f-feq )/tau+dt*force



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
    """ The first order euler scheme for df/dt=-cdf/dx+collision+body force
    """
    return f + dt * (collision - convection + force)


