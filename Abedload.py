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
def CalcMacroVars_bed(f: Array) -> tuple[Array, Array]:
    zb = jnp.sum(f, axis=-1, keepdims=True)
    return zb

@jit
def compute_leq(zb:Array,porosity:float,c:Array,e,qb):
    leq=jnp.broadcast_to(zb,(zb.shape[0],zb.shape[1],9))
    leq=leq.at[:,:,0:1].set( zb.at[:,:,0:1].get())
    leq=leq.at[:,:,1:5].set( (c.at[0:1,1:5].get()*qb.at[:,:,0:1].get() + c.at[1:2,1:5].get()*qb.at[:,:,1:2].get()  )/((1-porosity)*3*e*e) )
    leq=leq.at[:,:,5:9].set( (c.at[0:1,5:9].get()*qb.at[:,:,0:1].get() + c.at[1:2,5:9].get()*qb.at[:,:,1:2].get()  )/((1-porosity)*12*e*e) )
    return leq

@jit
def compute_qb_grass(u,A,m):
  
    qb=jnp.zeros(u.shape)
    uxy = (u[:,:,0:1]**2 + u[:,:,1:2]**2)**(0.5 * m - 0.5)

    qb = qb.at[:,:,0:1].set(A * u[:,:,0:1] * uxy)
    qb = qb.at[:,:,1:2].set(A * u[:,:,1:2] * uxy)
    return qb


@jit
def compute_qb_MPM(u,h,nx,ny,g):


    kd=0.02
    ke=0.3
    dp1=1.3e-3
    dp2=0.7e-3
    np1=0.0157
    np2=0.0141
    Fbp1=0.5
    Fbp2=0.5
    s=2.65
    thtac=0.047

    dltathtap1=jnp.zeros((nx,ny,1))
    dltathtap2=jnp.zeros((nx,ny,1))
    thtap1=jnp.zeros((nx,ny,1))
    thtap2=jnp.zeros((nx,ny,1))
    Gp1=jnp.zeros((nx,ny,1))
    Gp2=jnp.zeros((nx,ny,1))
    qb=jnp.zeros((nx,ny,2))
    uxy: Array = jnp.zeros((nx, ny, 1))
 
    thtap1= thtap1.at[:,:,0:1].set(  (np1**2)*(u.at[:,:,0:1].get()**2 + u.at[:,:,1:2].get()**2)/((s-1)*dp1*(h.at[:,:,0:1].get()**(1/3))) )
    thtap2= thtap2.at[:,:,0:1].set(  (np2**2)*(u.at[:,:,0:1].get()**2 + u.at[:,:,1:2].get()**2)/((s-1)*dp2*(h.at[:,:,0:1].get()**(1/3))) )
    dltathtap1 = dltathtap1.at[:,:,0:1].set( jnp.where(thtap1[:,:,0:1] > thtac, thtap1[:,:, 0:1] - thtac, 0) )
    dltathtap2 = dltathtap2.at[:,:,0:1].set( jnp.where(thtap2[:,:, 0:1] > thtac, thtap2[:,:, 0:1] - thtac, 0) )

    Gp1=Gp1.at[:,:,0:1].set(  (np1**3)*(g**0.5)*8*(dltathtap1.at[:,:,0:1].get()**0.5)*s*kd*np1/(  (s-1)*(h.at[:,:,0:1].get()**0.5)*(thtap1.at[:,:,0:1].get()**(3/2))*ke*dp1  ) )
    Gp2=Gp2.at[:,:,0:1].set(  (np2**3)*(g**0.5)*8*(dltathtap2.at[:,:,0:1].get()**0.5)*s*kd*np2/(  (s-1)*(h.at[:,:,0:1].get()**0.5)*(thtap2.at[:,:,0:1].get()**(3/2))*ke*dp2  ))
    Gp1 = jnp.nan_to_num(Gp1, nan=0)
    Gp2 = jnp.nan_to_num(Gp2, nan=0)
    uxy=uxy.at[:,:,:].set((u.at[:,:,0:1].get()**2+u.at[:,:,1:2].get()**2) )
    qb=qb.at[:,:,0:1].set( (Fbp1*Gp1.at[:,:,0:1].get()+Fbp2*Gp2.at[:,:,0:1].get())*uxy.at[:,:,0:1].get()*u.at[:,:,0:1].get())
    qb=qb.at[:,:,1:2].set( (Fbp1*Gp1.at[:,:,0:1].get()+Fbp2*Gp2.at[:,:,0:1].get())*uxy.at[:,:,0:1].get()*u.at[:,:,1:2].get())
    qb = qb.at[:, :, 0].set(jnp.where(u[:, :, 0] == 0, 0, qb[:, :, 0]))
    qb = qb.at[:, :, 1].set(jnp.where(u[:, :, 1] == 0, 0, qb[:, :, 1]))
    return qb

@jit
def compute_qb_MMp_0(n,h,glac,u,rhos,rhow,d,ThreshholdShields,fai,qb,rho):
    
    rho_muddywater=rhow*(1-rho)+rhos*rho
    Cf=(n*n*glac)/(h**(1/3))
    
    ux=u.at[:,:,0:1].get()
    uy=u.at[:,:,1:2].get()
    
    taubx=rho_muddywater*Cf*ux*((ux*ux+uy*uy)**0.5)
    tauby=rho_muddywater*Cf*uy*((ux*ux+uy*uy)**0.5)
    
    s=rhos/rhow-1
    Shieldsx=( taubx  )/( rho_muddywater*s*glac*d )
    Shieldsy=( tauby  )/( rho_muddywater*s*glac*d )
    
    deltax=jnp.maximum(Shieldsx-ThreshholdShields,0)
    deltay=jnp.maximum(Shieldsy-ThreshholdShields,0)
    qb=qb.at[:,:,0:1].set( fai*8*( (s*glac*(d**3))**0.5 ) * ( (deltax)**1.5 ) )
    qb=qb.at[:,:,1:2].set( fai*8*( (s*glac*(d**3))**0.5 ) * ( (deltay)**1.5 ) )
    
    return qb



