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
Array: TypeAlias = jax.Array
from Type import Lattice, DVS, Array


@jit
def ComputeSinking(d,g,RhoWater,RhoSton,NuWater):
    s=RhoSton/RhoWater-1
    sinking=((13.95*NuWater/d)**2+1.09*s*g*d)**0.5 - 13.95*NuWater/d
    return sinking

@jit 
def compute_alpha(p,c):
    c = jnp.where(c == 0, 1e-6, c)
    return jnp.minimum(2, (1 - p) / c)
    

@jit
def compute_Cstar_MMP(qb,h,u):
    ux=u.at[:,:,0:1].get()
    uy=u.at[:,:,1:2].get()
    return qb/(h*((ux**2+uy**2)**0.5))




@jit
def compute_sheels(nb,u,s,d,h):
    ux=u.at[:,:,0:1].get()
    uy=u.at[:,:,1:2].get()
    thta=nb*nb*(ux*ux+uy*uy)/( s*d*(h**(1/3)))
    return thta



@jit
def compute_MPM(fai,thta,thtac,rhos,rhow,d,gravity):
    s=rhos/rhow-1
    qbDirectionless=8*fai*(s*gravity*d*d*d)*((thta-thtac)**1.5)
    qbDirectionless=jnp.where(thta>thtac,qbDirectionless,0)
    return qbDirectionless

@jit
def compute_qb_MPM(u,qbDirectionless):
    
    qb=jnp.zeros(u.shape)
    ux=u.at[:,:,0:1].get()
    uy=u.at[:,:,1:2].get()
    uxy=(ux*ux+uy*uy)**0.5
    qb=qb.at[:,:,0:1].set(qbDirectionless* ux/uxy)
    qb=qb.at[:,:,1:2].set(qbDirectionless* uy/uxy)
    qb=jnp.where(uxy==0,0,qb)
    return qb


@jit
def compute_Cstar_MPM(qbDirectionless,h,u):
    ux=u.at[:,:,0:1].get()
    uy=u.at[:,:,1:2].get()
    return qbDirectionless/( h*((ux*ux+uy*uy)**0.5) )

@jit
def compute_Cstar_ZJR_1(u,gravity,h,SinkingSpeed,m1,k):

    
    ux=u.at[:,:,0:1].get()
    uy=u.at[:,:,1:2].get()
    uxy=(ux**2+uy**2)**0.5
    Cstar=k*(( (uxy**3)/(gravity*h*SinkingSpeed) )**m1)
    return Cstar

@jit
def compute_Cstar_ZJR(u,gravity,h,Sinkingspeed,rhos):
    ux=u.at[:,:,0:1].get()
    uy=u.at[:,:,1:2].get()
    uxy=(ux**2+uy**2)**0.5
    Cstar=(uxy**3/(gravity*h*Sinkingspeed))**1.5/( 20*rhos*( 1+(uxy**3/(45*gravity*h*Sinkingspeed))**1.15 ) )
    return Cstar

@jit
def MPM1(nb,gravity,u,d,fai,thtac,s,h):
    qb=jnp.zeros(u.shape)
    ux=u.at[:,:,0:1].get()
    uy=u.at[:,:,1:2].get()
    uxy=(ux*ux+uy*uy)**0.5
    thtax=nb*nb*ux*uxy/( s*d*(h**(1/3))  )
    thtay=nb*nb*uy*uxy/( s*d*(h**(1/3))  )
    dltathtax = jnp.where(thtax > thtac, thtax- thtac, 0) 
    dltathtay = jnp.where(thtay > thtac, thtay- thtac, 0) 
    qb=qb.at[:,:,0:1].set(fai*8*( (s*gravity*d*d*d)**0.5 )*(( dltathtax )**1.5 ))
    qb=qb.at[:,:,1:2].set(fai*8*( (s*gravity*d*d*d)**0.5 )*(( dltathtay )**1.5 ))
    return qb

@jit
def compute_qb_grass(u,A,m):
    qb=jnp.zeros(u.shape)

    uxy = (u.at[:,:,0:1].get()**2 + u.at[:,:,1:2].get()**2)**(0.5 * m - 0.5)

    qb = qb.at[:,:,0:1].set(A * u.at[:,:,0:1].get() * uxy)
    qb = qb.at[:,:,1:2].set(A * u.at[:,:,1:2].get() * uxy)
    return qb