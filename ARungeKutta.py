
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
import A2stNND
import AConvection5stWENO

import A2stNND
@jit
def CalcGrad(fGrad0, f,dx_,dy_):
    return fGrad0.at[2:-2, 2:-2, :].set(
        A2stNND.st2_NND(f,lattice["XI"],dx_,dy_)
    )

m11=0.24169426078821
m22=0.24169426078821
m21=-0.24169426078821
m32=0.75830573921179
m33=0.24169426078821
m41=0.06042356519705
m42=0.12915286960590
m43=0.13745860881769
m44=0.24169426078821


@jit
def RK_IE(f,Kn,dt,fGrad0,force,dx_,dy_,h_):
    h0,u0= Model.CalcMacroVars(f, lattice["XIT"])
    tau1=Kn/h0
    
    feq1=Model.BGKEquilibriumSWE4th(h0, u0, lattice["XI"], lattice["W"])
    f1=(f+dt*m11*feq1/tau1 +dt*force)/(1+dt*m11/tau1)
    
    
    h1,u1= Model.CalcMacroVars(f1, lattice["XIT"])
    tau2=Kn/h1
    feq2=Model.BGKEquilibriumSWE4th(h1, u1, lattice["XI"], lattice["W"])
    f2=(f+dt*m21*(feq1-f1)/tau1 + dt*m22*feq2/tau2 + dt*force)/(1+dt*m22/tau2)

    h2,u2= Model.CalcMacroVars(f2, lattice["XIT"])
    tau3=Kn/h2
    feq3=Model.BGKEquilibriumSWE4th(h2, u2, lattice["XI"], lattice["W"])
    f3=(f-dt*CalcGrad(fGrad0, f2,dx_,dy_) + dt*m32*(feq2-f2)/tau2 +dt*m33*feq3/tau3 + dt*force)/(1+dt*m33/tau3)

    h3,u3= Model.CalcMacroVars(f3, lattice["XIT"])
    tau4=Kn/h3
    feq4=Model.BGKEquilibriumSWE4th(h3, u3, lattice["XI"], lattice["W"])
    f4=(f-dt*( 0.25*CalcGrad(fGrad0, f2,dx_,dy_)+0.25*CalcGrad(fGrad0, f3,dx_,dy_) )+dt*m41*(feq1-f1)/tau1 + dt*m42*(feq2-f2)/tau2 + dt*m43*(feq3-f3)/tau3  + dt*m44*feq4/tau4 + dt*force)/(1+dt*m44/tau4)
    
    fnext=f-dt*( (1/6)*CalcGrad(fGrad0, f2,dx_,dy_)+ (1/6)*CalcGrad(fGrad0, f3,dx_,dy_)+ (2/3)*CalcGrad(fGrad0, f4,dx_,dy_) ) +dt*((1/6)*( feq2-f2 )/tau2+(1/6)*( feq3-f3 )/tau3+(2/3)*( feq4-f4)/tau4) +dt*force
    
    return fnext