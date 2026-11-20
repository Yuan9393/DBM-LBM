import os
import h5py as h5
import jax
import jax.numpy as jnp
from jax import jit
import math
import Model
import Scheme
import Utility
from Type import Array, DVS
from rich import pretty
from timeit import default_timer as timer
import sys
from Type import Lattice, DVS, Array
sqrt3 = math.sqrt(3)
import numpy as np
# import cv2
import matplotlib.pyplot as plt
import datetime
import sys
import time
import cv2  # 用于视频录制
import pandas as pd
pretty.install()
Utility.SetPrecision("64")
Utility.SetPlatform("gpu")
print("JAX local devices: ", jax.local_devices())
print("Jax device", jax.devices())
# from AComputeSlope import bed_slope_MidPoint,slope_FourPoint,compute_slope
from utils import show_tecplot_2d_2,show_tecplot_2d_4,show_tecplot_2d_1,show_txt_1D,outputCsv
import Abedload
import AD2Q9Scheme
import ASourceForce
import Asediment
import Acloser_function
from utils_mesh import get_zb,draw_img,get_edge_index_of_mesh,get_index_of_mesh,generate_house_S12,get_zb_by_file
from AComputeSlope import bed_slope_MidPoint
#___________________________________________________________________________________________





import A2stNND

@jit
def CalcGradnnd(fGrad0, f):
    return fGrad0.at[2:-2, 2:-2, :].set(
        A2stNND.st2_NND(f,lattice["XI"],dx_,dy_)
    )


@jit
def zerograndBoundary(f): 
    
    
    f= f.at[0:2,:,:].set(f.at[2:3,:,:].get())
    
    f= f.at[-2:,:,:].set(f.at[-3:-2,:,:].get())
    
    f=f.at[:,0:2,:].set(f.at[:,2:3,:].get())
    f=f.at[:,-2:,:].set(f.at[:,-3:-2,:].get())


    return f


#_______________________________________________________________________________________________
casename = "lake at rest"
h_output_dir = "./output_bed_load/h_" + casename
h_output_dir_stable = "./output_bed_load_stable/h_" + casename
if not os.path.exists(h_output_dir):
    os.makedirs(h_output_dir)
if not os.path.exists(h_output_dir_stable):
    os.makedirs(h_output_dir_stable)

lattice: DVS = Model.DefineDVS("D2Q16")

gravity=1
h0=1
Kv=(2**0.5)/((gravity*h0)**0.5)
Kt=(gravity*h0)**0.5/((2**0.5)*h0)
KF=2/gravity
Kh=1/h0
print(f"Kv={Kv},Kt={Kt},KF={KF},Kh={Kh},h0={h0}")
#宏观量

L=6
B=1
Q=10*B
Hin=0
hmin=0.001
Hout=0.005
Vin=0
t=2
dt=0.0002
dx=0.005
dy=dx
tau = 0.005

nx=round(L/dx)+1
ny=round(B/dy)+1
e=dx/dt
Cs2=(e*e)/3

L_ = Kh*L
B_ = Kh*B
Q_=Q * (2 / (gravity * h0**5)) ** 0.5
Hin_ = Kh*Hin
Hout_ = Kh*Hout
Vin_=Kv*Vin
t_=Kt*t
dt_ = Kt*dt
dx_=dx*Kh
dy_=dy*Kh
hmin_=hmin*Kh
nu=tau*((gravity*h0*h0*h0)**0.5)/(2**0.5)

x = jnp.linspace(0.0, L, nx).reshape((nx, 1))
y = jnp.linspace(0.0, B, ny).reshape((1, ny))
zb= jnp.ones((nx, ny, 1))*0
xx = jnp.arange(nx) * dx
yy = jnp.arange(ny) * dy
X, Y = jnp.meshgrid(xx, yy, indexing='ij')
zb = 0.8 * jnp.exp(-5*(X - 2.9)**2 - 50*(Y - 0.5)**2)
zb = zb[..., None]  

h= jnp.ones((nx, ny, 1))*1
h=h.at[round(2.05/dx):round(2.15/dx),:,:].set(1.01)

h=h-zb
h_=h*Kh

u: Array = jnp.zeros((nx, ny, 2))
u_=u*Kv

feq = jnp.zeros((nx, ny, 16))
f = jnp.zeros((nx, ny, 16))
feq = Model.BGKEquilibriumSWE4th(h_, u_, lattice["XI"], lattice["W"])
f = feq.copy()
fGrad0: Array = jnp.zeros(feq.shape)


nb=jnp.ones((nx,ny,1))*0

#----------------------------------------------------
period=1/dt
maxIter=int(t*period)
start = timer()
skip=2
mid=int(0.5*(ny-1))


import drysteptreatment
x_ = x/h0
y_ = y/h0
X_mesh, Y_mesh = jnp.meshgrid(x_.flatten(), y_.flatten(), indexing='ij')

for times in range (maxIter+1):
            
    f=zerograndBoundary(f)

    h_,u_= Model.CalcMacroVars(f, lattice["XIT"])
 
    fGrad = CalcGradnnd(fGrad0, f)
    force_1= drysteptreatment.calculate_source_NND_2d(lattice["W"], lattice["XI"], zb*Kh, h_, dx_, dy_, hmin_)
    force_2 = ASourceForce.CalForce_Friction(u_,h_,h0,nb,gravity,lattice["XI"],lattice["W"])
 
    force_=force_1+force_2
    f = Scheme.Euler1ST(f,fGrad,Model.BGKSWECollision(f, Model.BGKEquilibriumSWE4th(h_, u_, lattice["XI"], lattice["W"]), tau, h_),dt_,force_,0)
    
    if jnp.isnan(h).any() or jnp.isnan(u).any():
        print(times)
        sys.exit(1)
    
    h=h_.at[:,:,:].get()/Kh
    u=u_.at[:,:,:].get()/Kv
    
    if (times % int(0.1/dt)==0):  
    
        ux=u.at[:,:,0:1].get()
        uy=u.at[:,:,1:2].get()
        U=(ux**2+uy**2)**0.5
        Fr=U/((gravity*h)**0.5)
        
        show_tecplot_2d_4(casename,times,dt,(x-2),(y),nx,ny,(h+zb),u,zb,Fr)
        
    if (times % int(0.05/dt)==0):
        print(f'{times*dt:.2f}---seconds')
    
end =timer()
