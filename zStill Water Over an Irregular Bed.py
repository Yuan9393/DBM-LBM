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
from AComputeSlope import bed_slope_MidPoint,slope_FourPoint,compute_slope
from utils import show_tecplot_2d_2,show_tecplot_2d_4,show_tecplot_2d_1,show_txt_1D,outputCsv
import Abedload
import AD2Q9Scheme
import ASourceForce
from utils_mesh import get_zb,draw_img,get_edge_index_of_mesh,get_index_of_mesh,generate_house_S12,get_zb_by_file
import Asediment
import Acloser_function
#___________________________________________________________________________________________



import A2stNND
@jit
def CalcGrad(fGrad0, f):
    return fGrad0.at[2:-2, 2:-2, :].set(
        A2stNND.st2_NND(f,lattice["XI"],dx_,dy_)
    )


@jit
def deal_dry(h,u,hmin_): 
    
    flag=h<=hmin_
    h=h.at[:,:,:].set(flag*hmin_+~flag*h)
    u=u.at[:,:,:].set(~flag*u)
    return h,u



@jit
def OpenBoundary(g:Array):
 


   
    g=g.at[0:1,:,:].set(g.at[1:2,:,:].get())

    

    g=g.at[-1:,:].set(g.at[-2:-1,:,:].get())

    

    g=g.at[:,-1:,:].set(g.at[:,-2:-1,:].get())

    
    g=g.at[:,0:1,:].set(g.at[:,1:2,:].get())
    return g



@jit
def upwind1stBoundraryZeroGrand(f):


    f=f.at[1,:,:].set(2*f.at[2,:,:].get()-f.at[3,:,:].get())
    f=f.at[0,:,:].set(2*f.at[1,:,:].get()-f.at[2,:,:].get())
    f=f.at[-2,:,:].set(2*f.at[-3,:,:].get()-f.at[-4,:,:].get())
    f=f.at[-1,:,:].set(2*f.at[-2,:,:].get()-f.at[-3,:,:].get())


    
    f=f.at[:,-2,:].set(2*f.at[:,-3,:].get()-f.at[:,-4,:].get())
    f=f.at[:,-1,:].set(2*f.at[:,-2,:].get()-f.at[:,-3,:].get())

    
    f=f.at[:,1,:].set(2*f.at[:,2,:].get()-f.at[:,3,:].get())
    f=f.at[:,0,:].set(2*f.at[:,1,:].get()-f.at[:,2,:].get())
    return f


@jit
def zerpgrand(f):
  
    f=f.at[0:2,:,:].set(f.at[2:3,:,:].get())
    f=f.at[-2:,:,:].set(f.at[-3:-2,:,:].get())

    f=f.at[:,0:2,:].set(f.at[:,2:3,:].get())
    f=f.at[:,-2:,:].set(f.at[:,-3:-2,:].get())

    
    return f


#_______________________________________________________________________________________________
casename = "zstill water over an irregular bed"
h_output_dir = "./output_bed_load/h_" + casename
h_output_dir_stable = "./output_bed_load_stable/h_" + casename
if not os.path.exists(h_output_dir):
    os.makedirs(h_output_dir)
if not os.path.exists(h_output_dir_stable):
    os.makedirs(h_output_dir_stable)

lattice: DVS = Model.DefineDVS("D2Q16")
#无量纲系数
gravity=9.81
h0=10
Kv=(2**0.5)/((gravity*h0)**0.5)
Kt=(gravity*h0)**0.5/((2**0.5)*h0)
KF=2/gravity
Kh=1/h0
print(f"Kv={Kv},Kt={Kt},KF={KF},Kh={Kh},h0={h0}")

L=1500
B=20
Q=10*B
Hin=1

hmin=0.01
Hout=0.005
Vin=0
t=3240000
dt=0.01
dx=5
dy=5
tau = 0.01

nx=round(L/dx)+1
ny=round(B/dy)+1
e=dx/dt#ex=ey=e
print(f"nx={nx},ny={ny}")
print(f"Vin={Vin},Hin={Hin},dt={dt},dx={dx},dy={dy},B={B},L={L}")

L_ = Kh*L
B_ = Kh*B
Q_=Q * (2 / (gravity * h0**5)) ** 0.5
Hin_ = Kh*Hin
Hout_ = Kh*Hout
Vin_=Kv*Vin
t_=Kt*t
dt_ = Kt*dt


nu=(tau*(gravity*h0*h0*h0)**0.5)/(2**0.5)
dx_=dx*Kh
dy_=dy*Kh
nu=tau*((gravity*h0*h0*h0)**0.5)/(2**0.5)
print(f"模拟量入口速度:Vin_={Vin_},Hin_={Hin_},dt_={dt_},dx_={dx_},dy_={dy_},tau={tau},nu={nu}")
#+++++++++++<<<初始化flow>>>+++++++++++++++++++
x = jnp.arange(0, round(L + dx,4), dx).reshape((nx, 1))
y = jnp.arange(0,round(B+dy,4),dy).reshape((1, ny))

hOutlet = Hout_  *  jnp.ones((1, ny, 1))

#---------------------------------------------------------------------------------------------------
zb=jnp.ones((nx,ny,1))*0
x_profile = jnp.array([
    0, 50, 100, 150, 250, 300, 350, 400, 425, 435,
    450, 475, 500, 505, 530, 550, 565, 575, 600, 650,
    700, 750, 800, 820, 900, 950, 1000, 1500
])

zb_profile = jnp.array([
    0, 0, 2.5, 5, 5, 3, 5, 5, 7.5, 8,
    9, 9, 9.1, 9, 9, 6, 5.5, 5.5, 5, 4,
    3, 3, 2.3, 2, 1.2, 0.4, 0, 0
])


x_grid = jnp.arange(nx) * dx
zb_1d = jnp.interp(x_grid, x_profile, zb_profile)

zb = zb.at[:, :, 0].set(zb_1d[:, None])
zb_=zb*Kh

h= jnp.ones((nx, ny, 1))*10

h=h-zb
h_=h*Kh 

u = jnp.ones((nx, ny, 2))*0
u_=u*Kv

slope_ = jnp.zeros((nx, ny, 2))

feq = jnp.zeros((nx, ny, 16))
f = jnp.zeros((nx, ny, 16))
feq = Model.BGKEquilibriumSWE4th(h_, u_, lattice["XI"], lattice["W"])
f = feq.copy()
fGrad0: Array = jnp.zeros(feq.shape)



period=1/dt
maxIter=int(t*period)

start = time.time()

start = timer()
skip=2
mid=int(0.5*(ny-1))
hmin_=hmin*Kh
zbOld=zb
zbNew=zb

import drysteptreatment
for times in range (maxIter+1):
    
    f=zerpgrand(f)
    h_,u_= Model.CalcMacroVars(f, lattice["XIT"])
    h_,u_=deal_dry(h_,u_,hmin_)
    h_=jnp.where(h_<hmin_,hmin_,h_)
    u_=jnp.where(h<hmin_,0,u_)
    h=h_.at[:,:,:].get()/Kh
    u=u_.at[:,:,:].get()/Kv
    fGrad = CalcGrad(fGrad0, f)
    force_ = drysteptreatment.calculate_source_NND_2d(lattice["W"], lattice["XI"], zb*Kh, h_, dx_, dy_, hmin_)
    f = Scheme.Euler1ST(f,fGrad,Model.BGKSWECollision(f, Model.BGKEquilibriumSWE4th(h_, u_, lattice["XI"], lattice["W"]), tau, h_),dt_,force_,0)
    
    if jnp.isnan(h).any() or jnp.isnan(u).any():
        print(times)

        sys.exit(1)  

    if (times % int(32400/dt)==0):  
        Fr=u.at[:,:,0:1].get()/( np.sqrt(gravity*h))
        show_txt_1D(times, dt, x, zb, casename+"zb",mid)
        show_txt_1D(times, dt, x, zb+h, casename+"zbh",mid)
        show_txt_1D(times, dt, x, u.at[:,:,0:1].get(), casename+"u",mid)
        show_txt_1D(times, dt, x, Fr, casename+"Fr",mid)
      
    if (times % int(1000/dt)==0):
        print(f'{times*dt:.2f}---seconds')


end =timer()





    