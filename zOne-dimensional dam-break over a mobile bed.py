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
import cv2  
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
import drysteptreatment
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



from jax import jit, vmap
@jit
def stream(f, c, e, solid):
    nx, ny, q = f.shape
    #
    f_transposed = jnp.transpose(f, (2, 0, 1))  
    c_transposed = jnp.transpose(c)             
    
    def process_single_k(f_k, c_k):
        
        ex = (c_k[0] // e).astype(int)
        ey = (c_k[1] // e).astype(int)
        
        
        rolled_f = jnp.roll(f_k, shift=(ex, ey), axis=(0, 1))
        rolled_solid = jnp.roll(solid, shift=(ex, ey), axis=(0, 1))
      
        return rolled_f * rolled_solid * solid

    
    f_new_transposed = vmap(process_single_k)(f_transposed, c_transposed)
  
    return jnp.transpose(f_new_transposed, (1, 2, 0))



@jit
def OpenBoundary(g:Array):
 

    
  
    g=g.at[0:2,:,:].set(g.at[2:3,:,:].get())

    
    g=g.at[-2:,:].set(g.at[-3:-2,:,:].get())

    
    g=g.at[:,-2:,:].set(g.at[:,-3:-2,:].get())
 

    g=g.at[:,0:2,:].set(g.at[:,2:3,:].get())
    return g



@jit
def zerpgrand(f):

    f=f.at[0:2,:,:].set(f.at[2:3,:,:].get())
    f=f.at[-2:,:,:].set(f.at[-3:-2,:,:].get())
 
    f=f.at[:,0:2,:].set(f.at[:,2:3,:].get())
    f=f.at[:,-2:,:].set(f.at[:,-3:-2,:].get())
    
    return f


#_______________________________________________________________________________________________
casename = "zOne-dimensional dam-break over a mobile bed"
h_output_dir = "./output_bed_load/h_" + casename
h_output_dir_stable = "./output_bed_load_stable/h_" + casename
if not os.path.exists(h_output_dir):
    os.makedirs(h_output_dir)
if not os.path.exists(h_output_dir_stable):
    os.makedirs(h_output_dir_stable)

lattice: DVS = Model.DefineDVS("D2Q16")

gravity=9.81
h0=0.1
Kv=(2**0.5)/((gravity*h0)**0.5)
Kt=(gravity*h0)**0.5/((2**0.5)*h0)
KF=2/gravity
Kh=1/h0
print(f"Kv={Kv},Kt={Kt},KF={KF},Kh={Kh},h0={h0}")

L=2.5
B=0.02
Q=10*B
Hin=1
hmin=0.003
Hout=0.005
Vin=0
t=1
dt=0.0001
dx=0.005
dy=0.005
tau = 0.001
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
print(f"Vin_={Vin_},Hin_={Hin_},dt_={dt_},dx_={dx_},dy_={dy_},tau={tau},nu={nu}")
x = jnp.arange(0, round(L + dx,4), dx).reshape((nx, 1))
y = jnp.arange(0,round(B+dy,4),dy).reshape((1, ny))
x_ = x/h0
y_ = y/h0
zb=jnp.ones((nx,ny,1))*0.1
zb_=zb*Kh
h= jnp.ones((nx, ny, 1))*0.1
h=h.at[round(1.25/dx):,:,:].set(hmin)
h_=h*Kh 
u = jnp.ones((nx, ny, 2))*0
u_=u*Kv
feq = jnp.zeros((nx, ny, 16))
f = jnp.zeros((nx, ny, 16))
feq = Model.BGKEquilibriumSWE4th(h_, u_, lattice["XI"], lattice["W"])
f = feq.copy()
fGrad0: Array = jnp.zeros(feq.shape)
porosity=0.3
nb=0.025
d=3.5e-3
rhos=1540
rhow=1000
NuWater=1e-6
fai=0.7
thtac=0.01
s=rhos/rhow-1
lattice_D2Q9:DVS=AD2Q9Scheme.DefineLattice("D2Q9",e)
qb=jnp.zeros((nx,ny,2))
leq=jnp.zeros((nx,ny,9))
l=jnp.zeros((nx,ny,9))
taul=1
Cs2=(e*e)/3
taug=1;  
rho=jnp.ones((nx,ny,1))*0
g=jnp.zeros((nx,ny,9))
geq=jnp.zeros((nx,ny,9))
geq=Asediment.Compute_geq_normal(lattice_D2Q9["W"],rho,lattice_D2Q9["XI"],u,e,h)
g=geq.copy()
DD=Cs2*dt*(taug-0.5)
period=1/dt
maxIter=int(t*period)
start = time.time()
qb=Acloser_function.MPM1(nb,gravity,u,d,fai,thtac,s,h)
leq=Abedload.compute_leq(zb,porosity,lattice_D2Q9["C"],e,qb)
l=leq.copy()
start = timer()
skip=2
mid=int(0.5*(ny-1))
hmin_=hmin*Kh
zbOld=zb
zbNew=zb
Weight=lattice_D2Q9["W"]

@jit
def computeqb(nb,s,d,h,u,gravity,fai,thtac):
    ux=u.at[:,:,0:1].get()
    uy=u.at[:,:,1:2].get()
    thta=nb*nb*(ux**2+uy**2)/(s*d*(h**(1/3)))
    dltathta = jnp.where(thta > thtac, thta- thtac, 0) 
    qb=fai*8*((s*gravity*d*d*d)**0.5)*((dltathta)**1.5)
    return qb
SinkingSpeed=Acloser_function.ComputeSinking(d,gravity,rhow,rhos,NuWater)
print(f"SinkingSpeed={SinkingSpeed}")
kse=1e-6

@jit
def compute_sheels(nb,u,s,d,h):
    ux=u.at[:,:,0:1].get()
    uy=u.at[:,:,1:2].get()
    thta=nb*nb*(ux*ux+uy*uy)/( s*d*(h**(1/3)))
    return thta

import drysteptreatment

for times in range (maxIter+1):
    
    f=zerpgrand(f)
    h_,u_= Model.CalcMacroVars(f, lattice["XIT"])
    h_,u_=deal_dry(h_,u_,hmin_)
    h_,rho=deal_dry(h_,rho,hmin_)
    fGrad = CalcGrad(fGrad0, f)
    
    
    force_ = drysteptreatment.calculate_source_NND_2d(lattice["W"], lattice["XI"], zb*Kh, h_, dx_, dy_, hmin_) + ASourceForce.CalForce_Friction(u_,h_,h0,nb,gravity,lattice["XI"],lattice["W"])
    force_1 = ASourceForce.CalForce_BedTime(zbNew/h0,zbOld/h0,lattice["W"],dt_)
    zbOld=zb
    f = Scheme.Euler1ST(f,fGrad,Model.BGKSWECollision(f, Model.BGKEquilibriumSWE4th(h_, u_, lattice["XI"], lattice["W"]), tau, h_),dt_,(force_),0)
    
    h=h_.at[:,:,:].get()/Kh
    u=u_.at[:,:,:].get()/Kv
    
    thta=compute_sheels(nb,u,s,d,h)
    qbhe=computeqb(nb,s,d,h,u,gravity,fai,thtac)
    ux=u.at[:,:,0:1].get()
    uy=u.at[:,:,1:2].get()
    U=(ux**2+uy**2)**0.5
    
    
    Cstar=( qbhe ) /( h*U )
    alpha=jnp.minimum(2,(1-porosity)/(rho))
    E=alpha*SinkingSpeed*Cstar
    E=jnp.where(thta>=thtac,E,0)
    D=alpha*SinkingSpeed*rho

    qb=qb.at[:,:,0:1].set( (qbhe* ux)/ ( U + kse)  )
    qb=qb.at[:,:,1:2].set( (qbhe* uy)/ ( U + kse)  )
    
    g=OpenBoundary(g)
    rho = Asediment.CalcMacroVars_sedment(g,h)
    geq=Asediment.Compute_geq_normal(lattice_D2Q9["W"],rho,lattice_D2Q9["XI"],u,e,h)
    force_sediment = Weight*(E-D)
    g = AD2Q9Scheme.collision(g,taug,geq,dt,force_sediment)
    g = AD2Q9Scheme.StreamMap(g, lattice_D2Q9["C"],e)
 
    l=OpenBoundary(l)
    zb=Abedload.CalcMacroVars_bed(l)
    leq=Abedload.compute_leq(zb,porosity,lattice_D2Q9["C"],e,qb)
    force_bed = -1*Weight*(E-D)/(1-porosity)
    l = AD2Q9Scheme.collision(l,taul,leq,dt,force_bed)
    l = AD2Q9Scheme.StreamMap(l, lattice_D2Q9["C"],e)

    zbNew=zb
    if jnp.any(rho<0).item():
        print(times)
        sys.exit(1)  
    zbzb=zb-0.1
    hzb=(zb+h)-0.1
    if (times % int(0.25/dt)==0):  
        show_txt_1D(times, dt, x, hzb, casename+"2zbh",mid)
        show_txt_1D(times, dt, x, zbzb, casename+"2zb",mid)
        
    
    if (times % int(0.05/dt)==0):
        
        fff= h[:,mid,:]
        water = np.sum((fff[:-1] + fff[1:]) / 2) * dx
       
        seidment1=rho[:,mid,:]*h[:,mid,:]
        sedi = np.sum((seidment1[:-1] + seidment1[1:]) / 2) * dx
        

        bed=zb[:,mid,:]
        bed1 = np.sum((bed[:-1] + bed[1:]) / 2) * dx
        bed1 = bed1
        lossbed=(0.25-bed1)*(1-porosity)
        
        print(f'{times*dt:.2f} Sum of water volume：{water} bed reduction：{lossbed} suspended increase：{sedi} sum:{(sedi-lossbed)}')

    
end =timer()





    