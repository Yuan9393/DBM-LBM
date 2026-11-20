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
def AutoBoundary(f):
   
    f=f.at[masksolidnode].set(0)

    f= f.at[0:2,:,:].set(f.at[2:3,:,:].get())

    f= f.at[-2:,:,:].set(f.at[-3:-2,:,:].get())
    f=f.at[:,0:2,:].set(f.at[:,2:3,:].get())
    f=f.at[:,-2:,:].set(f.at[:,-3:-2,:].get())
  
    for i,idx_VG in enumerate(idx_VG_list):
        
        id_VG_type=id_VG_type_list[i]
        idx_x=idx_VG[:,0]
        idx_y=idx_VG[:,1]

        x_delta=idx_delta_VG_list[i][0]
        y_delta=idx_delta_VG_list[i][1]
        

        h_temp, u_temp = Model.CalcMacroVars(f[[idx_x+x_delta], [idx_y+y_delta], :], lattice["XIT"])
        u_= switch_u(id_VG_type,[set_bottom_top_u,set_left_right_u,set_corner_u],u_temp)
        f = f.at[[idx_x],[idx_y],:].set(
            Model.BGKEquilibriumSWE4th(h_temp,u_,lattice["XI"],lattice["W"],)
            +f.at[[idx_x+x_delta], [idx_y+y_delta], :].get()
            - Model.BGKEquilibriumSWE4th(h_temp, u_temp, lattice["XI"], lattice["W"])
        )
        f = f.at[[idx_x-x_delta],[idx_y-y_delta],:].set( f.at[[idx_x],[idx_y],:].get() )
    return f




@jit
def AutoBoundaryD2Q9_sediment(g,f,h,e): 
    
    
    g=g.at[masksolidnode].set(0)

    g= g.at[0:1,:,:].set(g.at[1:2,:,:].get())

    
    g= g.at[-1:,:,:].set(g.at[-2:-1,:,:].get())

    g=g.at[:,0:1,:].set(g.at[:,1:2,:].get())
    g=g.at[:,-1:,:].set(g.at[:,-2:-1,:].get())
    
    for i,idx_VG in enumerate(idx_VG_list):
        
        id_VG_type=id_VG_type_list[i]
        idx_x=idx_VG[:,0]
        idx_y=idx_VG[:,1]

        x_delta=idx_delta_VG_list[i][0]
        y_delta=idx_delta_VG_list[i][1]
        

        h_temp, u_temp = Model.CalcMacroVars(f[[idx_x+x_delta], [idx_y+y_delta], :], lattice["XIT"])
        rho_temp=Asediment.CalcMacroVars_sedment(g[[idx_x+x_delta], [idx_y+y_delta], :],h[[idx_x+x_delta], [idx_y+y_delta], :])
        u_= switch_u(id_VG_type,[set_bottom_top_u,set_left_right_u,set_corner_u],u_temp)

        g = g.at[[idx_x],[idx_y],:].set(
            Asediment.Compute_geq_normal(lattice_D2Q9["W"],rho_temp,lattice_D2Q9["XI"],u_,e,h_temp)
            +g.at[[idx_x+x_delta], [idx_y+y_delta], :].get()
            -Asediment.Compute_geq_normal(lattice_D2Q9["W"],rho_temp,lattice_D2Q9["XI"],u_temp,e,h_temp)
        )
    return g

@jit
def AutoBoundaryD2Q9_bed(g,f):

    g=g.at[masksolidnode].set(0)

    g= g.at[0:1,:,:].set(g.at[1:2,:,:].get())



    g= g.at[-1:,:,:].set(g.at[-2:-1,:,:].get())

    g=g.at[:,0:1,:].set(g.at[:,1:2,:].get())
    g=g.at[:,-1:,:].set(g.at[:,-2:-1,:].get())
   

    for i,idx_VG in enumerate(idx_VG_list):
        
        id_VG_type=id_VG_type_list[i]
        idx_x=idx_VG[:,0]
        idx_y=idx_VG[:,1]

        x_delta=idx_delta_VG_list[i][0]
        y_delta=idx_delta_VG_list[i][1]
        

        h_temp, u_temp = Model.CalcMacroVars(f[[idx_x+x_delta], [idx_y+y_delta], :], lattice["XIT"])
        zb_temp=Abedload.CalcMacroVars_bed(g[[idx_x+x_delta], [idx_y+y_delta], :])
        u_= switch_u(id_VG_type,[set_bottom_top_u,set_left_right_u,set_corner_u],u_temp)

        g = g.at[[idx_x],[idx_y],:].set(
            Abedload.compute_leq(zb_temp,porosity,lattice_D2Q9["C"],e,Acloser_function.MPM1(nb,gravity,u_,d,fai,thtac,s,h_temp))
            +g.at[[idx_x+x_delta], [idx_y+y_delta], :].get()
            -Abedload.compute_leq(zb_temp,porosity,lattice_D2Q9["C"],e,Acloser_function.MPM1(nb,gravity,u_temp,d,fai,thtac,s,h_temp))
        )
        
    return g




from jax import jit, vmap
@jit
def stream(f, c, e, solid):
   
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
def computeqb(nb,s,d,h,u,gravity,fai,thtac):
    ux=u.at[:,:,0:1].get()
    uy=u.at[:,:,1:2].get()
    thta=nb*nb*(ux**2+uy**2)/(s*d*(h**(1/3)))
    dltathta = jnp.where(thta > thtac, thta- thtac, 0) 
    qb=fai*8*((s*gravity*d*d*d)**0.5)*((dltathta)**1.5)
    return qb
#_______________________________________________________________________________________________
casename = "XWidening"
h_output_dir = "./output_bed_load/h_" + casename
h_output_dir_stable = "./output_bed_load_stable/h_" + casename
if not os.path.exists(h_output_dir):
    os.makedirs(h_output_dir)
if not os.path.exists(h_output_dir_stable):
    os.makedirs(h_output_dir_stable)

lattice: DVS = Model.DefineDVS("D2Q16")

gravity=9.8
h0=0.25
Kv=(2**0.5)/((gravity*h0)**0.5)
Kt=(gravity*h0)**0.5/((2**0.5)*h0)
KF=2/gravity
Kh=1/h0
print(f"Kv={Kv},Kt={Kt},KF={KF},Kh={Kh},h0={h0}")

L=6.1
B=0.6
Q=10*B
Hin=0
hmin=0.005
Vin=0
t=20
dt=0.0002
dx=0.01
dy=0.01
tau = 0.001
nx=round(L/dx)+1
ny=round(B/dy)+1
e=dx/dt
Cs2=(e*e)/3

L_ = Kh*L
B_ = Kh*B
Q_=Q * (2 / (gravity * h0**5)) ** 0.5
Hin_ = Kh*Hin
Vin_=Kv*Vin
t_=Kt*t
dt_ = Kt*dt
dx_=dx*Kh
dy_=dy*Kh
hmin_=hmin*Kh
nu=tau*((gravity*h0*h0*h0)**0.5)/(2**0.5)
x = jnp.arange(0, round(L + dx,4), dx).reshape((nx, 1))
y = jnp.arange(0,round(B+dy,4),dy).reshape((1, ny))

solidnode=jnp.ones((nx,ny))
solidnode=solidnode.at[:,0:round(0.05/dy+1)].set(0)
solidnode=solidnode.at[0:round(4.1/dx+1),round(0.3/dy):].set(0)
solidnode=solidnode.at[:,round(0.55/dy):].set(0)
solidnode=solidnode.at[0:round(0.1/dx+1),:].set(0)

masksolidnode=solidnode==0

#---------------------------------------------------------------------------------------------------------------

Nodetype=np.zeros((nx,ny))
Nodetype[solidnode.at[:,:].get()==0]=-1
Geometrytype=np.zeros((nx,ny))-1
from setup_domain import *

for i in range(1,nx-1):
    for j in range(1,ny-1):
        Geometrytype,Nodetype=SetEmbededBodyGeometry(Nodetype,i,j, Geometrytype)

idx_VG_list_org,idx_delta_VG_list_org=get_idx_of_VG(Geometrytype)

id_VG_type_list=[]
idx_VG_list=[]
idx_delta_VG_list=[]
for i,idx_VG in enumerate(idx_VG_list_org):
    if idx_VG.shape[0]>0:
    
        if i in [0,1]:
            id_VG_type_list.append(0)
        elif i in [2,3]:
            id_VG_type_list.append(1)
        else:
            id_VG_type_list.append(2)
        idx_VG_list.append(idx_VG_list_org[i])
        idx_delta_VG_list.append(idx_delta_VG_list_org[i])

def set_bottom_top_u(u_temp):
    return u_temp.at[:,:,1:].set(0)

def set_left_right_u(u_temp):
    return u_temp.at[:,:,0:1].set(0)
    

def set_corner_u(u_temp):
    return u_temp

def switch_u(index, branches, operand):
    return branches[index](operand)


idx8,idy8 = jnp.where(Geometrytype == 8)
idx9,idy9 = jnp.where(Geometrytype == 9)
idx10,idy10 = jnp.where(Geometrytype == 10)
idx11,idy11 = jnp.where(Geometrytype == 11)


idx4,idy4 = jnp.where(Geometrytype == 4)
idx5,idy5 = jnp.where(Geometrytype == 5)
idx6,idy6 = jnp.where(Geometrytype == 6)
idx7,idy7 = jnp.where(Geometrytype == 7)

idx0,idy0 = jnp.where(Geometrytype == 0)
idx1,idy1 = jnp.where(Geometrytype == 1)
idx2,idy2 = jnp.where(Geometrytype == 2)
idx3,idy3 = jnp.where(Geometrytype == 3)



zb=jnp.ones((nx,ny,1))*0.1
zb=zb.at[masksolidnode].set(0)

h= jnp.ones((nx, ny, 1))*hmin
h=h.at[round(0.1/dx+1):int(3.1/dx),int(0.05/dy+1):int(0.3/dy),:].set(0.25)

h_=h*Kh

u: Array = jnp.zeros((nx, ny, 2))
u_=u*Kv

h_,u_=deal_dry(h_,u_,hmin_)
h_=h_.at[masksolidnode].set(hmin_)
u_=u_.at[masksolidnode].set(0)


feq = jnp.zeros((nx, ny, 16))
f = jnp.zeros((nx, ny, 16))
feq = Model.BGKEquilibriumSWE4th(h_, u_, lattice["XI"], lattice["W"])
f = feq.copy()
fGrad0: Array = jnp.zeros(feq.shape)


lattice_D2Q9:DVS=AD2Q9Scheme.DefineLattice("D2Q9",e)
Weight=lattice_D2Q9["W"]
 
porosity=0.39
nb=0.023
d=1.72e-3
rhos=2630
rhow=1000
NuWater=1e-6
fai=0.17
thtac=0.035
s=rhos/rhow-1


taul=1
qb=jnp.zeros((nx,ny,2))
leq=Abedload.compute_leq(zb,porosity,lattice_D2Q9["C"],e,qb)
l=leq.copy()



taug=1.2;  
DD=Cs2*dt*(taug-0.5)
rho=jnp.ones((nx,ny,1))*0

g=jnp.zeros((nx,ny,9))
geq=jnp.zeros((nx,ny,9))

#------------------------------------------------------
period=1/dt
maxIter=int(t*period)
start = timer()
skip=2
mid=int(0.5*(ny-1))

zbOld=zb
zbNew=zb
SinkingSpeed=Acloser_function.ComputeSinking(d,gravity,rhow,rhos,NuWater)
kse=1e-6

import drysteptreatment
for times in range (maxIter+1):
            
    f=AutoBoundary(f)
    h_,u_= Model.CalcMacroVars(f, lattice["XIT"])
    h_,u_=deal_dry(h_,u_,hmin_)
    h_,rho=deal_dry(h_,rho,hmin_)
    h_=h_.at[masksolidnode].set(hmin_)
    u_=u_.at[masksolidnode].set(0)
    rho=rho.at[masksolidnode].set(0)

    h=h_.at[:,:,:].get()/Kh
    u=u_.at[:,:,:].get()/Kv

    fGrad = CalcGrad(fGrad0, f)
    force_ = drysteptreatment.calculate_source_NND_2d(lattice["W"], lattice["XI"], zb*Kh, h_, dx_, dy_, hmin_) + ASourceForce.CalForce_Friction(u_,h_,h0,nb,gravity,lattice["XI"],lattice["W"])
    force_1 = ASourceForce.CalForce_BedTime(zbNew/h0,zbOld/h0,lattice["W"],dt_)
    zbOld=zb
    f = Scheme.Euler1ST(f,fGrad,Model.BGKSWECollision(f, Model.BGKEquilibriumSWE4th(h_, u_, lattice["XI"], lattice["W"]), tau, h_),dt_,(force_+force_1),0)
    if jnp.isnan(h).any() or jnp.isnan(u).any():
        print(i)
        sys.exit(1)
       

    thta=Acloser_function.compute_sheels(nb,u,s,d,h)
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
          

    g = AutoBoundaryD2Q9_sediment(g,f,h,e)
    rho = Asediment.CalcMacroVars_sedment(g,h)
    geq=Asediment.Compute_geq_normal(lattice_D2Q9["W"],rho,lattice_D2Q9["XI"],u,e,h)
    force_sediment = Weight*(E-D)
    g = AD2Q9Scheme.collision(g,taug,geq,dt,force_sediment)
    g = stream(g,lattice_D2Q9["C"],e,solidnode)

    
    l=AutoBoundaryD2Q9_bed(l,f)
    zb=Abedload.CalcMacroVars_bed(l)
    leq=Abedload.compute_leq(zb,porosity,lattice_D2Q9["C"],e,qb)
    force_bed = -1*Weight*(E-D)/(1-porosity)
    l=AD2Q9Scheme.collision(l,taul,leq,dt,force_bed)
    l=stream(l, lattice_D2Q9["C"], e, solidnode)

    
    zb=jnp.where(zb<0,0,zb)
    zbNew=zb
    

    if (times % int(0.1/dt)==0):

        with open("outputp1Widening.txt", "a") as file:
            value = h[int(3.85/dx), int(0.175/dy), 0]+zb[int(3.85/dx), int(0.175/dy), 0]  # 取对应时间的值
            file.write(f"{times*dt:.2f}, {value}\n")

        with open("outputp2Widening.txt", "a") as file:
            value = h[int(4.3/dx), int(0.175/dy), 0]+zb[int(4.3/dx), int(0.175/dy), 0]  # 取对应时间的值
            file.write(f"{times*dt:.2f}, {value}\n")

        with open("outputp3Widening.txt", "a") as file:
            value = h[int(4.55/dx), int(0.175/dy), 0]+zb[int(4.55/dx), int(0.175/dy), 0]  # 取对应时间的值
            file.write(f"{times*dt:.2f}, {value}\n")

        with open("outputp4Widening.txt", "a") as file:
            value = h[int(5.05/dx), int(0.175/dy), 0] + zb[int(5.05/dx), int(0.175/dy), 0] # 取对应时间的值
            file.write(f"{times*dt:.2f}, {value}\n")

        with open("outputp5Widening.txt", "a") as file:
            value = h[int(4.3/dx), int(0.425/dy), 0] +zb[int(4.3/dx), int(0.425/dy), 0] # 取对应时间的值
            file.write(f"{times*dt:.2f}, {value}\n")

        with open("outputp6Widening.txt", "a") as file:
            value = h[int(5.05/dx), int(0.425/dy), 0]+zb[int(5.05/dx), int(0.425/dy), 0]  # 取对应时间的值
            file.write(f"{times*dt:.2f}, {value}\n")



    
    if ((times % int(0.5/dt)==0)):  
        U=(u.at[:,:,0:1].get()**2+u.at[:,:,1:2].get()**2)**0.5
        Fr=U/((gravity*h)**0.5)
        show_tecplot_2d_4(casename,times,dt,(x-0.1),(y-0.05),nx,ny,(h+zb),u,zb,Fr)
                
    if ((times % int(0.05/dt)==0) and (times!=0)):

        print(f'{times*dt:.2f}---seconds')

    
end =timer()
