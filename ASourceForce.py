
import jax
from jax import jit
@jit
def CalForce_slope(W, XI,slope,h,u):
    
    Fx=2*slope.at[:,:,0:1].get()
    Fy=2*slope.at[:,:,1:2].get()
    ux=u.at[:,:,0:1].get()
    uy=u.at[:,:,1:2].get()
    eifi=XI.at[0:1,:].get()*Fx+XI.at[1:2,:].get()*Fy
    eiui=XI.at[0:1,:].get()*u.at[:,:,0:1].get()+XI.at[1:2,:].get()*u.at[:,:,1:2].get()
    Fiui=ux*Fx+uy*Fy
    # return  -1*W*h*(  eifi + eiui*eifi -Fiui )
    return  -1*W*h*(  eifi  )
    




@jit
def bed_concentration_MidPoint(c,dx,dy,slope_c):
   
    slope_c=slope_c.at[1:-1,:,0:1].set((c.at[2:,:,:].get()-c.at[:-2,:,:].get())/(2*dx))
    slope_c=slope_c.at[0:1,:,0:1].set(slope_c.at[1:2,:,0:1].get())
    slope_c=slope_c.at[-1:,:,0:1].set(slope_c.at[-2:-1,:,0:1].get())

    slope_c=slope_c.at[:,1:-1,1:2].set((c.at[:,2:,:].get()-c.at[:,:-2,:].get())/(2*dy))
    slope_c=slope_c.at[:,0:1,1:2].set(slope_c.at[:,1:2,1:2].get())
    slope_c=slope_c.at[:,-1:,1:2].set(slope_c.at[:,-2:-1,1:2].get())
    return slope_c


@jit
def CalForce_Friction(u,h,h0,nb,gravaty,XI,W):
    ux=u.at[:,:,0:1].get()
    uy=u.at[:,:,1:2].get()
    uxy=(ux*ux+uy*uy)**0.5
    Fx=gravaty*nb*nb*ux*uxy/(h0**(1/3)*h**(4/3))
    Fy=gravaty*nb*nb*uy*uxy/(h0**(1/3)*h**(4/3))
    fici=XI.at[0:1,:].get()*Fx+XI.at[1:2,:].get()*Fy
    uici=XI.at[0:1,:].get()*u.at[:,:,0:1].get()+XI.at[1:2,:].get()*u.at[:,:,1:2].get()
    Fiui=ux*Fx+uy*Fy
   
    return -1*W*h*( fici)

@jit
def CalForce_concentration(rhos,rhow,h,slope_c,XI,W,u,rho,h0):
    
    NON=(rhos-rhow)/rho
    Fx=NON*h*h0*slope_c.at[:,:,0:1].get()
    Fy=NON*h*h0*slope_c.at[:,:,1:2].get()
    ux=u.at[:,:,0:1].get()
    uy=u.at[:,:,1:2].get()
    fici=XI.at[0:1,:].get()*Fx+XI.at[1:2,:].get()*Fy
    uici=XI.at[0:1,:].get()*u.at[:,:,0:1].get()+XI.at[1:2,:].get()*u.at[:,:,1:2].get()
    Fiui=ux*Fx+uy*Fy
    
    return -h*W*(fici+fici*uici-Fiui)

@jit
def CalForce_up_and_down_flux(rho0,h,XI,W,u,rho,E,D,porosity,gravity,h0):
    ux=u.at[:,:,0:1].get()
    uy=u.at[:,:,1:2].get()
    
    Fx=(2**0.5)*ux*((gravity*h0)**0.5)*(rho0-rho)*(E-D)/(gravity*rho*h*h0*(1-porosity) )
    Fy=(2**0.5)*uy*((gravity*h0)**0.5)*(rho0-rho)*(E-D)/(gravity*rho*h*h0*(1-porosity) )
    
    fici=XI.at[0:1,:].get()*Fx+XI.at[1:2,:].get()*Fy
    uici=XI.at[0:1,:].get()*u.at[:,:,0:1].get()+XI.at[1:2,:].get()*u.at[:,:,1:2].get()
    Fiui=ux*Fx+uy*Fy
    
    return -h*W*(fici+fici*uici-Fiui)


@jit
def CalForce_ED(E,D,W,porosity,Kv):
    
    return W*(E-D)*Kv/(1-porosity)


@jit
def CalForce_BedTime(zb_new,zb_old,W,dt):
   
    return -1*W*(zb_new-zb_old)/dt



import jax.numpy as jnp
@jit
def viscosity(zb,dx,dy):
    fx=jnp.zeros(zb.shape)
    fy=jnp.zeros(zb.shape)
    fx=fx.at[1:-1,:,:].set((zb.at[2:,:,:].get()-2*zb.at[1:-1,:,:].get()+zb.at[0:-2,:,:].get())/(dx*dx))
    fy=fy.at[:,1:-1,:].set((zb.at[:,2:,:].get()-2*zb.at[:,1:-1,:].get()+zb.at[:,0:-2,:].get())/(dy*dy))
    return fx+fy