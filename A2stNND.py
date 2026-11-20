import jax
from jax import jit
import jax.numpy as jnp

@jit
def minmod(a, b):

    return 0.5*jnp.minimum(abs(a),abs(b))*(jnp.sign(a)+jnp.sign(b))
    

@jit
def van_leer_limiter(a,b):
    denominater=a+b+1e-10
    return (a*b>0)*(2*a*b)/denominater


@jit
def convection_flux_NND_fluX(f, xi, dx:float):
    
    Fp_i=0.5 * (xi[0:1,:] + jnp.abs(xi[0:1,:])) * f[2:-2,2:-2,:]
    Fm_i=0.5 * (xi[0:1,:] - jnp.abs(xi[0:1,:])) * f[2:-2,2:-2,:]

    Fp_i_plus_1=0.5*(xi[0:1,:] + jnp.abs(xi[0:1,:]))*f[3:-1,2:-2,:]
    Fm_i_plus_1=0.5 * (xi[0:1,:] - jnp.abs(xi[0:1,:])) * f[3:-1,2:-2,:]

    Fp_i_minus_1=0.5*(xi[0:1,:] + jnp.abs(xi[0:1,:]))*f[1:-3,2:-2,:]
    Fm_i_minus_1=0.5 * (xi[0:1,:] - jnp.abs(xi[0:1,:])) * f[1:-3,2:-2,:]

    Fp_i_plus_2=0.5*(xi[0:1,:] + jnp.abs(xi[0:1,:]))*f[4:,2:-2,:]
    Fm_i_plus_2=0.5 * (xi[0:1,:] - jnp.abs(xi[0:1,:])) * f[4:,2:-2,:]

    Fp_i_minus_2=0.5*(xi[0:1,:] + jnp.abs(xi[0:1,:]))*f[0:-4,2:-2,:]
    Fm_i_minus_2=0.5 * (xi[0:1,:] - jnp.abs(xi[0:1,:])) * f[0:-4,2:-2,:]

   
    deltaFp_i_plus_1_2=Fp_i_plus_1 - Fp_i
    deltaFm_i_plus_1_2=Fm_i_plus_1 - Fm_i

    deltaFp_i_plus_3_2=Fp_i_plus_2 - Fp_i_plus_1
    deltaFm_i_plus_3_2=Fm_i_plus_2 - Fm_i_plus_1

    deltaFp_i_minus_1_2=Fp_i - Fp_i_minus_1
    deltaFm_i_minus_1_2=Fm_i - Fm_i_minus_1

    deltaFp_i_minus_3_2=Fp_i_minus_1 - Fp_i_minus_2
    deltaFm_i_minus_3_2=Fm_i_minus_1 - Fm_i_minus_2



    FL_i_plus_1_2=Fp_i+0.5*minmod(deltaFp_i_plus_1_2,deltaFp_i_minus_1_2)
    FR_i_plus_1_2=Fm_i_plus_1-0.5*minmod(deltaFm_i_plus_1_2,deltaFm_i_plus_3_2)
    F_i_plus_1_2=FL_i_plus_1_2+FR_i_plus_1_2

    FL_i_minus_1_2=Fp_i_minus_1+0.5*minmod(deltaFp_i_minus_1_2,deltaFp_i_minus_3_2)
    FR_i_minus_1_2=Fm_i-0.5*minmod(deltaFm_i_minus_1_2,deltaFm_i_plus_1_2)
    F_i_minus_1_2=FL_i_minus_1_2+FR_i_minus_1_2
    Flux=(F_i_plus_1_2-F_i_minus_1_2)/dx
    #--------------------------

    return Flux


@jit
def convection_flux_NND_fluY(f, xi,dy:float):
    
    Fp_i=0.5 * (xi[1:2,:] + jnp.abs(xi[1:2,:])) * f[2:-2,2:-2,:]
    Fm_i=0.5 * (xi[1:2,:] - jnp.abs(xi[1:2,:])) * f[2:-2,2:-2,:]

    Fp_i_plus_1=0.5*(xi[1:2,:] + jnp.abs(xi[1:2,:]))*f[2:-2,3:-1,:]
    Fm_i_plus_1=0.5 * (xi[1:2,:] - jnp.abs(xi[1:2,:])) * f[2:-2,3:-1,:]

    Fp_i_minus_1=0.5*(xi[1:2,:] + jnp.abs(xi[1:2,:]))*f[2:-2,1:-3,:]
    Fm_i_minus_1=0.5 * (xi[1:2,:] - jnp.abs(xi[1:2,:])) * f[2:-2,1:-3,:]

    Fp_i_plus_2=0.5*(xi[1:2,:] + jnp.abs(xi[1:2,:]))*f[2:-2,4:,:]
    Fm_i_plus_2=0.5 * (xi[1:2,:] - jnp.abs(xi[1:2,:])) * f[2:-2,4:,:]

    Fp_i_minus_2=0.5*(xi[1:2,:] + jnp.abs(xi[1:2,:]))*f[2:-2,0:-4,:]
    Fm_i_minus_2=0.5 * (xi[1:2,:] - jnp.abs(xi[1:2,:])) * f[2:-2,0:-4,:]
    
    deltaFp_i_plus_1_2=Fp_i_plus_1 - Fp_i
    deltaFm_i_plus_1_2=Fm_i_plus_1 - Fm_i

    deltaFp_i_plus_3_2=Fp_i_plus_2 - Fp_i_plus_1
    deltaFm_i_plus_3_2=Fm_i_plus_2 - Fm_i_plus_1

    deltaFp_i_minus_1_2=Fp_i - Fp_i_minus_1
    deltaFm_i_minus_1_2=Fm_i - Fm_i_minus_1

    deltaFp_i_minus_3_2=Fp_i_minus_1 - Fp_i_minus_2
    deltaFm_i_minus_3_2=Fm_i_minus_1 - Fm_i_minus_2


    FL_i_plus_1_2=Fp_i+0.5*minmod(deltaFp_i_plus_1_2,deltaFp_i_minus_1_2)
    FR_i_plus_1_2=Fm_i_plus_1-0.5*minmod(deltaFm_i_plus_1_2,deltaFm_i_plus_3_2)
    F_i_plus_1_2=FL_i_plus_1_2+FR_i_plus_1_2

    FL_i_minus_1_2=Fp_i_minus_1+0.5*minmod(deltaFp_i_minus_1_2,deltaFp_i_minus_3_2)
    FR_i_minus_1_2=Fm_i-0.5*minmod(deltaFm_i_minus_1_2,deltaFm_i_plus_1_2)
    F_i_minus_1_2=FL_i_minus_1_2+FR_i_minus_1_2
    Fluy=(F_i_plus_1_2-F_i_minus_1_2)/dy
    

    return Fluy
@jit
def st2_NND(f,xi,dx,dy):
    return convection_flux_NND_fluX(f, xi, dx)+convection_flux_NND_fluY(f,xi,dy)