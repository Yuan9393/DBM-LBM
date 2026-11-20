import jax
import jax.numpy as jnp
from jax import jit
import Model
from functools import partial
from typing import TypeAlias
Array: TypeAlias = jax.Array


@jit
def calculate_well_balanced_source_2d(
    w: jnp.ndarray,    
    xi: jnp.ndarray,   
    x: jnp.ndarray,    
    y: jnp.ndarray,    
    zb: jnp.ndarray,  
    h: jnp.ndarray,    
    epsilonh2
) -> jnp.ndarray:    
    
    
    # -----------------------------------------------------------------------
    h_center = h[1:-1, 1:-1]   # (Nx-2, Ny-2, 1)
    zb_center = zb[1:-1, 1:-1] # (Nx-2, Ny-2, 1)

    
    # -----------------------------------------------------------------------
    zb_left   = zb[0:-2, 1:-1] # (i-1, j)
    zb_right  = zb[2:,   1:-1] # (i+1, j)
    
    dx_left   = x[1:-1, 1:-1] - x[0:-2, 1:-1] 
    dx_right  = x[2:,   1:-1] - x[1:-1, 1:-1] 
    
    dx_left   = dx_left[..., jnp.newaxis]
    dx_right  = dx_right[..., jnp.newaxis]

    
    # -----------------------------------------------------------------------
    zb_bottom = zb[1:-1, 0:-2] # (i, j-1)
    zb_top    = zb[1:-1, 2:]   # (i, j+1)
    
    dy_bottom = y[1:-1, 1:-1] - y[1:-1, 0:-2] 
    dy_top    = y[1:-1, 2:]   - y[1:-1, 1:-1] 

    dy_bottom = dy_bottom[..., jnp.newaxis]
    dy_top    = dy_top[..., jnp.newaxis]

    # =======================================================================
  
    
    water_level = h_center + zb_center


    zb_left_eff   = jnp.minimum(zb_left,   water_level)
    zb_right_eff  = jnp.minimum(zb_right,  water_level)
    zb_bottom_eff = jnp.minimum(zb_bottom, water_level)
    zb_top_eff    = jnp.minimum(zb_top,    water_level)


    h_eff_left   = jnp.maximum(epsilonh2, water_level - zb_left_eff)
    h_eff_right  = jnp.maximum(epsilonh2, water_level - zb_right_eff)
    h_eff_bottom = jnp.maximum(epsilonh2, water_level - zb_bottom_eff)
    h_eff_top    = jnp.maximum(epsilonh2, water_level - zb_top_eff)



   
    feq_loc = Model.BGKEquilibriumSWE4th_Static(h_center, xi, w)
    
    feq_l = Model.BGKEquilibriumSWE4th_Static(h_eff_left,   xi, w)
    feq_r = Model.BGKEquilibriumSWE4th_Static(h_eff_right,  xi, w)
    feq_b = Model.BGKEquilibriumSWE4th_Static(h_eff_bottom, xi, w)
    feq_t = Model.BGKEquilibriumSWE4th_Static(h_eff_top,    xi, w)

    xi_x = xi[0, :].reshape(1, 1, -1) 
    xi_y = xi[1, :].reshape(1, 1, -1)
    term_x = (
        (xi_x > 0) * xi_x * (feq_loc - feq_l) / dx_left +
        (xi_x < 0) * xi_x * (feq_r - feq_loc) / dx_right
    )
    term_y = (
        (xi_y > 0) * xi_y * (feq_loc - feq_b) / dy_bottom +
        (xi_y < 0) * xi_y * (feq_t - feq_loc) / dy_top
    )

    sources_interior = term_x + term_y
    sources = jnp.zeros(h.shape[:-1] + (w.shape[0],), dtype=h.dtype)
    sources = sources.at[1:-1, 1:-1, :].set(sources_interior)
    return sources


@jit
def minmod(a, b):
  
    return 0.5 * jnp.minimum(jnp.abs(a), jnp.abs(b)) * (jnp.sign(a) + jnp.sign(b))

@jit
def van_leer_limiter(a, b):
   
    eps = 1e-10  # 防止除以零
    return (a * b > 0) * (2 * a * b) / (a + b + eps)

@jit
def mc_limiter(a, b):
    
    return jnp.maximum(0, jnp.minimum(jnp.minimum(2 * a, 2 * b), 0.5 * (a + b)))

g = 9.81
@jit
def calculate_source_NND_2d(
    w: jnp.ndarray,    
    xi: jnp.ndarray,   
    zb: jnp.ndarray,   
    h: jnp.ndarray,    
    dx: float,         
    dy: float,         
    epsilonh2: float
) -> jnp.ndarray:      
    
    # -----------------------------------------------------------------------
    c  = jnp.s_[2:-2, 2:-2]
    
    
    r1 = jnp.s_[3:-1, 2:-2]   # i+1
    l1 = jnp.s_[1:-3, 2:-2]   # i-1
    r2 = jnp.s_[4:,   2:-2]   # i+2
    l2 = jnp.s_[0:-4, 2:-2]   # i-2

    
    t1 = jnp.s_[2:-2, 3:-1]   # j+1
    b1 = jnp.s_[2:-2, 1:-3]   # j-1
    t2 = jnp.s_[2:-2, 4:]     # j+2
    b2 = jnp.s_[2:-2, 0:-4]   # j-2

    
    water_level = h[c] + zb[c]

    
    h_eff_c  = h[c]  
    h_eff_r1 = jnp.maximum(epsilonh2, water_level - jnp.minimum(zb[r1], water_level))
    h_eff_l1 = jnp.maximum(epsilonh2, water_level - jnp.minimum(zb[l1], water_level))
    h_eff_r2 = jnp.maximum(epsilonh2, water_level - jnp.minimum(zb[r2], water_level))
    h_eff_l2 = jnp.maximum(epsilonh2, water_level - jnp.minimum(zb[l2], water_level))

  
    F_c  = Model.BGKEquilibriumSWE4th_Static(h_eff_c, xi, w)
    F_r1 = Model.BGKEquilibriumSWE4th_Static(h_eff_r1, xi, w)
    F_l1 = Model.BGKEquilibriumSWE4th_Static(h_eff_l1, xi, w)
    F_r2 = Model.BGKEquilibriumSWE4th_Static(h_eff_r2, xi, w)
    F_l2 = Model.BGKEquilibriumSWE4th_Static(h_eff_l2, xi, w)

    
    xi_x = xi[0:1, :]
    xi_x_abs = jnp.abs(xi_x)
    
 
    Fp_i         = 0.5 * (xi_x + xi_x_abs) * F_c
    Fm_i         = 0.5 * (xi_x - xi_x_abs) * F_c
    Fp_i_plus_1  = 0.5 * (xi_x + xi_x_abs) * F_r1
    Fm_i_plus_1  = 0.5 * (xi_x - xi_x_abs) * F_r1
    Fp_i_minus_1 = 0.5 * (xi_x + xi_x_abs) * F_l1
    Fm_i_minus_1 = 0.5 * (xi_x - xi_x_abs) * F_l1
    Fp_i_plus_2  = 0.5 * (xi_x + xi_x_abs) * F_r2
    Fm_i_plus_2  = 0.5 * (xi_x - xi_x_abs) * F_r2
    Fp_i_minus_2 = 0.5 * (xi_x + xi_x_abs) * F_l2
    Fm_i_minus_2 = 0.5 * (xi_x - xi_x_abs) * F_l2

 
    deltaFp_i_plus_1_2  = Fp_i_plus_1 - Fp_i
    deltaFp_i_minus_1_2 = Fp_i - Fp_i_minus_1
    deltaFm_i_plus_1_2  = Fm_i_plus_1 - Fm_i
    deltaFm_i_minus_1_2 = Fm_i - Fm_i_minus_1
    deltaFp_i_plus_3_2  = Fp_i_plus_2 - Fp_i_plus_1
    deltaFm_i_plus_3_2  = Fm_i_plus_2 - Fm_i_plus_1
    deltaFp_i_minus_3_2 = Fp_i_minus_1 - Fp_i_minus_2

    FL_i_plus_1_2  = Fp_i + 0.5 * minmod(deltaFp_i_plus_1_2, deltaFp_i_minus_1_2)
    FR_i_plus_1_2  = Fm_i_plus_1 - 0.5 * minmod(deltaFm_i_plus_1_2, deltaFm_i_plus_3_2)
    F_i_plus_1_2   = FL_i_plus_1_2 + FR_i_plus_1_2

    FL_i_minus_1_2 = Fp_i_minus_1 + 0.5 * minmod(deltaFp_i_minus_1_2, deltaFp_i_minus_3_2)
    FR_i_minus_1_2 = Fm_i - 0.5 * minmod(deltaFm_i_minus_1_2, deltaFm_i_plus_1_2)
    F_i_minus_1_2  = FL_i_minus_1_2 + FR_i_minus_1_2

    
    term_x = (F_i_plus_1_2 - F_i_minus_1_2) / dx

    
    h_eff_t1 = jnp.maximum(epsilonh2, water_level - jnp.minimum(zb[t1], water_level))
    h_eff_b1 = jnp.maximum(epsilonh2, water_level - jnp.minimum(zb[b1], water_level))
    h_eff_t2 = jnp.maximum(epsilonh2, water_level - jnp.minimum(zb[t2], water_level))
    h_eff_b2 = jnp.maximum(epsilonh2, water_level - jnp.minimum(zb[b2], water_level))

    F_t1 = Model.BGKEquilibriumSWE4th_Static(h_eff_t1, xi, w)
    F_b1 = Model.BGKEquilibriumSWE4th_Static(h_eff_b1, xi, w)
    F_t2 = Model.BGKEquilibriumSWE4th_Static(h_eff_t2, xi, w)
    F_b2 = Model.BGKEquilibriumSWE4th_Static(h_eff_b2, xi, w)

   
    xi_y = xi[1:2, :]
    xi_y_abs = jnp.abs(xi_y)
    
    Fp_i_y         = 0.5 * (xi_y + xi_y_abs) * F_c
    Fm_i_y         = 0.5 * (xi_y - xi_y_abs) * F_c
    Fp_i_plus_1_y  = 0.5 * (xi_y + xi_y_abs) * F_t1
    Fm_i_plus_1_y  = 0.5 * (xi_y - xi_y_abs) * F_t1
    Fp_i_minus_1_y = 0.5 * (xi_y + xi_y_abs) * F_b1
    Fm_i_minus_1_y = 0.5 * (xi_y - xi_y_abs) * F_b1
    Fp_i_plus_2_y  = 0.5 * (xi_y + xi_y_abs) * F_t2
    Fm_i_plus_2_y  = 0.5 * (xi_y - xi_y_abs) * F_t2
    Fp_i_minus_2_y = 0.5 * (xi_y + xi_y_abs) * F_b2
    Fm_i_minus_2_y = 0.5 * (xi_y - xi_y_abs) * F_b2

    dFp_i_p1_2 = Fp_i_plus_1_y - Fp_i_y
    dFp_i_m1_2 = Fp_i_y - Fp_i_minus_1_y
    dFm_i_p1_2 = Fm_i_plus_1_y - Fm_i_y
    dFm_i_m1_2 = Fm_i_y - Fm_i_minus_1_y
    dFp_i_p3_2 = Fp_i_plus_2_y - Fp_i_plus_1_y
    dFm_i_p3_2 = Fm_i_plus_2_y - Fm_i_plus_1_y
    dFp_i_m3_2 = Fp_i_minus_1_y - Fp_i_minus_2_y

    FL_i_p1_2 = Fp_i_y + 0.5 * minmod(dFp_i_p1_2, dFp_i_m1_2)
    FR_i_p1_2 = Fm_i_plus_1_y - 0.5 * minmod(dFm_i_p1_2, dFm_i_p3_2)
    F_i_p1_2  = FL_i_p1_2 + FR_i_p1_2

    FL_i_m1_2 = Fp_i_minus_1_y + 0.5 * minmod(dFp_i_m1_2, dFp_i_m3_2)
    FR_i_m1_2 = Fm_i_y - 0.5 * minmod(dFm_i_m1_2, dFm_i_p1_2)
    F_i_m1_2  = FL_i_m1_2 + FR_i_m1_2

    term_y = (F_i_p1_2 - F_i_m1_2) / dy

    
    sources_interior = term_x + term_y
    sources = jnp.zeros(h.shape[:-1] + (w.shape[0],), dtype=h.dtype)
    
    sources = sources.at[2:-2, 2:-2, :].set(sources_interior)
    
    return sources





@jit
def ApplySemiImplicitFriction(h0, g, nb, coef_u_transform, u_pred, h, dt):
   
    h_safe = jnp.maximum(h, 0.01 / h0) 

    u_mag = jnp.sqrt(jnp.sum(u_pred**2, axis=-1, keepdims=True))
    dimensionless_factor = (g * (nb**2)) / (h0**(1/3))
    
    K = dimensionless_factor * u_mag / (h_safe**(4/3))
    denominator = 1.0 + dt * K
    u_new = u_pred / denominator
    
    return u_new


@jit
def ApplySemiImplicitFriction2(h0, g, nb, coef_u_transform, u_pred, h, zb, dt):
    
    h_safe = jnp.maximum(h, 0.001 / h0) 
    u_mag = jnp.sqrt(jnp.sum(u_pred**2, axis=-1, keepdims=True))
    
   
    dimensionless_factor = (g * (nb**2)) / (h0**(1/3))
    K_friction = dimensionless_factor * u_mag / (h_safe**(4/3))
    
    
    dz_x = zb[1:, :, :] - zb[:-1, :, :]  
    h_i = h[:-1, :, :]
    h_ip1 = h[1:, :, :]
    
    
    eta_i = h_i + zb[:-1, :, :]
    eta_ip1 = h_ip1 + zb[1:, :, :]
    
    
    xi_interface = 0.5 * jnp.abs(h_i - h_ip1) / jnp.maximum(h_i, 1e-6)
    
   
    xi = jnp.zeros_like(h)
    step_threshold = 0.1 / h0  
    
    
    overtop_up_mask = eta_i > zb[1:, :, :]
    step_up_mask = (dz_x > step_threshold) & overtop_up_mask
    xi = xi.at[:-1, :, :].add(jnp.where(step_up_mask, xi_interface, 0.0))
    
    
    overtop_down_mask = eta_ip1 > zb[:-1, :, :]
    step_down_mask = (dz_x < -step_threshold) & overtop_down_mask
    xi = xi.at[1:, :, :].add(jnp.where(step_down_mask, xi_interface, 0.0))
    
    
    K_step = xi * u_mag / h_safe
    
   
    K_total = K_friction + K_step
    denominator = 1.0 + dt * K_total
    u_new = u_pred / denominator
    
    return u_new


@jit
def apply_wetting_drying_strict(h, u, f, XI, W, h_dry=1e-4, h_wet=1e-2):
    
    max_h_neighbors = jax.lax.reduce_window(
        h, -jnp.inf, jax.lax.max, (3, 3, 1), (1, 1, 1), 'SAME'
    )
    
    
    wet_mask = (h > h_wet).astype(h.dtype)
    
    dry_mask = ((h < h_dry) & (max_h_neighbors < h_dry)).astype(h.dtype)
    
    partial_mask = 1.0 - wet_mask - dry_mask
    
    non_wet_mask = dry_mask + partial_mask  
    
    u_new = u * wet_mask
    
    
    feq_static = Model.BGKEquilibriumSWE4th_Static(h, XI, W)
    
    f_new = f * wet_mask + feq_static * non_wet_mask
    
    return h, u_new, f_new


# 以下为错误实现

@jit
def extract_TVD_correction_X(feq_st, xi, dx: float):
    
    Fp_i = 0.5 * (xi[0:1, :] + jnp.abs(xi[0:1, :])) * feq_st[2:-2, 2:-2, :]
    Fm_i = 0.5 * (xi[0:1, :] - jnp.abs(xi[0:1, :])) * feq_st[2:-2, 2:-2, :]
    Fp_i_plus_1 = 0.5 * (xi[0:1, :] + jnp.abs(xi[0:1, :])) * feq_st[3:-1, 2:-2, :]
    Fm_i_plus_1 = 0.5 * (xi[0:1, :] - jnp.abs(xi[0:1, :])) * feq_st[3:-1, 2:-2, :]
    Fp_i_minus_1 = 0.5 * (xi[0:1, :] + jnp.abs(xi[0:1, :])) * feq_st[1:-3, 2:-2, :]
    Fm_i_minus_1 = 0.5 * (xi[0:1, :] - jnp.abs(xi[0:1, :])) * feq_st[1:-3, 2:-2, :]
    Fp_i_plus_2 = 0.5 * (xi[0:1, :] + jnp.abs(xi[0:1, :])) * feq_st[4:, 2:-2, :]
    Fm_i_plus_2 = 0.5 * (xi[0:1, :] - jnp.abs(xi[0:1, :])) * feq_st[4:, 2:-2, :]
    Fp_i_minus_2 = 0.5 * (xi[0:1, :] + jnp.abs(xi[0:1, :])) * feq_st[0:-4, 2:-2, :]
    Fm_i_minus_2 = 0.5 * (xi[0:1, :] - jnp.abs(xi[0:1, :])) * feq_st[0:-4, 2:-2, :]

    
    deltaFp_i_plus_1_2 = Fp_i_plus_1 - Fp_i
    deltaFp_i_minus_1_2 = Fp_i - Fp_i_minus_1
    deltaFm_i_plus_1_2 = Fm_i_plus_1 - Fm_i
    deltaFm_i_minus_1_2 = Fm_i - Fm_i_minus_1
    deltaFp_i_plus_3_2 = Fp_i_plus_2 - Fp_i_plus_1
    deltaFm_i_plus_3_2 = Fm_i_plus_2 - Fm_i_plus_1
    deltaFp_i_minus_3_2 = Fp_i_minus_1 - Fp_i_minus_2

    
    FL_TVD_i_plus_1_2 = 0.5 * minmod(deltaFp_i_plus_1_2, deltaFp_i_minus_1_2)
    FR_TVD_i_plus_1_2 = -0.5 * minmod(deltaFm_i_plus_1_2, deltaFm_i_plus_3_2)
    F_TVD_i_plus_1_2 = FL_TVD_i_plus_1_2 + FR_TVD_i_plus_1_2

    FL_TVD_i_minus_1_2 = 0.5 * minmod(deltaFp_i_minus_1_2, deltaFp_i_minus_3_2)
    FR_TVD_i_minus_1_2 = -0.5 * minmod(deltaFm_i_minus_1_2, deltaFm_i_plus_1_2)
    F_TVD_i_minus_1_2 = FL_TVD_i_minus_1_2 + FR_TVD_i_minus_1_2
    
    
    tvd_source_x = (F_TVD_i_plus_1_2 - F_TVD_i_minus_1_2) / dx
    return tvd_source_x

@jit
def extract_TVD_correction_Y(feq_st, xi, dy: float):
   
    Fp_i = 0.5 * (xi[1:2, :] + jnp.abs(xi[1:2, :])) * feq_st[2:-2, 2:-2, :]
    Fm_i = 0.5 * (xi[1:2, :] - jnp.abs(xi[1:2, :])) * feq_st[2:-2, 2:-2, :]
    Fp_i_plus_1 = 0.5 * (xi[1:2, :] + jnp.abs(xi[1:2, :])) * feq_st[2:-2, 3:-1, :]
    Fm_i_plus_1 = 0.5 * (xi[1:2, :] - jnp.abs(xi[1:2, :])) * feq_st[2:-2, 3:-1, :]
    Fp_i_minus_1 = 0.5 * (xi[1:2, :] + jnp.abs(xi[1:2, :])) * feq_st[2:-2, 1:-3, :]
    Fm_i_minus_1 = 0.5 * (xi[1:2, :] - jnp.abs(xi[1:2, :])) * feq_st[2:-2, 1:-3, :]
    Fp_i_plus_2 = 0.5 * (xi[1:2, :] + jnp.abs(xi[1:2, :])) * feq_st[2:-2, 4:, :]
    Fm_i_plus_2 = 0.5 * (xi[1:2, :] - jnp.abs(xi[1:2, :])) * feq_st[2:-2, 4:, :]
    Fp_i_minus_2 = 0.5 * (xi[1:2, :] + jnp.abs(xi[1:2, :])) * feq_st[2:-2, 0:-4, :]
    Fm_i_minus_2 = 0.5 * (xi[1:2, :] - jnp.abs(xi[1:2, :])) * feq_st[2:-2, 0:-4, :]

   
    deltaFp_i_plus_1_2 = Fp_i_plus_1 - Fp_i
    deltaFp_i_minus_1_2 = Fp_i - Fp_i_minus_1
    deltaFm_i_plus_1_2 = Fm_i_plus_1 - Fm_i
    deltaFm_i_minus_1_2 = Fm_i - Fm_i_minus_1
    deltaFp_i_plus_3_2 = Fp_i_plus_2 - Fp_i_plus_1
    deltaFm_i_plus_3_2 = Fm_i_plus_2 - Fm_i_plus_1
    deltaFp_i_minus_3_2 = Fp_i_minus_1 - Fp_i_minus_2

    
    FL_TVD_i_plus_1_2 = 0.5 * minmod(deltaFp_i_plus_1_2, deltaFp_i_minus_1_2)
    FR_TVD_i_plus_1_2 = -0.5 * minmod(deltaFm_i_plus_1_2, deltaFm_i_plus_3_2)
    F_TVD_i_plus_1_2 = FL_TVD_i_plus_1_2 + FR_TVD_i_plus_1_2

    FL_TVD_i_minus_1_2 = 0.5 * minmod(deltaFp_i_minus_1_2, deltaFp_i_minus_3_2)
    FR_TVD_i_minus_1_2 = -0.5 * minmod(deltaFm_i_minus_1_2, deltaFm_i_plus_1_2)
    F_TVD_i_minus_1_2 = FL_TVD_i_minus_1_2 + FR_TVD_i_minus_1_2
    
    
    tvd_source_y = (F_TVD_i_plus_1_2 - F_TVD_i_minus_1_2) / dy
    return tvd_source_y

@jit
def calculate_well_balanced_source_2d_2ND(
    w: jnp.ndarray,    
    xi: jnp.ndarray,   
    x: jnp.ndarray,    
    y: jnp.ndarray,    
    zb: jnp.ndarray,   
    h: jnp.ndarray,    
    epsilonh2: float
) -> jnp.ndarray:      
    
    h_center = h[2:-2, 2:-2]   
    zb_center = zb[2:-2, 2:-2] 

    
    zb_left   = zb[1:-3, 2:-2] 
    zb_right  = zb[3:-1, 2:-2] 
    zb_bottom = zb[2:-2, 1:-3] 
    zb_top    = zb[2:-2, 3:-1] 
    
    dx_left   = (x[2:-2, 2:-2] - x[1:-3, 2:-2])[..., jnp.newaxis] 
    dx_right  = (x[3:-1, 2:-2] - x[2:-2, 2:-2])[..., jnp.newaxis] 
    dy_bottom = (y[2:-2, 2:-2] - y[2:-2, 1:-3])[..., jnp.newaxis] 
    dy_top    = (y[2:-2, 3:-1] - y[2:-2, 2:-2])[..., jnp.newaxis] 

    
    water_level = h_center + zb_center
    zb_left_eff   = jnp.minimum(zb_left,   water_level)
    zb_right_eff  = jnp.minimum(zb_right,  water_level)
    zb_bottom_eff = jnp.minimum(zb_bottom, water_level)
    zb_top_eff    = jnp.minimum(zb_top,    water_level)

    h_eff_left   = jnp.maximum(epsilonh2, water_level - zb_left_eff)
    h_eff_right  = jnp.maximum(epsilonh2, water_level - zb_right_eff)
    h_eff_bottom = jnp.maximum(epsilonh2, water_level - zb_bottom_eff)
    h_eff_top    = jnp.maximum(epsilonh2, water_level - zb_top_eff)

    
    feq_loc = Model.BGKEquilibriumSWE4th_Static(h_center, xi, w)
    feq_l   = Model.BGKEquilibriumSWE4th_Static(h_eff_left,   xi, w)
    feq_r   = Model.BGKEquilibriumSWE4th_Static(h_eff_right,  xi, w)
    feq_b   = Model.BGKEquilibriumSWE4th_Static(h_eff_bottom, xi, w)
    feq_t   = Model.BGKEquilibriumSWE4th_Static(h_eff_top,    xi, w)

    
    xi_x = xi[0, :].reshape(1, 1, -1) 
    xi_y = xi[1, :].reshape(1, 1, -1)
    
    term_x_1st = (
        (xi_x > 0) * xi_x * (feq_loc - feq_l) / dx_left +
        (xi_x < 0) * xi_x * (feq_r - feq_loc) / dx_right
    )
    term_y_1st = (
        (xi_y > 0) * xi_y * (feq_loc - feq_b) / dy_bottom +
        (xi_y < 0) * xi_y * (feq_t - feq_loc) / dy_top
    )

    
    feq_st = Model.BGKEquilibriumSWE4th_Static(h, xi, w) 

    
    dx_val = x[2, 2] - x[1, 2]
    dy_val = y[2, 2] - y[2, 1]
    
    tvd_corr_x = extract_TVD_correction_X(feq_st, xi, dx_val)
    tvd_corr_y = extract_TVD_correction_Y(feq_st, xi, dy_val)

    
    sources_interior = term_x_1st + term_y_1st + tvd_corr_x + tvd_corr_y
    
   
    sources = jnp.zeros(h.shape[:-1] + (w.shape[0],), dtype=h.dtype)
    sources = sources.at[2:-2, 2:-2, :].set(sources_interior)
    
    return sources


