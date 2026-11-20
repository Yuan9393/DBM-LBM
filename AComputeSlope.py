import jax.numpy as jnp
from jax import jit


@jit
def bed_slope_MidPoint(zb,dx,dy,slope):
   
    slope=slope.at[1:-1,:,0:1].set((zb.at[2:,:,:].get()-zb.at[:-2,:,:].get())/(2*dx))
    slope=slope.at[0:1,:,0:1].set(slope.at[1:2,:,0:1].get())
    slope=slope.at[-1:,:,0:1].set(slope.at[-2:-1,:,0:1].get())

    slope=slope.at[:,1:-1,1:2].set((zb.at[:,2:,:].get()-zb.at[:,:-2,:].get())/(2*dy))
    slope=slope.at[:,0:1,1:2].set(slope.at[:,1:2,1:2].get())
    slope=slope.at[:,-1:,1:2].set(slope.at[:,-2:-1,1:2].get())
    return slope


@jit
def slope_FourPoint(zb, dx, dy,slope):
    slope = slope.at[2:-2, :,0:1].set((-zb[4:, :,:] + 8 * zb[3:-1, :,:] - 8 * zb[1:-3, :,:] + zb[0:-4, :,:]) / (12 * dx))
    slope = slope.at[1:2,:,0:1].set( (zb[2:3,:,:]-zb[0:1,:,:])/2*dx )
    slope = slope.at[-2:-1,:,0:1].set( (zb[-1:,:,:]-zb[-3:-2,:,:])/2*dx )
    slope = slope.at[0:1,:,0:1].set( (zb[1:2,:,:]-zb[0:1,:,:])/dx )
    slope = slope.at[-1:,:,0:1].set( (zb[-1:,:,:]-zb[-2:-1,:,:])/dx )

    slope = slope.at[:, 2:-2,1:2].set( (-zb[:, 4:,:] + 8 * zb[:, 3:-1,:] - 8 * zb[:, 1:-3,:] + zb[:, 0:-4,:]) / (12 * dy))
    slope = slope.at[:,1:2,1:2].set( (zb[:,2:3,:]-zb[:,0:1,:])/2*dy )
    slope = slope.at[:,-2:-1,1:2].set( (zb[:,-1:,:]-zb[:,-3:-2,:])/2*dy )
    slope = slope.at[:,0:1,1:2].set( (zb[:,1:2,:]-zb[:,0:1,:])/dy )
    slope = slope.at[:,-1:,1:2].set( (zb[:,-1:,:]-zb[:,-2:-1,:])/dy )
    
    return slope


@jit
def compute_slope(zb, x, y,slope):
  
    zb = zb.squeeze(axis=-1)
    slopex = jnp.squeeze(slope[:, :, 0:1], axis=-1)
    slopey = jnp.squeeze(slope[:, :, 1:2], axis=-1)
   
    slopex = slopex.at[0:-1,:].set( (zb[1:, :] - zb[:-1, :]) / (x[1:, :] - x[:-1, :]) ) 
    slopex = slopex.at[-1:,:].set(slopex.at[-2:-1,:].get())
  
    slopey = slopey.at[:,0:-1].set( (zb[:, 1:] - zb[:, :-1]) / (y[:, 1:] - y[:, :-1]) )  
    slopey = slopey.at[:,-1:].set(slopey.at[:,-2:-1].get())
    slope = slope.at[:,:,0:1].set(slopex[:,:,None])
    slope = slope.at[:,:,1:2].set(slopey[:,:,None])
    return slope

@jit
def slope_self(zb,dx,dy,slope):
    zb_i_plus_1_2x=(zb[2:,:,:]+zb[1:-1,:,:])/2
    zb_i_minus_1_2x=(zb[1:-1,:,:]+zb[0:-2,:,:])/2
    slope=slope.at[1:-1,:,0:1].set((zb_i_plus_1_2x-zb_i_minus_1_2x)/dx)
    slope=slope.at[0:1,:,0:1].set(slope.at[1:2,:,0:1].get())
    slope=slope.at[-1:,:,0:1].set(slope.at[-2:-1,:,0:1].get())

    zb_i_plus_1_2y=(zb[:,2:,:]+zb[:,1:-1,:])/2
    zb_i_minus_1_2y=(zb[:,1:-1,:]+zb[:,0:-2,:])/2
    slope=slope.at[:,1:-1,1:2].set((zb_i_plus_1_2y-zb_i_minus_1_2y)/dy)
    slope=slope.at[:,0:1,1:2].set(slope.at[:,1:2,1:2].get())
    slope=slope.at[:,-1:,1:2].set(slope.at[:,-2:-1,1:2].get())
    return slope




@jit 
def slope_2upwind(zb,dx,dy,slope):
    slope=slope.at[2:-2,:,0:1].set( (3*zb[2:-2,:,:]-4*zb[1:-3,:,:]+zb[0:-4,:,:])/(2*dx))
    slope=slope.at[:,2:-2,1:2].set( (3*zb[:,2:-2,:]-4*zb[:,1:-3,:]+zb[:,0:-4,:])/(2*dy))
    slope=slope.at[0:2,:,0:1].set( slope.at[2:3,:,0:1].get() )
    slope=slope.at[-2:,:,0:1].set( slope.at[-3:-2,:,0:1].get() )
    slope=slope.at[:,0:2,1:2].set( slope.at[:,2:3,0:1].get() )
    slope=slope.at[:,-2:,1:2].set( slope.at[:,-3:-2,0:1].get() )
    return slope





    

