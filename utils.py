import imageio.v2 as imageio
import os
import matplotlib.pyplot as plt
import numpy as np
from math import ceil
def compose_gif(dir,output_name):
    # files=os.listdir(dir)
    # img_paths = [f'./output/h_dt=0.001_h0=0.01_nb=0.012/{file}' for file in files]
    img_paths = [f'./output/Fr_dt=0.001_h0=0.01_nb=0.012/Fr.0{file}.0.png' for file in range(1,10)]
    img_paths += [f'./output/Fr_dt=0.001_h0=0.01_nb=0.012/Fr.{file}.0.png' for file in range(10,100)]
    gif_images = [imageio.imread(path) for path in img_paths]
    print(len(gif_images))
    imageio.mimsave(f"{output_name}.gif", gif_images, fps=5)


class isPointInQuadrangle(object):

    def __int__(self):
        self.__isInQuadrangleFlag = False

    def cross_product(self, xp, yp, x1, y1, x2, y2):
        return (x2 - x1) * (yp - y1)-(y2 - y1) * (xp - x1)

    def compute_para(self, xp, yp, xa, ya, xb, yb, xc, yc, xd, yd):
        cross_product_ab = isPointInQuadrangle().cross_product(xp, yp, xa, ya, xb, yb)
        cross_product_bc = isPointInQuadrangle().cross_product(xp, yp, xb, yb, xc, yc)
        cross_product_cd = isPointInQuadrangle().cross_product(xp, yp, xc, yc, xd, yd)
        cross_product_da = isPointInQuadrangle().cross_product(xp, yp, xd, yd, xa, ya)
        return cross_product_ab,cross_product_bc,cross_product_cd,cross_product_da

    def is_in_rect(self, aa, bb, cc, dd):
        if (aa > 0 and bb > 0 and cc > 0 and dd > 0) or (aa < 0 and bb < 0 and cc < 0 and dd < 0):
            
            self.__isInQuadrangleFlag= True
        else:
           
            self.__isInQuadrangleFlag = False

        return self.__isInQuadrangleFlag
    

import math

import matplotlib.pyplot as plt
import numpy as np



def BresenhamAlgorithm(x0_original,y0_original,x1_original,y1_original,dx,dy):
    x0=int(x0_original/dx)
    y0=int(y0_original/dy)
    x1=ceil(x1_original/dx)
    y1=ceil(y1_original/dy)
    
    if x0 == x1 and y0 == y1:
        return [x0, y0]
    elif x0 == x1:
        if y0 < y1:
            y_min = y0
            y_max = y1
        else:
            y_min = y1 + 1
            y_max = y0 + 1

        result = []
        for y in range(y_min, y_max):
            result.append([x0, y])
        return result
    elif y0 == y1:
        if x0 < x1:
            x_min = x0
            x_max = x1
        else:
            x_min = x1 + 1
            x_max = x0 + 1
        result = []
        for x in range(x_min, x_max):
            result.append([x, y0])
        return result

    origin_x = x0
    origin_y = y0
    x0 -= origin_x
    y0 -= origin_y
    x1 -= origin_x
    y1 -= origin_y

    situation = 0
    if x1 > x0 and y1 > y0:
        if (y1 - y0) < (x1 - x0):
            situation = 11
        else:
            situation = 12
    elif x1 < x0 and y1 > y0:
        if (y1 - y0) < (x0 - x1):
            situation = 24
        else:
            situation = 23
    elif x1 < x0 and y1 < y0:
        if (y0 - y1) < (x0 - x1):
            situation = 35
        else:
            situation = 36
    elif x1 > x0 and y1 < y0:
        if (y0 - y1) < (x1 - x0):
            situation = 48
        else:
            situation = 47

    
    def Swap(xt, yt):
        tmp = xt
        xt = yt
        yt = tmp
        return xt, yt

    def TransformQuadrant(xt, yt, pose, back=False):
        if pose == 12:
            xt, yt = Swap(xt, yt)
        elif pose == 23:
            xt = -xt
            xt, yt = Swap(xt, yt)
        elif pose == 24:
            xt = -xt
        elif pose == 35:
            xt = -xt
            yt = -yt
        elif pose == 36:
            xt = -xt
            yt = -yt
            xt, yt = Swap(xt, yt)
        elif pose == 47:
            yt = -yt
            xt, yt = Swap(xt, yt)
        elif pose == 48:
            yt = -yt

        if back:
            if pose == 23 or pose == 47:
                xt = -xt
                yt = -yt
        return xt, yt

   
    x1, y1 = TransformQuadrant(x1, y1, situation)
   
    delta_x = x1 - x0
    delta_y = y1 - y0
    error = 0
    y = y0
    result = []
    for x in range(x0, x1):
        result.append([x, y])
        error += delta_y
        if 2 * error > delta_x:
      
            error -= delta_x
            y += 1


    for pos in result:
        pos[0], pos[1] = TransformQuadrant(pos[0], pos[1], situation, True)

    for pos in result:
        pos[0] += origin_x
        pos[1] += origin_y
        
    return result


import jax.numpy as jnp

def show_tecplot_2d(casename,time,dt0,x,y,nx,ny,var,var_name='h'):
     current_dir = os.getcwd()
     subfolder_name = var_name+'_'+casename
     subfolder_path = os.path.join(current_dir,"output_for_tecplot", subfolder_name)
     os.makedirs(subfolder_path, exist_ok=True)
     filename = f'output{time*dt0}.dat'
     full_path = os.path.join(subfolder_path, filename)
    
     data = np.column_stack((np.tile(x[:,0], ny), jnp.repeat(y[0, :], nx), var.flatten('F')))
     with open(full_path, 'w') as file:
          file.write('TITLE="text1"\n')
          file.write(f'VARIABLES="X" "Y" "{var_name}"\n')
          file.write(f'ZONE T="ZONE1", I={int(nx)}, J={int(ny)}, F=POINT\n')
          np.savetxt(file, data, fmt='%.9f')

def show_tecplot_2d_2(casename,time,dt0,x,y,nx,ny,h,u,zb):
     """
     输出四个量h,u,v,zb
     """
     current_dir = os.getcwd()
     subfolder_name = 'huvzb_'+casename
     subfolder_path = os.path.join(current_dir,"output_for_tecplot", subfolder_name)
     os.makedirs(subfolder_path, exist_ok=True)
     filename = f'output{time*dt0}.dat'
     full_path = os.path.join(subfolder_path, filename)
    #  h=h.at[jnp.isnan(h)].set(0)
    #  u=u.at[jnp.isnan(u)].set(0)
     data = np.column_stack((np.tile(x[:,0], ny), np.repeat(y[0, :], nx), h.at[:,:,0].get().flatten('F'),u.at[:,:,0].get().flatten('F'),u.at[:,:,1].get().flatten('F'),zb.at[:,:,0].get().flatten('F')))
    #  data=data[~np.isnan(data).any(axis=1)]
     with open(full_path, 'w') as file:
          file.write('TITLE="text1"\n')
          file.write(f'VARIABLES="X" "Y" "h" "u" "v" "zb"\n')
          file.write(f'ZONE T="ZONE1", I={int(nx)}, J={int(ny)}, F=POINT\n')
          np.savetxt(file, data, fmt='%.9f')

def show_tecplot_2d_4(casename,time,dt0,x,y,nx,ny,h,u,zb,rho):
    
     current_dir = os.getcwd()
     subfolder_name = 'h_u_v_zb_rho_'+casename
     subfolder_path = os.path.join(current_dir,"output_for_tecplot", subfolder_name)
     os.makedirs(subfolder_path, exist_ok=True)
     filename = f'output{time*dt0:.2f}.dat'
     full_path = os.path.join(subfolder_path, filename)
    #  h=h.at[jnp.isnan(h)].set(0)
    #  u=u.at[jnp.isnan(u)].set(0)
     data = np.column_stack((np.tile(x[:,0], ny), jnp.repeat(y[0, :], nx), h.at[:,:,0].get().flatten('F'),u.at[:,:,0].get().flatten('F'),u.at[:,:,1].get().flatten('F'),zb.at[:,:,0].get().flatten('F'),rho.at[:,:,0].get().flatten('F')))
     with open(full_path, 'w') as file:
          file.write('TITLE="text1"\n')
          file.write(f'VARIABLES="X" "Y" "h" "u" "v" "zb" "rho"\n')
          file.write(f'ZONE T="ZONE1", I={int(nx)}, J={int(ny)}, F=POINT\n')
          np.savetxt(file, data, fmt='%.13f')

def show_tecplot_2d_1(casename,time,dt0,x,y,nx,ny,K):
    
     current_dir = os.getcwd()
     subfolder_name = '单个量_'+casename
     subfolder_path = os.path.join(current_dir,"output_for_tecplot", subfolder_name)
     os.makedirs(subfolder_path, exist_ok=True)
     filename = f'output{time*dt0:.2f}.dat'
     full_path = os.path.join(subfolder_path, filename)
     data = np.column_stack((np.tile(x[:,0], ny), jnp.repeat(y[0, :], nx), K.at[:,:,0].get().flatten('F')))
    #  data=data[~np.isnan(data).any(axis=1)]
     with open(full_path, 'w') as file:
          file.write('TITLE="text1"\n')
          file.write(f'VARIABLES="X" "Y" "K"\n')
          file.write(f'ZONE T="ZONE1", I={int(nx)}, J={int(ny)}, F=POINT\n')
          np.savetxt(file, data, fmt='%.9f')



def show_txt_1D(time, dt, x: np.ndarray, rho: np.ndarray, filename_prefix: str,mid):
    current_dir = os.getcwd()
    subfolder_name = 'cout_txt_1D'
    subfolder_path = os.path.join(current_dir, subfolder_name)
    os.makedirs(subfolder_path, exist_ok=True)

    
    subfolder_for_file = os.path.join(subfolder_path, filename_prefix)
    os.makedirs(subfolder_for_file, exist_ok=True)

    
    filename = f'output{time*dt}.dat'
    full_path = os.path.join(subfolder_for_file, filename)

    
    data = np.hstack((x, rho[:, mid, 0].reshape(-1, 1)))
    np.savetxt(full_path, data, fmt='%f')


import pandas as pd
def outputCsv(name,type1):
   
    df = pd.DataFrame(type1)
    df.to_csv(name, index=False, header=False)
if __name__ == '__main__':


    compose_gif('./output/h_2DRunoff某城市1014','入口固定流量')    

    pass


