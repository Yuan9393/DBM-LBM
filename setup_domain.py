import numpy as np
import jax.numpy as jnp
from jax import jit


Vertex_Fluid = 10
Vertex_Boundary = 1000

Vertex_KineticDiffuseWall = 1001
Vertex_KineticSpelluarWall = 1002
Vertex_SlipWall = 1003
Vertex_VelocityInlet = 1004
Vertex_VelocityOutlet = 1005
Vertex_ExtrapolPressure1ST = 1006
Vertex_ExtrapolPressure2ND = 1007
Vertex_Periodic = 1008
Vertex_Uniform = 1009
Vertex_BounceBackWall = 1010
Vertex_FreeFlux = 1011
Vertex_ZouHeVelocity = 1012
Vertex_NoneqExtrapol = 1013
Vertex_EQMDiffuseRefl = 1014
Vertex_NonEqExtrapolPressure = 1015
Vertex_SpecularReflection = 1016
Vertex_FreeSlipWall = 1017
Vertex_PartiallySaturatedBoundaryScheme = 1018

Vertex_BoundaryCorner = 2000
Vertex_ImmersedSolid = -1



VG_Fluid = 0  
VG_ImmersedSolid = -1 

VG_IP =0  
VG_IM =1  #
VG_JP = 2
VG_JM = 3
VG_IPJP_O = 4
VG_IPJM_O = 5
VG_IMJP_O = 6
VG_IMJM_O = 7
VG_IPJP_I = 8
VG_IPJM_I = 9
VG_IMJP_I = 10
VG_IMJM_I = 11

def SetEmbededBodyGeometry(nodeType,x,y, geometryProperty) :

 
    OPS_ACC0=nodeType[x-1:x+2,y-1:y+2]
 
    vt=nodeType[x,y] 
    if (Vertex_ImmersedSolid == vt) :
        neiborVertexType=np.zeros(8)
        """
                        6*****2*****4
                        *     *     *
                        *     *     *
                        1***********0
                        *     *     *
                        *     *     *
                        5*****3*****7
        """
        neiborVertexType[0] = OPS_ACC0[1,-1];     
        neiborVertexType[1] = OPS_ACC0[1,0];      
        neiborVertexType[2] = OPS_ACC0[0,1];      
        neiborVertexType[3] = OPS_ACC0[-1,1];     
        neiborVertexType[4] = OPS_ACC0[0,-1];     
        neiborVertexType[5] = OPS_ACC0[-1,0];     
        neiborVertexType[6] = OPS_ACC0[0,0];      
        neiborVertexType[7] = OPS_ACC0[-1,-1];   
         
        fluidNeiborNum=0
        for i in range(8) :
            if (Vertex_ImmersedSolid != neiborVertexType[i]) :
                fluidNeiborNum+=1
            
        
        solidNeiborNumatCoord=0
        for i in range(4):
            if (Vertex_ImmersedSolid == neiborVertexType[i]) :
                solidNeiborNumatCoord+=1

        if (fluidNeiborNum > 0) :
            if (1 == solidNeiborNumatCoord) :
                geometryProperty[x,y] =-2
             # outer corner
            if (2 == solidNeiborNumatCoord) :
                if ((Vertex_ImmersedSolid == neiborVertexType[0] and
                     Vertex_ImmersedSolid == neiborVertexType[1]) or
                    (Vertex_ImmersedSolid == neiborVertexType[2] and
                     Vertex_ImmersedSolid == neiborVertexType[3])) :
                    print(f"({x},{y})This may be a dangling point.") #悬空
                    nodeType[x,y]=0

                

                if (Vertex_ImmersedSolid == neiborVertexType[2] and
                    Vertex_ImmersedSolid == neiborVertexType[1]) :
                    if (Vertex_ImmersedSolid == neiborVertexType[6]) :
                        geometryProperty[x,y] = VG_IPJM_O
                    else :
                        print(f"({x},{y})The solid body may be too thin.")
                        nodeType[x,y]=0
                

                if (Vertex_ImmersedSolid == neiborVertexType[3] and
                    Vertex_ImmersedSolid == neiborVertexType[1]) :
                    if (Vertex_ImmersedSolid == neiborVertexType[5]) :
                        geometryProperty[x,y] = VG_IPJP_O
                    else :
                       print(f"({x},{y})The solid body may be too thin.")
                       nodeType[x,y]=0
                #     
                # 

                if (Vertex_ImmersedSolid == neiborVertexType[3] and
                    Vertex_ImmersedSolid == neiborVertexType[0]) :
                    if (Vertex_ImmersedSolid == neiborVertexType[7]) :
                        geometryProperty[ x,y] = VG_IMJP_O
                    else :
                       print(f"({x},{y})The solid body may be too thin.")
                       nodeType[x,y]=0
                    
                # 
                if (Vertex_ImmersedSolid == neiborVertexType[2] and
                    Vertex_ImmersedSolid == neiborVertexType[0]) :
                    if (Vertex_ImmersedSolid == neiborVertexType[4]) :
                        geometryProperty[ x,y] = VG_IMJM_O
                    else :
                       print(f"({x},{y})The solid body may be too thin.")
                       nodeType[x,y]=0
                #     
                # 
            # 
           
            if (3 == solidNeiborNumatCoord) :
                if (Vertex_ImmersedSolid != neiborVertexType[0]) :
                    geometryProperty[ x,y] = VG_IP
                
                if (Vertex_ImmersedSolid != neiborVertexType[1]) :
                    geometryProperty[ x,y] = VG_IM
                
                if (Vertex_ImmersedSolid != neiborVertexType[2]) :
                    geometryProperty[ x,y] = VG_JP
                
                if (Vertex_ImmersedSolid != neiborVertexType[3]) :
                    geometryProperty[ x,y] = VG_JM
                
            if (4 == solidNeiborNumatCoord) :
                if (1 == fluidNeiborNum) :
                    if (Vertex_ImmersedSolid != neiborVertexType[4]) :
                        geometryProperty[ x,y] = VG_IPJP_I
                    
                    if (Vertex_ImmersedSolid != neiborVertexType[5]) :
                        geometryProperty[ x,y] = VG_IMJM_I
                    
                    if (Vertex_ImmersedSolid != neiborVertexType[6]) :
                        geometryProperty[ x,y] = VG_IMJP_I
                    
                    if (Vertex_ImmersedSolid != neiborVertexType[7]) :
                        geometryProperty[ x,y] = VG_IPJM_I
                    
                else :
                    print(f"({x},{y})There appears to be hanged solid points")
                    nodeType[x,y]=0
                
    return geometryProperty,nodeType


def SetOuterBoundaryGeometry(nodeType,x,y,geometryProperty):
    if nodeType[x,y]==0: 
        return geometryProperty
    if x==0:
        if  y==0:
           
            if nodeType[1,1]==0:
                geometryProperty[ x,y]=9
        elif y==nodeType.shape[1]-1: 
            if nodeType[1,-2]==0:
                geometryProperty[ x,y]=11
        else:
            
            if nodeType[x+1,y]==0 and ((nodeType[x+1,y-1]==0 or nodeType[x+1,y+1]==0  ) or (nodeType[x+1,y-1]==-1 and nodeType[x+1,y+1]==-1  )):
                geometryProperty[ x,y]=3
            elif nodeType[x+1,y]==-1 and nodeType[x+1,y+1]==0 and nodeType[x+1,y-1]==-1 and nodeType[x-1,y]==-1:
                geometryProperty[ x,y]=9
            elif nodeType[x+1,y]==-1 and nodeType[x+1,y+1]==-1 and nodeType[x+1,y-1]==0 and nodeType[x,y+1]==-1:
                geometryProperty[ x,y]=11

    elif x==nodeType.shape[0]-1:
        if  y==0:
            
            if nodeType[-2,1]==0:
                geometryProperty[ x,y]=8
        elif y==nodeType.shape[1]-1: 
            if nodeType[-2,-2]==0:
                geometryProperty[ x,y]=10
        else:
           
            if nodeType[x-1,y]==0 and ((nodeType[x-1,y+1]==0 or nodeType[x-1,y-1]==0) or (nodeType[x-1,y+1]==-1 and nodeType[x-1,y-1]==-1)):
                geometryProperty[ x,y]=2
            elif nodeType[x-1,y+1]==-1 and nodeType[x-1,y]==-1 and nodeType[x-1,y-1]==0 and nodeType[x,y+1]==-1:
                geometryProperty[ x,y]=10
            elif nodeType[x-1,y+1]==0 and nodeType[x-1,y]==-1 and nodeType[x-1,y-1]==-1 and nodeType[x,y-1]==-1:
                geometryProperty[ x,y]=8 
    else:
        if y==0:
            
            if ((nodeType[x+1,y+1]==0 or nodeType[x-1,y+1]==0) or (nodeType[x+1,y+1]==-1 and nodeType[x-1,y+1]==-1) )and nodeType[x,y+1]==0:
                geometryProperty[ x,y]=0
            elif nodeType[x+1,y+1]==0 and nodeType[x-1,y+1]==-1 and nodeType[x,y+1]==-1 and nodeType[x-1,y]==-1:
                geometryProperty[ x,y]=9
            elif nodeType[x+1,y+1]==-1 and nodeType[x-1,y+1]==0 and nodeType[x,y+1]==-1 and nodeType[x+1,y]==-1:
                geometryProperty[ x,y]=8

        else:
            
            if ((nodeType[x+1,y-1]==0 or nodeType[x-1,y-1]==0) or (nodeType[x+1,y-1]==-1 and nodeType[x-1,y-1]==-1)) and nodeType[x,y-1]==0:
                geometryProperty[ x,y]=1
            elif nodeType[x+1,y-1]==0 and nodeType[x-1,y-1]==-1 and nodeType[x,y-1]==-1 and nodeType[x-1,y]==-1:
                geometryProperty[ x,y]=11
            elif nodeType[x+1,y-1]==-1 and nodeType[x-1,y-1]==0 and nodeType[x,y-1]==-1 and nodeType[x+1,y]==-1:
                geometryProperty[ x,y]=10

    return geometryProperty

    
def get_idx_of_VG(Geometrytype):
    
    idx_x,idx_y=np.where(Geometrytype==VG_IP)
    idx_VG_IP=np.array(list(zip(idx_x,idx_y)))


    idx_x,idx_y=np.where(Geometrytype==VG_IM)
    idx_VG_IM=np.array(list(zip(idx_x,idx_y)))


    idx_x,idx_y=np.where(Geometrytype==VG_JP)
    idx_VG_JP=np.array(list(zip(idx_x,idx_y)))

    idx_x,idx_y=np.where(Geometrytype==VG_JM)
    idx_VG_JM=np.array(list(zip(idx_x,idx_y)))


    idx_x,idx_y=np.where(Geometrytype==VG_IPJP_O)
    idx_VG_IPJP_O=np.array(list(zip(idx_x,idx_y)))


    idx_x,idx_y=np.where(Geometrytype==VG_IPJM_O)
    idx_VG_IPJM_O=np.array(list(zip(idx_x,idx_y)))


    idx_x,idx_y=np.where(Geometrytype==VG_IMJP_O)
    idx_VG_IMJP_O=np.array(list(zip(idx_x,idx_y)))


    idx_x,idx_y=np.where(Geometrytype==VG_IMJM_O)
    idx_VG_IMJM_O=np.array(list(zip(idx_x,idx_y)))


    idx_x,idx_y=np.where(Geometrytype==VG_IPJP_I)
    idx_VG_IPJP_I=np.array(list(zip(idx_x,idx_y)))


    idx_x,idx_y=np.where(Geometrytype==VG_IPJM_I)
    idx_VG_IPJM_I=np.array(list(zip(idx_x,idx_y)))


    idx_x,idx_y=np.where(Geometrytype==VG_IMJP_I)
    idx_VG_IMJP_I=np.array(list(zip(idx_x,idx_y)))


    idx_x,idx_y=np.where(Geometrytype==VG_IMJM_I)
    idx_VG_IMJM_I=np.array(list(zip(idx_x,idx_y)))

    idx_VG_list_org=[idx_VG_IP,idx_VG_IM,idx_VG_JP,idx_VG_JM,
                idx_VG_IPJP_O,idx_VG_IPJM_O,idx_VG_IMJP_O,idx_VG_IMJM_O,
                idx_VG_IPJP_I,idx_VG_IPJM_I,idx_VG_IMJP_I,idx_VG_IMJM_I]

    idx_delta_VG_list_org=[[0,1],[0,-1],[-1,0],[1,0],
                           [-1,1],[1,1],[-1,-1],[1,-1],
                           [-1,1],[1,1],[-1,-1],[1,-1]]
    return idx_VG_list_org,idx_delta_VG_list_org


def deal_dry_wet_boundary(node,h,h0):
    
    pass


@jit
def check_dry(node,u):
    indx=node==0
    node1=node.copy()
    node2=node.copy()
    node3=node.copy()
    node4=node.copy()
    node5=node.copy()
    node6=node.copy()
    node7=node.copy()
    node8=node.copy()
    node9=node.copy()
    u1=u.copy()
    u2=u.copy()
    u3=u.copy()
    u4=u.copy()
    u5=u.copy()
    u6=u.copy()
    u7=u.copy()
    u8=u.copy()
    u9=u.copy()

    temp1=node1.at[1:-1,1,0].get()
    
    arr1=node1.at[0:-2,2:,0].get()*node1.at[1:-1,2:,0].get()*node1.at[2:,2:,0].get()*node1.at[0:-2,1:-1,0].get()*node1.at[2:,1:-1,0].get()*node1.at[0:-2,0:-2,0].get()*node1.at[1:-1,0:-2,0].get()*node1.at[2:,0:-2,0].get()
    temp1=jnp.where(arr1 % 2 == 0,1,3)
    temp1=temp1[:, 0] 
    node1=node1.at[1:-1,1,0].set(temp1)
    temp10=arr1 % 2 == 0
    temp10=temp10[:,0]
    u1=u1.at[1:-1,1,0].set(~temp10*u.at[1:-1,1,0].get())
    u=jnp.where(indx,u1,u)
    node=jnp.where(indx,node1,node)

    temp2=node2.at[1:-1,0,0].get()
   
    arr2=node2.at[0:-2,0,0].get() * node2.at[0:-2,1,0].get() * node2.at[1:-1,1,0].get() * node2.at[2:,1,0].get() * node2.at[2:,0,0].get()
    temp2=jnp.where(arr2 % 2 ==0,1,3)
    node2=node2.at[1:-1,0,0].set(temp2)
    temp20=arr2 % 2 == 0
    u2=u2.at[1:-1,0,0].set(~temp20*u.at[1:-1,0,0].get())
    u=jnp.where(indx,u2,u)
    node=jnp.where(indx,node2,node)

    temp3=node3.at[1:-1,-1,0].get()
    
    arr3=node3.at[0:-2,-1,0].get() * node3.at[0:-2,1,0].get()* node3.at[1:-1,1,0].get() * node3.at[2:,1,0].get() * node3.at[2:,-1,0].get()
    temp3=jnp.where(arr3 % 2 ==0,1,3)
    node3=node3.at[1:-1,-1,0].set(temp3)
    temp30=arr3 % 2 == 0
    u3=u3.at[1:-1,-1,0].set(~temp30*u.at[1:-1,-1,0].get())
    u=jnp.where(indx,u3,u)
    node=jnp.where(indx,node3,node)

    temp4=node4.at[0,0,0].get()
    
    arr4=node4.at[0,1,0].get() * node4.at[1,0,0].get() * node4.at[1,1,0].get()
    temp4=jnp.where(arr4 % 2 ==0,1,3)
    node4=node4.at[0,0,0].set(temp4)
    temp40=arr4 % 2 == 0
    u4=u4.at[0,0,0].set(~temp40*u.at[0,0,0].get())
    u=jnp.where(indx,u4,u)
    node=jnp.where(indx,node4,node)

    temp5=node5.at[0,-1,0].get()
    
    arr5=node5.at[0,1,0].get() * node5.at[1,-1,0].get() * node5.at[1,1,0].get()
    temp5=jnp.where(arr5 % 2 ==0,1,3)
    node5=node5.at[0,-1,0].set(temp5)
    temp50=arr5 % 2 == 0
    u5=u5.at[0,-1,0].set(~temp50*u.at[0,-1,0].get())
    u=jnp.where(indx,u5,u)
    node=jnp.where(indx,node5,node)

    temp6=node6.at[0,1,0].get()
  
    arr6=node6.at[0,0,0].get() * node6.at[1,-1,0].get() * node6.at[1,1,0].get()* node6.at[0,-1,0].get() *node6.at[1,0,0].get()
    temp6=jnp.where(arr6 % 2 ==0,1,3)
    node6=node6.at[0,1,0].set(temp6)
    temp60=arr6 % 2 == 0
    u6=u6.at[0,1,0].set(~temp60*u.at[0,1,0].get())
    u=jnp.where(indx,u6,u)
    node=jnp.where(indx,node6,node)

    temp7=node7.at[-1,0,0].get()
  
    arr7=node7.at[-2,1,0].get() * node7.at[-2,0,0].get() * node7.at[-1,1,0].get()
    temp7=jnp.where(arr7 % 2 ==0,1,3)
    node7=node7.at[-1,0,0].set(temp7)
    temp70=arr7 % 2 == 0
    u7=u7.at[-1,0,0].set(~temp70*u.at[-1,0,0].get())
    u=jnp.where(indx,u7,u)
    node=jnp.where(indx,node7,node)

    temp8=node8.at[-1,-1,0].get()

    arr8=node8.at[-2,1,0].get() *node8.at[-2,-1,0].get() * node8.at[-1,1,0].get()
    temp8=jnp.where(arr8 % 2 ==0,1,3)
    node8=node8.at[-1,-1,0].set(temp8)
    temp80=arr8 % 2 == 0
    u8=u8.at[-1,-1,0].set(~temp80*u.at[-1,-1,0].get())
    u=jnp.where(indx,u8,u)
    node=jnp.where(indx,node8,node)

    temp9=node9.at[-1,1,0].get()

    arr9=node9.at[-2,1,0].get() * node9.at[-2,-1,0].get() * node9.at[-1,0,0].get() * node9.at[-2,0,0].get() * node9.at[-1,-1,0].get()
    temp9=jnp.where(arr9 % 2 ==0,1,3)
    node9=node9.at[-1,1,0].set(temp9)
    temp90=arr9 % 2 == 0
    u9=u9.at[-1,1,0].set(~temp90*u.at[-1,1,0].get())
    u=jnp.where(indx,u9,u)
    node=jnp.where(indx,node9,node)
    return u,node
if __name__ == '__main__':
    import pandas as pd
    
    nodeType=pd.read_excel(r"固体.xlsx",header=None).values
    

    geometryProperty=np.ones_like(nodeType)*-1
    row,col=geometryProperty.shape

 

    for i in range(row):
        for j in range(col):
            if i==0 or j==0 or i==row-1 or j==col-1:
                geometryProperty=SetOuterBoundaryGeometry(nodeType,i,j,geometryProperty)

    for i in range(1,row-1):
        for j in range(1,col-1):
            geometryProperty,_=SetEmbededBodyGeometry(nodeType,i,j, geometryProperty)
    
    pd.DataFrame(nodeType).to_excel("固体测试0210.xlsx",header=False,index=False)
    print("Done")


