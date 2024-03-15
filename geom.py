import numpy as np
from matplotlib import pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.patches as mpatches
import pandas as pd
class box():
    def __init__(self,P0,a,b,c,name,weight,color):   #P0 is the center of the cuboid while a,b,c respectively are it's x,y,z sizes
        self.P0=P0
        self.a=a
        self.b=b
        self.c=c
        self.col=color
        self.name=name
        self.weight=weight
    def plot(self,ax): #plots one box. This method should NOT be used by itself
        P0=self.P0
        a=self.a
        b=self.b
        c=self.c
        P1=[P0[0]-a/2,P0[1]-b/2,P0[2]-c/2]
        P2=[P0[0]+a/2,P0[1]-b/2,P0[2]-c/2]
        P3=[P0[0]+a/2,P0[1]+b/2,P0[2]-c/2]
        P4=[P0[0]-a/2,P0[1]+b/2,P0[2]-c/2]
        P5=[P0[0]-a/2,P0[1]-b/2,P0[2]+c/2]
        P6=[P0[0]+a/2,P0[1]-b/2,P0[2]+c/2]
        P7=[P0[0]+a/2,P0[1]+b/2,P0[2]+c/2]
        P8=[P0[0]-a/2,P0[1]+b/2,P0[2]+c/2]
        Z=np.array([P1,P2,P3,P4,P5,P6,P7,P8])
        r = [-1,1]

        X, Y = np.meshgrid(r, r)
        # plot vertices
        #ax.scatter3D(Z[:, 0], Z[:, 1], Z[:, 2])

        # list of sides' polygons of figure
        verts = [[Z[0],Z[1],Z[2],Z[3]],
        [Z[4],Z[5],Z[6],Z[7]], 
        [Z[0],Z[1],Z[5],Z[4]], 
        [Z[2],Z[3],Z[7],Z[6]], 
        [Z[1],Z[2],Z[6],Z[5]],
        [Z[4],Z[7],Z[3],Z[0]]]
        ax.add_collection3d(Poly3DCollection(verts,facecolors=self.col,alpha=0.25))
class grid():
    def __init__(self,x_lims,y_lims,z_lims): #Constructor of the grid x_lims,y_lims,z_lims are array such as [0,10] to limit the grid space
        ax=plt.subplot(111,projection='3d')
        ax.set_xlim(x_lims[0],x_lims[1])
        ax.set_ylim(y_lims[0],y_lims[1])
        ax.set_zlim(z_lims[0],z_lims[1])
        ax.set_xlabel("x[m]")
        ax.set_ylabel("y[m]")
        ax.set_zlabel("z[m]")
        self.ax=ax
        self.boxes=[]
    def add_box(self,box_obj): #To add some boxes manually
        self.boxes.append(box_obj)
    def add_box_from_txt(self,file_name): #Adds boxes defined in "components.txt" 
        txt=np.genfromtxt(file_name,dtype=str)
        for t in txt:
            name=t[0]
            weight=t[1]
            P0=[np.float64(t[2]),np.float64(t[3]),np.float64(t[4])]
            a=np.float64(t[5])
            b=np.float64(t[6])
            c=np.float64(t[7])
            color=t[8]
            self.boxes.append(box(P0,a,b,c,name,color))
    def plot(self): #Plots all boxes on the grid
        handles=[]
        for b in self.boxes:
            patch = mpatches.Patch(color=b.col, label=b.name)
            handles.append(patch)
            b.plot(self.ax)
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.17),ncol=3, fancybox=True, shadow=True,fontsize=11,handles=handles)


g=grid(x_lims=[0,10],y_lims=[0,10],z_lims=[0,10])
g.add_box_from_txt("./data/components.txt")
g.plot()
plt.show()