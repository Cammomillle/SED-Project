import numpy as np
from matplotlib import pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d.art3d import Line3DCollection
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
    def inertia_matrix(self):
        Ixx=1/12*self.weight*(self.c**2+self.b**2) 
        Iyy=1/12*self.weight*(self.a**2+self.c**2)
        Izz=1/12*self.weight*(self.b**2+self.a**2)
        I=np.matrix([[Ixx,0,0],[0,Iyy,0],[0,0,Izz]]) #Diagonal because in the main inertia axis of the parallelepiped 
        #print("I",I)
        return I
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
        ax.add_collection3d(Poly3DCollection(verts,facecolors=self.col,alpha=0.5))
class cylinder():
    def __init__(self,P0,length,radius,name,weight,color):
        self.P0=P0
        self.length=length
        self.radius=radius
        self.name=name
        self.weight=weight
        self.col=color
        
class grid():
    def __init__(self,x_lims,y_lims,z_lims,m_approx): #Constructor of the grid x_lims,y_lims,z_lims are array such as [0,10] to limit the grid space
        ax=plt.subplot(111,projection='3d')
        ax.set_xlim(x_lims[0],x_lims[1])
        ax.set_ylim(y_lims[0],y_lims[1])
        ax.set_zlim(z_lims[0],z_lims[1])
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        ax.set_zlabel("z [m]")
        x_ticks=np.arange(0,x_lims[1]+0.01,0.1)
        print(x_ticks)
        y_ticks=np.arange(0,y_lims[1]+0.01,0.1)
        z_ticks=np.arange(0,z_lims[1]+0.01,0.1)
        ax.set_yticks(y_ticks)
        ax.set_xticks(x_ticks)
        ax.set_zticks(z_ticks)
        self.ax=ax
        self.boxes=[]
        self.x_size=x_lims[1]
        self.y_size=y_lims[1]
        self.z_size=z_lims[1]
        self.weight=m_approx
    def add_box(self,box_obj): #To add some boxes manually
        self.boxes.append(box_obj)
    def add_box_from_txt(self,file_name): #Adds boxes defined in "components.txt" 
        txt=np.genfromtxt(file_name,dtype=str)
        for t in txt:
            name=t[0]
            weight=np.float64(t[1])
            P0=[np.float64(t[2]),np.float64(t[3]),np.float64(t[4])]
            a=np.float64(t[5])
            b=np.float64(t[6])
            c=np.float64(t[7])
            color=t[8]
            self.boxes.append(box(P0,a,b,c,name,weight,color))
    def define_structure(self,x_size,y_size,z_size):
        ax=self.ax
        P0=(0,0,0)
        P1=(x_size,0,0)
        P2=(0,y_size,0)
        P3=(0,0,z_size)
        P4=(x_size,0,z_size)
        P5=(x_size,y_size,z_size)
        P6=(x_size,y_size,0)
        P7=(0,y_size,z_size)
        segments=[[P0,P1],[P0,P2],[P0,P3],[P3,P4],[P1,P6],[P6,P5],[P5,P7],[P2,P7],[P1,P4],[P4,P5],[P7,P3],[P2,P6]]
        lines=Line3DCollection(segments,color="black")
        ax.add_collection3d(lines)
        
    def center_of_mass(self):
        CG=[0,0,0]
        w_tot=0
        x_cg,y_cg,z_cg=0,0,0
        for b in self.boxes:
            w_tot=w_tot+b.weight
            x_cg=x_cg+b.P0[0]*b.weight
            y_cg=y_cg+b.P0[1]*b.weight
            z_cg=z_cg+b.P0[2]*b.weight
        CG[0]=x_cg/w_tot
        CG[1]=y_cg/w_tot
        CG[2]=z_cg/w_tot
        return CG
    def inertia_matrix(self):
        J=np.zeros((3,3))
        CG=np.array(self.center_of_mass())
        for b in self.boxes:
            P0=b.P0
            R=np.abs(CG-P0)
            print(R)
            J=J+b.inertia_matrix()+b.weight*(R @ R- np.outer(R,R)) #J=I+m*(RRI-tensorial_prod(R,R))
        return J
    def plot(self): #Plots all boxes on the grid
        handles=[]
        self.ax.grid(False)
        for b in self.boxes:
            patch = mpatches.Patch(color=b.col, label=b.name)
            handles.append(patch)
            b.plot(self.ax)
        CG=self.center_of_mass()
        self.ax.scatter(CG[0],CG[1],CG[2],label="CG",color="black")
        patch=Line2D([0], [0], marker='o', color='w', label='CG',markerfacecolor="black", markersize=9)
        handles.append(patch)
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.17),ncol=3, fancybox=True, shadow=True,fontsize=11,handles=handles)
    def approx_inertia(self):
        Ixx=1/12*self.weight*(self.z_size**2+self.y_size**2) 
        Iyy=1/12*self.weight*(self.x_size**2+self.z_size**2)
        Izz=1/12*self.weight*(self.y_size**2+self.x_size**2)
        I=np.matrix([[Ixx,0,0],[0,Iyy,0],[0,0,Izz]]) #Diagonal because in the main inertia axis of the parallelepiped 
        return I
x_max=0.4
y_max=0.4
z_max=0.3

print("coucou")
g=grid(x_lims=[0,x_max],y_lims=[0,y_max],z_lims=[0,z_max],m_approx=100)
g.add_box_from_txt("./data/components.txt")
g.plot()
g.define_structure(x_max,y_max,z_max)
for azim in range(0,360,90):
        g.ax.view_init(elev=10., azim=azim)
        plt.savefig("mecha_design_azim{0}.pdf".format(azim),bbox_inches='tight')
plt.savefig("mecha_design.pdf",bbox_inches='tight')
print("Inertia matrix J \n",g.inertia_matrix(),"m^2.kg")
print("Approx inertia   \n",g.approx_inertia(),"m^2.kg")
plt.show()