import numpy as np
from matplotlib import pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d.art3d import Line3DCollection
import pandas as pd
from mpl_toolkits.mplot3d import art3d
plt.close()
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
    def plot(self,ax,alpha=0.5): #plots one box. This method should NOT be used by itself
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
        ax.add_collection3d(Poly3DCollection(verts,facecolors=self.col,alpha=alpha))
class cylinder():
    def __init__(self,P0,length,radius,name,weight,color): #P0 is the initial position given length and radius
        self.P0=P0
        self.length=length
        self.radius=radius
        self.name=name
        self.weight=weight
        self.col=color
    def inertia_matrix(self):
        Ixx=1/12*self.weight*(3*self.radius**2+self.length**2)
        Iyy=1/12*self.weight*(3*self.radius**2+self.length**2)
        Izz=1/2*self.weight*self.radius**2   
        I=np.matrix([[Ixx,0,0],[0,Iyy,0],[0,0,Izz]]) #Diagonal because in the main inertia axis of the parallelepiped 
        #print("I",I)
        return I
    def plot(self,ax):
        size=2
        size_xy=200
        X, Y, Z = np.zeros((3, size, size_xy), dtype=float)
        z=np.linspace(self.P0[2]-self.length/2,self.P0[2]+self.length/2,size)
        theta_tab=np.linspace(0,2*np.pi,size_xy)
        x=self.P0[0]+np.cos(theta_tab)*self.radius
        y=self.P0[1]+np.sin(theta_tab)*self.radius
        for i in range(len(z)):
            X[i] = x
            Y[i] = y
            Z[i] = np.repeat(z[i], size_xy)
            i = i+1
        ax.plot_surface(X, Y, Z,alpha=0.5,color=self.col)
        p = mpatches.Circle((self.P0[0],self.P0[1]),radius=self.radius,color=self.col)
        p2 = mpatches.Circle((self.P0[0],self.P0[1]),radius=self.radius,color=self.col)
        ax.add_patch(p)
        ax.add_patch(p2)
        art3d.pathpatch_2d_to_3d(p, z=self.P0[2]-self.length/2, zdir="z")
        art3d.pathpatch_2d_to_3d(p2, z=self.P0[2]+self.length/2, zdir="z")
class pannels():
    def __init__(self,weight,Faces_concerned,number_of_hinges,tickness,color,is_deployed):
        self.name="Solar Pannels"
        self.weight_per_m2=weight
        self.Faces_concerned=Faces_concerned
        self.number_of_hinges=number_of_hinges
        self.tickness=tickness
        self.col=color
        P1=[x_size/2,0,z_size/2]
        P2=[x_size/2,y_size,z_size/2]
        P3=[0,y_size/2,z_size/2]
        P4=[x_size,y_size/2,z_size/2]
        if("{0}".format(is_deployed)=="True"):
            P1=[x_size/2,-self.number_of_hinges*z_size/2,z_size]
            P2=[x_size/2,y_size+self.number_of_hinges*z_size/2,z_size]
            P3=[-self.number_of_hinges*z_size/2,y_size/2,z_size]
            P4=[x_size+self.number_of_hinges*z_size/2,y_size/2,z_size]
        S1=x_size*z_size
        S2=x_size*z_size
        S3=y_size*z_size
        S4=y_size*z_size
        length1=[x_size,self.tickness,z_size]
        length2=[x_size,self.tickness,z_size]
        length3=[self.tickness,y_size,z_size]
        length4=[self.tickness,y_size,z_size]
        if("{0}".format(is_deployed)=="True"):
            length1=[length1[0],self.number_of_hinges*length1[2],length1[1]]
            length2=[length2[0],self.number_of_hinges*length2[2],length2[1]]
            length3=[self.number_of_hinges*length3[2],length3[1],length3[0]]
            length4=[self.number_of_hinges*length4[2],length4[1],length4[0]]
        if "{0}".format(self.Faces_concerned)=="[XZ1,XZ2,YZ1,YZ2]":
            Surface_tot=self.number_of_hinges*(S1+S2+S3+S4)
            self.surfaces_mid=[P1,P2,P3,P4]
            self.surf=[S1,S2,S3,S4]
            self.lengths=[length1,length2,length3,length4]
        if "{0}".format(self.Faces_concerned)=="[XZ1,XZ2]":
            Surface_tot=self.number_of_hinges*(S1+S2)
            self.surfaces=[P1,P2]
            self.surf=[S1,S2]
            self.lengths=[length1,length2]
        if "{0}".format(self.Faces_concerned)=="[YZ1,YZ2]":
            Surface_tot=self.number_of_hinges*(S3+S4)
            self.surfaces=[P3,P4]
            self.surf=[S3,S4]
            self.lengths=[length3,length4]
        self.surface_tot=Surface_tot
        print("Installed array surface",self.surface_tot,"[m]")
        self.weight=weight*Surface_tot
        self.boxes_tab=[]
        for P0,s,l in zip(self.surfaces_mid,self.surf,self.lengths):
                a=l[0]
                b=l[1]
                c=l[2]
                weight=self.weight_per_m2*self.number_of_hinges*s
                self.boxes_tab.append(box(P0,a,b,c,None,weight,self.col))
    def plot(self,ax):
        for b in self.boxes_tab:
            b.plot(ax,alpha=0.4)
        
            
            
class grid():
    def __init__(self,x_lims,y_lims,z_lims,x_size,y_size,z_size,m_approx): #Constructor of the grid x_lims,y_lims,z_lims are array such as [0,10] to limit the grid space
        ax=plt.subplot(111,projection='3d')
        ax.grid(visible=True)
        ax.set_xlim(x_lims[0],x_lims[1])
        ax.set_ylim(y_lims[0],y_lims[1])
        ax.set_zlim(z_lims[0],z_lims[1])
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        ax.set_zlabel("z [m]")
        x_ticks=np.arange(x_lims[0],x_lims[1]+0.01,0.1)
        y_ticks=np.arange(y_lims[0],y_lims[1]+0.01,0.1)
        z_ticks=np.arange(z_lims[0],z_lims[1]+0.01,0.1)
        #ax.set_yticks(y_ticks)
        #ax.set_xticks(x_ticks)
        #ax.set_zticks(z_ticks)
        self.ax=ax
        self.components=[]
        self.x_size=x_size
        self.y_size=y_size
        self.z_size=z_size
        self.weight=m_approx
    def add_box(self,box_obj): #To add some boxes manually
        self.components.append(box_obj)
    def add_objects_from_txt(self,file_name): #Adds boxes defined in "components.txt" 
        txt=np.genfromtxt(file_name,dtype=str)
        for t in txt:
            if("{0}".format(t[9])=="box"):
                name=t[0]
                weight=np.float64(t[1])
                P0=[np.float64(t[2])/100,np.float64(t[3])/100,np.float64(t[4])/100]
                a=np.float64(t[5])/100
                b=np.float64(t[6])/100
                c=np.float64(t[7])/100
                color=t[8]
                self.components.append(box(P0,a,b,c,name,weight,color))
                continue
            if("{0}".format(t[9])=="cylinder"):
                name=t[0]
                weight=np.float64(t[1])
                P0=[np.float64(t[2])/100,np.float64(t[3])/100,np.float64(t[4])/100]
                length=np.float64(t[5])/100
                radius=np.float64(t[6])/100
                c=t[7]
                color=t[8]
                self.components.append(cylinder(P0,length,radius,name,weight,color))
                continue
            if("{0}".format(t[9]=="pannels")):
                name="{0}".format(t[0]+t[1])
                weight=np.float64(t[2])
                Faces_concerned=t[3]
                number_of_hinges=np.float64(t[4])
                tickness=np.float64(t[5])/100
                color=t[6]
                is_deployed=t[7]
                self.components.append(pannels(weight,Faces_concerned,number_of_hinges,tickness,color,is_deployed))
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
        for b in self.components:
            try:
                for b2 in b.boxes_tab:
                    x_cg=x_cg+b2.P0[0]*b2.weight
                    y_cg=y_cg+b2.P0[1]*b2.weight
                    z_cg=z_cg+b2.P0[2]*b2.weight
            except:
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
        for b in self.components:
            try:
              for b2 in b.boxes_tab:
                    P0=b2.P0
                    R=np.abs(CG-P0)
                    J=J+b2.inertia_matrix()+b2.weight*(R @ R- np.outer(R,R))
            except:
              P0=b.P0
              R=np.abs(CG-P0)
              J=J+b.inertia_matrix()+b.weight*(R @ R- np.outer(R,R)) #J=I+m*(RRI-tensorial_prod(R,R))
        return J
    def plot(self): #Plots all boxes on the grid
        handles=[]
        for b in self.components:
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
x_size=0.3
y_size=0.3
z_size=0.5
fact=2
x_lims=[-fact*x_size+x_size/2,fact*x_size+x_size/2]
y_lims=[-fact*y_size+y_size/2,fact*y_size+y_size/2]
z_lims=[0,2*z_size]
g=grid(x_size=x_size,y_size=y_size,z_size=z_size,x_lims=x_lims,y_lims=y_lims,z_lims=z_lims,m_approx=100)
g.add_objects_from_txt("./data/components.txt")
g.plot()
g.define_structure(x_size,y_size,z_size)
plt.savefig("mecha_design.pdf",bbox_inches='tight')
print("Inertia matrix J \n",g.inertia_matrix(),"m^2.kg")
print("Approx inertia   \n",g.approx_inertia(),"m^2.kg")
plt.show()