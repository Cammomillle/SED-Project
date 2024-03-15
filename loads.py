import numpy as np
from matplotlib import pyplot as plt 

m_tot=60    
g=9.81
numb_g=6
n_x,n_y,n_z=3,4,4
Lx=0.1*n_x
Ly=0.1*n_y
Lz=0.1*n_z
a_v=0.01
b_v=0.01
a_t=0.01
b_t=0.01
rho=2710 #kg/m^3 assuming aluminium !

def compute_loads(V_struct,Lx,Ly,Lz,m_tot,numb_g,y_max,I_bending,A_v,A_t):
    F=m_tot*g*numb_g
    print("Force",F,"N")
    gamma=F/V_struct
    print("volumic force",gamma,"N/m^3")
    "Force verticale"
    F_v_axial=A_v*gamma*Lx
    print("Vertical Force",F_v_axial,"N")
    sigma_v=F_v_axial/A_v
    "Bending"
    p=gamma*A_t
    M_ty_bending_max=p*Ly**2/12
    sigma_bending_y=M_ty_bending_max*y_max/I_bending
    M_tz_bending_max=p*Lz**2/12
    sigma_bending_z=M_tz_bending_max*y_max/I_bending
    
    return sigma_v,sigma_bending_y,sigma_bending_z
    
def define_geometry(Lx,Ly,Lz,a_v,b_v,a_t,b_t):
    A_v=a_v*b_v
    A_t=a_t*b_t
    V_struct=(4*A_v*Lx+4*A_t*Ly+4*A_t*Lz)
    m_tot_struct=rho*V_struct
    I_bending=a_t**3*b_t/12 #rectangular section
    return V_struct,m_tot_struct,I_bending,a_t/2
def plot_geometry(Lx,Ly,Lz,a_v,b_v,a_t,b_t):
    axes = [4, 3, 4]
    data = np.ones(axes, dtype=bool)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.voxels(data, facecolors='white', edgecolors='grey')
    plt.show()
    
if __name__=="__main__":
    V_struct,m_tot_struct,I_bending,y_max=define_geometry(Lx,Ly,Lz,a_v,b_v,a_t,b_t)
    m_tot=m_tot+m_tot_struct
    print("m_tot",m_tot,"kg")
    A_v=a_v*b_v
    A_t=a_t*b_t
    sigma_v,sigma_bending_y,sigma_bending_z=compute_loads(V_struct,Lx, Ly, Lz, m_tot, numb_g, y_max, I_bending, A_v, A_t)
    print("sigma_v",sigma_v*10**(-6),"in Mpa")
    print("sigma_bending y",sigma_bending_y*10**(-6),"in Mpa")
    print("sigma_bending z",sigma_bending_z*10**(-6),"in Mpa")
    plot_geometry(Lx, Ly, Lz, a_v, b_v, a_t, b_t)
    
    
    