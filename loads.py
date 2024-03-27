import numpy as np
from matplotlib import pyplot as plt 
from matplotlib.patches import Rectangle
m_tot=100    
g=9.81
numb_g=10
n_x,n_y,n_z=4,4,3
Lx=0.1*n_x
Ly=0.1*n_y
Lz=0.1*n_z
a_v=0.01
b_v=0.01
a_t=0.015
b_t=0.015
rho=2700 #kg/m^3 assuming aluminium !
E=70*10**9 #Young modulus in GPA
print("Total volume",Lx*Ly*Lz*10**6)
def compute_loads(V_struct,Lx,Ly,Lz,m_tot,numb_g,y_max,I_bending,I_v,A_v,A_t):
    F=m_tot*g*numb_g
    print("Force",F,"N")
    gamma=F/V_struct
    print("volumic force",gamma,"N/m^3")
    "Force verticale"
    F_v_axial=gamma*((A_v)*Lz+1/4*((2*A_t)*Ly+(2*A_t)*Lx))
    print("Vertical Force",F_v_axial,"N")
    sigma_v=F_v_axial/A_v
    "Bending"
    p=gamma*A_t
    M_ty_bending_max=p*Lx**2/12
    sigma_bending_y=M_ty_bending_max*y_max/I_bending
    M_tz_bending_max=p*Ly**2/12
    sigma_bending_z=M_tz_bending_max*y_max/I_bending
    "Buckling"
    K=1.2 #worst case 
    P_crit=np.pi**2*E*I_v/(K*Lz)**2
    print("Buckling P_crit",P_crit)

    
    return sigma_v,sigma_bending_y,sigma_bending_z
    
def define_geometry(Lx,Ly,Lz,a_v,b_v,a_t,b_t):
    A_v=a_v*b_v
    A_t=a_t*b_t
    V_struct=(4*A_v*Lx+4*A_t*Ly+4*A_t*Lz)
    m_tot_struct=rho*V_struct
    I_bending=a_t**3*b_t/12 #rectangular section
    I_v=a_v**3*b_t/12
    return V_struct,m_tot_struct,I_bending,I_v,b_t/2
def plot_beams(a_t,b_t,a_v,b_v):
      a_t=a_t*1000
      b_t=b_t*1000
      a_v=a_v*1000
      b_v=b_v*1000
      ax=plt.subplot()
      ax.add_patch(Rectangle((0,0),a_t,b_t,fill=None,hatch="/",edgecolor="black"))
      ax.set_xlim([-0.02*a_t,1.1*a_t])
      ax.set_ylim([-0.02*b_t,1.1*b_t])
      ax.set_xticks([0,a_t])
      ax.set_yticks([0,b_t])
      ax.set_xlabel("[mm]",fontsize=15)
      ax.set_ylabel("[mm]",fontsize=15)
      ax.spines['top'].set_visible(False)
      ax.spines['right'].set_visible(False)
      ax.spines['bottom'].set_visible(False)
      ax.spines['left'].set_visible(False)
      plt.xticks(fontsize=14)
      plt.yticks(fontsize=14)
      plt.savefig("transversal_beams.svg",bbox_inches='tight')
      plt.show()
      ax2=plt.subplot()
      ax2.add_patch(Rectangle((0,0),a_v,b_v,fill=None,hatch="/",edgecolor="black"))
      ax2.set_xlim([-0.02*a_v,1.1*a_v])
      ax2.set_ylim([-0.02*b_v,1.1*b_v])
      ax2.set_xticks([0,a_v])
      ax2.set_yticks([0,b_v])
      ax2.set_xlabel("[mm]",fontsize=15)
      ax2.set_ylabel("[mm]",fontsize=15)
      ax2.spines['top'].set_visible(False)
      ax2.spines['right'].set_visible(False)
      ax2.spines['bottom'].set_visible(False)
      ax2.spines['left'].set_visible(False)
      plt.xticks(fontsize=14)
      plt.yticks(fontsize=14)
      plt.savefig("vertical_beams.svg",bbox_inches='tight')
      plt.show()
if __name__=="__main__":
    V_struct,m_tot_struct,I_bending,I_v,y_max=define_geometry(Lx,Ly,Lz,a_v,b_v,a_t,b_t)
    print("Guess of mass",m_tot,"kg")
    print("Structural weight",m_tot_struct,"kg")
    A_v=a_v*b_v
    A_t=a_t*b_t
    sigma_v,sigma_bending_y,sigma_bending_z=compute_loads(V_struct,Lx, Ly, Lz, m_tot, numb_g, y_max, I_bending,I_v, A_v, A_t)
    print("sigma_v",sigma_v*10**(-6),"in Mpa")
    print("sigma_bending y",sigma_bending_y*10**(-6),"in Mpa")
    print("sigma_bending z",sigma_bending_z*10**(-6),"in Mpa")
    plot_beams(a_t,b_t,a_v,b_v)    
    
    