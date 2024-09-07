# %% 
# Importing packages

# %%
import numpy as np
import sys, os

from scipy.integrate import odeint
#from IASTrigCal import IAST_bi, PredLinIAST3D
from TargIsotherm import f_IAST
import pickle

import matplotlib.pyplot as plt

import time
from datetime import datetime

# %%
# Parameters with time-related units

# %%
L = 1               # (m)
v = float(sys.argv[1])             # (m/sec) Default=0.05
N = 21              # -
MTC_input = float(sys.argv[2]) # Default = 0.1 (1/sec)
k_mass = [MTC_input, MTC_input] # mass transfer coefficient
D_dif = 1E-6        # (m^2/sec) axial dispersion coeffic.

t_end = float(sys.argv[3])
N_time = int(sys.argv[4])
# %%
# Parma. w/o time term

# %%
R_gas = 8.3145      # (J/mol/K) gas const.
T_gas = 300         # (K)
rho_s = 1000        # (kg/m^3)
epsi = 0.3          # (m^3/m^3) void frac

# %%
# Boundary condition

# %%
C1_sta = 1*8E5/R_gas/T_gas  # (mol/m^3)
C2_sta = 0*8E5/R_gas/T_gas  # (mol/m^3)

# %%
# FDM matrix generating

# %%
h = L/(N-1)
h_arr = h*np.ones(N)
d0 = np.diag(-1/h_arr[1:], k=-1)
d1 = np.diag(1/h_arr, k = 0)
d = d0+d1
d[0,0] = 0
d[-1,-1] = 0
d[-1, -2] = 0
#print('d = ')
#print(d)

dd0 = np.diag(1/h_arr[1:]**2, k=-1)
dd1 = np.diag(-2/h_arr**2)
dd2 = np.diag(1/h_arr[1:]**2,k=1)
dd =dd0 + dd1 + dd2
dd[0,:] = 0
dd[-1,-2] = 1/h**2
dd[-1,-1] = 1/h**2
#print('dd=')
#print(dd)

# %%
# ODE equation in function form 

# %%
T_sys = 300*np.ones(N)
def massbal(y,t):
    C1 = y[0:N]
    C2 = y[1*N:2*N]
    q1 = y[2*N:3*N]
    q2 = y[3*N:4*N]
    
    dC1 = d@C1
    dC2 = d@C2
    
    ddC1 = dd@C1
    ddC2 = dd@C2
    P1 = C1*R_gas*T_gas
    P2 = C2*R_gas*T_gas

    qsta = f_IAST(P1/1E5, P2/1E5,T_sys)
    qsta1 = qsta[:,0]
    qsta2 = qsta[:,1]
    dq1dt = k_mass[0]*(qsta1 - q1)
    dq2dt = k_mass[1]*(qsta2 - q2)

    dC1dt= -v*dC1 + D_dif*ddC1 - (1-epsi)/epsi*rho_s*dq1dt
    dC2dt= -v*dC2 + D_dif*ddC2 - (1-epsi)/epsi*rho_s*dq2dt
    dC1dt[0] = v*(C1_sta-C1[0])/h -(1-epsi)/epsi*rho_s*dq1dt[0]
    dC2dt[0] = v*(C2_sta-C2[0])/h-(1-epsi)/epsi*rho_s*dq1dt[0]
    dC1dt[-1] = v*(C1[-2] - C1[-1])/h - (1-epsi)/epsi*rho_s*dq1dt[-1]
    dC2dt[-1] = v*(C2[-2] - C2[-1])/h - (1-epsi)/epsi*rho_s*dq2dt[-1]

    dydt = np.concatenate((dC1dt,dC2dt,dq1dt,dq2dt))
    return dydt
# %% 
# Inital conditions
# %%
C1_init = 0*8E5/R_gas/T_gas*np.ones(N)  # initial mol frac = 0
C2_init = 1*8E5/R_gas/T_gas*np.ones(N)  # initial mol frac = 0
P_init = (C1_init + C2_init)*R_gas*T_gas
y1_init = C1_init/(C1_init + C2_init)
T_init = 300*np.ones(N)
q_init = f_IAST(y1_init*P_init/1e5, (1-y1_init)*P_init/1e5, T_init)
q1_init = q_init[:,0]
q2_init = q_init[:,1]

# %%
# Solve PDE
tic = time.time()
y0 = np.concatenate((C1_init, C2_init, q1_init, q2_init))
t_test =np.linspace(0,t_end, N_time)
y_res = odeint(massbal, y0, t_test)
toc = time.time() - tic

# %%
# Save the output and log 

# %%

now = datetime.now()
now_date  = now.date()
#print('CPU time : ', toc/60, 'min')
fnamCPU = 'res_lin'+ str(now.date())+'_v'+ sys.argv[1]+'_k'+ sys.argv[2] + '.txt'
fnamPick = 'res_lin'+ str(now.date())+'_v'+sys.argv[1]+ '_k'+sys.argv[2] + '.pkl'

dirnam = 'sim_result'
base_path = os.getcwd()
if not os.path.exists(dirnam):
    os.mkdir(dirnam)

os.chdir(base_path)
os.chdir(dirnam)

f = open(fnamCPU, 'w')
f.write(str(now) + '\n{0:.6f} sec'.format(toc))
f.close()

f = open(fnamPick, 'wb')
pickle.dump(y_res, f,)
f.close()

os.chdir(base_path)
