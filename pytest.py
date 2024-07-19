# %%
# IMPORT PACKAGES

# %%
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

# %%
# SIMPLE ODE PROBLEMS

# %%
def model1(y,t):
    dydt = -0.5*y
    return dydt

# %%
# SOLVE IT

# %%
t = np.linspace(0,100,101)
y0 = 1
y_res = odeint(model1, y0, t)

# %%
# GRAPH

# %%
plt.plot(t,y_res)
plt.xlabel('time')
plt.ylabel('y value')
plt.savefig('testfig.png')
