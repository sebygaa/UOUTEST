import numpy as np
import matplotlib.pyplot as plt
from TargIsotherm import f_IAST


y1_test = 0.2
p_tmp = np.linspace(0, 10, 41) # 1~10 bar

q_tmp = f_IAST(y1_test*p_tmp, (1-y1_test)*p_tmp)

plt.plot(p_tmp, q_tmp[:,0], label ='q1')
plt.plot(p_tmp, q_tmp[:,1], label ='q2')
plt.xlabel('pressure (bar)')
plt.ylabel('uptake (mol/kg)')
plt.legend(fontsize = 14)
plt.savefig('Test_isotherm_rest.png',dpi=300)
