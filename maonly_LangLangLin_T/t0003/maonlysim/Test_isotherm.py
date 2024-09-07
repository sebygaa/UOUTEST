import numpy as np
import matplotlib.pyplot as plt
from TargIsotherm import f_IAST


y1_test = 0.2
p_tmp = np.linspace(0, 10, 41) # 1~10 bar
T1 = 290*np.ones_like(p_tmp)
T2 = 310*np.ones_like(p_tmp)

q_tmp = f_IAST(y1_test*p_tmp, (1-y1_test)*p_tmp, T1 )

plt.figure()
plt.plot(p_tmp, q_tmp[:,0], label ='q1 T={0:.0f}'.format(T1[0]))
plt.plot(p_tmp, q_tmp[:,1], label ='q2 T={0:.0f}'.format(T1[0]))
plt.xlabel('pressure (bar)')
plt.ylabel('uptake (mol/kg)')
plt.legend(fontsize = 14)
plt.savefig('Test_isotherm_rest_T1.png',dpi=300)

q_tmp = f_IAST(y1_test*p_tmp, (1-y1_test)*p_tmp, T2 )

plt.figure()
plt.plot(p_tmp, q_tmp[:,0], label ='q1 T={0:.0f}'.format(T2[0]))
plt.plot(p_tmp, q_tmp[:,1], label ='q2 T={0:.0f}'.format(T2[0]))
plt.xlabel('pressure (bar)')
plt.ylabel('uptake (mol/kg)')
plt.legend(fontsize = 14)
plt.savefig('Test_isotherm_rest_T2.png',dpi=300)
