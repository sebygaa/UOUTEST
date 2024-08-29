import os
import numpy as np

#from move2dir import move
k_list = [1e-9, 1e-8, 1e-7, 1e-6, 1e-5]
v_list = [0.001, 0.002, 0.005, 0.01]
#k_list = [1e-7, 1e-6,]
#v_list = [0.002, 0.005]
t_end = 3
N_time = 300

if not os.path.exists('logs'):
    os.mkdir('logs')
for kk in k_list:
    for vv in v_list:
        argv_str = str(vv)+' '+ str(kk)+' '+str(t_end)+' '+str(N_time)
        os.system('python ./maonlysim/maonly_argv.py '+argv_str+
                '> logs/v' +str(vv)+'k' +str(kk) +'t'+ str(t_end) + 'N'+str(N_time) +'.log') 
        
### Note: the spatial domain is 20 (N=20)
### Change the number of nodes later --> N=100, N=200 ! (original IAST takes too long...)