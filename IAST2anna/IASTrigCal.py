# %%
# importing packages

# %%
import numpy as np
from scipy.optimize import minimize
from scipy.integrate import simpson
import pickle

# %%
# Spreading pressure calculation

# %%
def iso2pi(fun, P_o, N_integ):
    P_ran = np.linspace(1E-7, P_o, N_integ)
    q_ran = fun(P_ran)
    q_ov_P= q_ran/P_ran
    pi_ov_RT = simpson(q_ov_P, x = P_ran)
    return pi_ov_RT

# %%
# Error estimating
# %%
def x2err_reg(fun1, fun2, y1, P, x1_est1):
    # y1*P = x1*(P_o1)
    P_o1 = y1*P/x1_est1
    # (1-y1)*P = (1-x1)*P_o2
    P_o2 = (1-y1)*P/(1-x1_est1)
    N_integ_node = 25
    pi_ov_RT1 = iso2pi(fun1, P_o1, N_integ_node)
    pi_ov_RT2 = iso2pi(fun2, P_o2, N_integ_node)
    mse = (pi_ov_RT1 - pi_ov_RT2)**2
    return mse

# %%
# x to q val function x2q
# %%
def x2q(fun1, fun2, y1, P, x1):
    # y1*P = x1*(P_o1)
    P_o1 = y1*P/x1
    # (1-y1)*P = (1-x1)*P_o2
    P_o2 = (1-y1)*P/(1-x1)
    # 1/q_tot = sum( 1/q_o )
    q_o1 = fun1(P_o1)
    q_o2 = fun2(P_o2)
    q_tot = 1/(x1/q_o1 + (1-x1)/q_o2)
    q1 = x1*q_tot
    q2 = (1-x1)*q_tot
    return q1,q2

# %%
## Set Five diff cases
## Case1 : y1 >= 1-1E-6
## Case2 : y1 >= 1-0.1
## Case3 : 0.1 < y1 < 1-0.1
## Case4 : y1 <= 0.1 
## Case5 : y1 <= 1E-6

# %%

# Searching each digit (grid search)
def gridupdate(fun, x_center, mesh, N):
    # x range
    N_left = np.round(N/2)
    N_right = N-N_left
    x_left = x_center - (N_left-1)*mesh
    x_right = x_center + N_right*mesh
    x_ran = np.linspace(x_left,x_right, N)
    # y estimateion
    y_list = []
    for xx in x_ran:
        y_tmp = fun([xx])
        y_list.append(y_tmp)
    y_ran = np.array(y_list)
    # Find minimum
    arg_min = np.argmin(y_ran)
    xmin = x_ran[arg_min]
    ymin = y_ran[arg_min]
    # Activate to see the results
    #print('y_values:')
    #print(ymin)
    #print('arg_min = ', arg_min)
    return xmin, ymin
    
def gridsearch(fun, x_init, mesh_init, 
            n_iter = 10, r_shrink_mesh = 0.3,N_search=20):
    #N_search = 20
    x_update = x_init
    mesh_size = mesh_init
    for ii in range(n_iter):
        x_update, fval = gridupdate(fun,x_update, 
                            mesh_size, N_search)
        mesh_size = mesh_size*r_shrink_mesh
        # Activate to see the results
        #print('x  value = ', x_update)
        #print('Func val. =',fun([x_update]))
        #print('mesh size = ', mesh_size*5)
    return x_update, fval

# FINDING IAST 
def IAST_bi(fun1, fun2, y1, P):
    # Case1: y1 > 1-1E-7
    if P < 1E-4:
        return [0,0], 0
    if y1 > 1-1E-7:
        q = fun1(P)
        return [q,0], 0

    # Case2 : y1 <= 1E-7
    if y1 < 1E-7:
        q= fun2(P)
        return [0 ,q], 0

    # OBJ function 
    def obj_err(x):
        Penalty = 0
        x_est = x[0]
        if x_est > 1-1E-7:
            Penalty = Penalty+1E5*(x_est-1)**2
            x_est = 1-1E-7
        elif x_est < 1E-7:
            Penalty = Penalty+1E5*(x_est)**2
            x_est = 1E-7
        obj_mse = x2err_reg(fun1, fun2,
                        y1, P, x_est)
        return obj_mse + Penalty
    
    # Case3: Regular case with optim solver
    is_comp_err = False
    try:
        x_est_0 = [0.5]
        opt_res = minimize(obj_err, x_est_0,method = 'nelder-mead')
        fval = opt_res.fun
        
        if opt_res.fun > 1E-2:
            is_comp_err = True
        
        if is_comp_err == False:
            x_sol = opt_res.x[0]
            q1,q2 = x2q(fun1, fun2, 
                        y1, P, x_sol)    
#######            print('SuCCeSSS with optim solver !')
            return [q1,q2], fval
    except:
        is_comp_err = True
        #print('Except is working')
    
    # Case4: Optim solver failing case
    if is_comp_err:
        #print('Failing cases')
        if y1 > 0.5:
            x_est0 = 0.8
        else:
            x_est0 = 0.8
#######        print("Failing Case")
        x_sol, fval = gridsearch(obj_err, x_est0, 0.01,
                                n_iter = 7, r_shrink_mesh = 0.4)
        if fval < 1E-5:
            q1,q2 = x2q(fun1, fun2, 
                        y1, P, x_sol)
            return [q1,q2], fval
        else:
            x_est0 = x_sol
            x_sol, fval = gridsearch(obj_err, x_est0, 0.008,
                                n_iter = 7, r_shrink_mesh = 0.15)
        
        if fval < 1E-5:
            q1,q2 = x2q(fun1, fun2, 
                        y1, P, x_sol)
            return [q1,q2], fval
        else:
            x_est0 = x_sol
            x_sol, fval = gridsearch(obj_err, x_est0, 0.005,
                                n_iter = 10, r_shrink_mesh = 0.4)
        
        if fval < 1E-5:
            q1,q2 = x2q(fun1, fun2, 
                        y1, P, x_sol)
            return [q1,q2], fval
        else:
            x_est0 = x_sol
            x_sol, fval = gridsearch(obj_err, x_est0, 0.002,
                                n_iter = 10, r_shrink_mesh = 0.15)
        
            q1,q2 = x2q(fun1, fun2, 
                        y1, P, x_sol)    
            return [q1,q2], fval


# %%
# Generate and Interpolation-based surrogate model
def genIASTdata2D(iso1, iso2, y1_ran, P_ran,
                file_name='genIASTdata2D.pkl'):
    # y1_ran = np.linspace(0,1,50+1)
    # P_ran = np.linspace(0,25,10+1)
    q1_list = []
    q2_list = []

    for yy in y1_ran:
        q1_tmp_list = []
        q2_tmp_list = []
        for PP in P_ran:
            [q1_tmp, q2_tmp], fval = IAST_bi(iso1, iso2,
                                            yy, PP)
            if fval > 1E-2:
                print('[Warnning] pi/RT err = ', fval)
                print('at y1=', yy, 'P=',PP)
            q1_tmp_list.append(q1_tmp)
            q2_tmp_list.append(q2_tmp)

        q1_list.append(q1_tmp_list)
        q2_list.append(q2_tmp_list)

    q1_arr_data = np.array(q1_list)
    q2_arr_data = np.array(q2_list)
    IASTdata = {'q1': q1_arr_data,
                'q2': q2_arr_data,
                'y1': y1_ran,
                'P': P_ran}
    with open(file_name, 'wb') as file:
        pickle.dump(IASTdata, file)
# %%
# Generate and Interpolation-based surrogate model
# %%

def genIASTdata3D(iso1, iso2, y1_ran, P_ran, T_ran,
                file_name='genIASTdata2D.pkl'):
    # y1_ran = np.linspace(0,1,50+1)
    # P_ran = np.linspace(0,25,10+1)
    q1_list = []
    q2_list = []

    for TT in T_ran:
        iso1_fixT = lambda ppp: iso1(ppp,TT)
        iso2_fixT = lambda ppp: iso2(ppp,TT)
        q1_tmp_fixT = []
        q2_tmp_fixT = []

        for yy in y1_ran:
            q1_tmp_list = []
            q2_tmp_list = []
            for PP in P_ran:
                [q1_tmp, q2_tmp], fval = IAST_bi(iso1_fixT, iso2_fixT,
                                                yy, PP)
                if fval > 1E-2:
                    print('[Warnning] pi/RT err = ', fval)
                    print('at y1=', yy, 'P=',PP)
                q1_tmp_list.append(q1_tmp)
                q2_tmp_list.append(q2_tmp)

            q1_tmp_fixT.append(q1_tmp_list)
            q2_tmp_fixT.append(q2_tmp_list)
        q1_list.append(q1_tmp_fixT)
        q2_list.append(q2_tmp_fixT)

    q1_arr_data = np.array(q1_list)
    q2_arr_data = np.array(q2_list)
    IASTdata = {'q1': q1_arr_data,
                'q2': q2_arr_data,
                'y1': y1_ran,
                'P': P_ran,
                'T': T_ran,}
    with open(file_name, 'wb') as file:
        pickle.dump(IASTdata, file)

# %%
# interLinIAST2D
# %%
# Functionize this interpolation algorithm
def interLinIAST2D(x1_targ, x2_targ, x1_minmax, x2_minmax,
                f11,f12,f21,f22):
    x_x1 = (x1_targ - x1_minmax[0])/(x1_minmax[1]-x1_minmax[0])
    x_x2 = (x2_targ - x2_minmax[0])/(x2_minmax[1]-x2_minmax[0])

    # First  dim. reduction 
    f_sol_x1 = (1-x_x1)*f11 + x_x1*f21
    f_sol_x2 = (1-x_x1)*f12 + x_x1*f22
    # second dim. reduction
    f_sol = (1-x_x2)*f_sol_x1 + x_x2*f_sol_x2
    return f_sol

def determ(x_len3):
    x1,x2,x3 = x_len3
    det_res = 1/(x1-x2)/(x1-x3)/(x2-x3)
    return det_res

def interQuad(x_targ, x_len3, y_len3):
    x1,x2,x3 = x_len3
    y1,y2,y3 = y_len3
    d_tmp = determ(x_len3)
    a_tmp = d_tmp*((x3-x1)*(y2-y1) - (x2-x1)*(y3-y1))
    b_tmp = d_tmp*(-(x3-x1)**2*(y2-y1)+(x2-x1)**2*(y3-y1))
    y_pred = a_tmp*(x_targ-x1)**2 + b_tmp*(x_targ-x1) + y1
    return y_pred

def interQuaIAST2D(x1_targ, x2_targ, 
                   x1_da, x2_da,
                   f11,f12,f13,f21,f22,f23,f31,f32,f33):
    
    # First  dim. reduction 
    f_sol_x1 = interQuad(x1_targ, x1_da, [f11,f21,f31])
    f_sol_x2 = interQuad(x1_targ, x1_da, [f12,f22,f32])
    f_sol_x3 = interQuad(x1_targ, x1_da, [f13,f23,f33])
    
    # second dim. reduction
    f_sol = interQuad(x2_targ, x2_da, [f_sol_x1, f_sol_x2, f_sol_x3])
    return f_sol


# %%
# interLinIAST3D
# %%
def interLinIAST3D(y_targ, P_targ,T_targ,
                y1_ran, P_ran, T_ran,
                q_list_T1, q_list_T2):
    q_sol_T1 = interLinIAST2D(y_targ, P_targ,
                           y1_ran,P_ran,
                           q_list_T1[0],q_list_T1[1],
                           q_list_T1[2],q_list_T1[3])
    q_sol_T2 = interLinIAST2D(y_targ, P_targ,
                           y1_ran,P_ran,
                           q_list_T2[0],q_list_T2[1],
                           q_list_T2[2],q_list_T2[3])
    x_T = (T_targ - T_ran[0])/(T_ran[1] - T_ran[0])
    q_sol = (1-x_T)*q_sol_T1 + x_T*q_sol_T2
    return q_sol
# %%
# PredLinIAST2D
# %%
#with open('data.pkl', 'rb') as file:
#    IASTdata = pickle.load(file)

#y1_ran = np.linspace(0,1,50+1)
#P_ran = np.linspace(0,25,10+1)

#q1_arr_data = IASTdata['q1']
#q2_arr_data = IASTdata['q2']

class PredLinIAST2D:
    def __init__(self, file_name='genIASTdata2D.pkl'):
        with open(file_name, 'rb') as file:
            IASTdata = pickle.load(file)
        y1_ran = IASTdata['y1']
        P_ran = IASTdata['P']
        q1_arr_data = IASTdata['q1']
        q2_arr_data = IASTdata['q2']
        
        self.y1 = y1_ran
        self.P = P_ran 
        self.q1 = q1_arr_data
        self.q2 = q2_arr_data
        del(IASTdata)
        
    def predict(self, y1_targ,P_targ,):
        y1_ran = self.y1
        P_ran = self.P
        q1_arr_data = self.q1
        q2_arr_data = self.q2

        y_diff = y1_targ- y1_ran[:-1]
        P_diff = P_targ - P_ran[:-1]

        i_1 = np.argmin(y_diff**2)
        i_2 = np.argmin(P_diff**2)

        if y_diff[i_1] < 0:
            i_1 = i_1 - 1
        if P_diff[i_2] < 0:
            i_2 =i_2 - 1
        
        y1_1 = y1_ran[i_1]
        y1_2 = y1_ran[i_1+1]
        y_list = [y1_1, y1_2]

        P_1 = P_ran[i_2]
        P_2 = P_ran[i_2+1]
        P_list = [P_1, P_2]
        # q1: interLinIAST2D
        q11 = q1_arr_data[i_1, i_2]
        q12 = q1_arr_data[i_1, i_2+1]
        q21 = q1_arr_data[i_1+1, i_2]
        q22 = q1_arr_data[i_1+1, i_2+1]

        q1_sol = interLinIAST2D(y1_targ,P_targ,
                            y_list,P_list,
                            q11,q12,q21,q22)
        
        # q2: interLinIAST2D
        q11 = q2_arr_data[i_1, i_2]
        q12 = q2_arr_data[i_1, i_2+1]
        q21 = q2_arr_data[i_1+1, i_2]
        q22 = q2_arr_data[i_1+1, i_2+1]

        q2_sol = interLinIAST2D(y1_targ,P_targ,
                            y_list,P_list,
                            q11,q12,q21,q22)
        return q1_sol, q2_sol

# %%
# PredLinIAST3D
# %%
class PredLinIAST3D:
    def __init__(self, file_name='genIASTdata3D.pkl'):
        with open(file_name, 'rb') as file:
            IASTdata = pickle.load(file)
        y1_ran = IASTdata['y1']
        P_ran = IASTdata['P']
        T_ran = IASTdata['T']
        q1_arr_data = IASTdata['q1']
        q2_arr_data = IASTdata['q2']
        
        self.y1 = y1_ran
        self.P = P_ran 
        self.T = T_ran
        self.q1 = q1_arr_data
        self.q2 = q2_arr_data
        del(IASTdata)
        
    def predict(self, y1_targ, P_targ, T_targ):
        T_ran = self.T
        y1_ran = self.y1
        P_ran = self.P
        
        q1_arr_data = self.q1
        q2_arr_data = self.q2

        T_diff = T_targ - T_ran[:-1]
        y_diff = y1_targ- y1_ran[:-1]
        P_diff = P_targ - P_ran[:-1]

        i_1 = np.argmin(T_diff**2)
        i_2 = np.argmin(y_diff**2)
        i_3 = np.argmin(P_diff**2)

        if T_diff[i_1] < 0:
            i_1 = i_1 -1
        if y_diff[i_2] < 0:
            i_2 = i_2 - 1
        if P_diff[i_3] < 0:
            i_3 =i_3 - 1
        
        T_1 = T_ran[i_1]
        T_2 = T_ran[i_1+1]
        T_list = [T_1, T_2]

        y1_1 = y1_ran[i_2]
        y1_2 = y1_ran[i_2+1]
        y_list = [y1_1, y1_2]

        P_1 = P_ran[i_3]
        P_2 = P_ran[i_3+1]
        P_list = [P_1, P_2]
        # q1: interLinIAST3D
        # T1
        q11T1 = q1_arr_data[i_1, i_2, i_3]
        q12T1 = q1_arr_data[i_1, i_2, i_3+1]
        q21T1 = q1_arr_data[i_1, i_2+1, i_3]
        q22T1 = q1_arr_data[i_1, i_2+1, i_3+1]
        q_T1 = [q11T1,q12T1,q21T1,q22T1]
        # T2
        q11T2 = q1_arr_data[i_1+1, i_2, i_3]
        q12T2 = q1_arr_data[i_1+1, i_2, i_3+1]
        q21T2 = q1_arr_data[i_1+1, i_2+1, i_3]
        q22T2 = q1_arr_data[i_1+1, i_2+1, i_3+1]
        q_T2 = [q11T2,q12T2,q21T2,q22T2]

        q1_sol = interLinIAST3D(y1_targ,P_targ, T_targ,
                                y_list,P_list,T_list,
                                q_T1, q_T2)
        
        # q2: interLinIAST2D
        # T1
        q11T1 = q2_arr_data[i_1, i_2, i_3]
        q12T1 = q2_arr_data[i_1, i_2, i_3+1]
        q21T1 = q2_arr_data[i_1, i_2+1, i_3]
        q22T1 = q2_arr_data[i_1, i_2+1, i_3+1]
        q_T1 = [q11T1,q12T1,q21T1,q22T1]
        # T2
        q11T2 = q2_arr_data[i_1+1, i_2, i_3]
        q12T2 = q2_arr_data[i_1+1, i_2, i_3+1]
        q21T2 = q2_arr_data[i_1+1, i_2+1, i_3]
        q22T2 = q2_arr_data[i_1+1, i_2+1, i_3+1]
        q_T2 = [q11T2,q12T2,q21T2,q22T2]

        q2_sol = interLinIAST3D(y1_targ, P_targ, T_targ,
                                y_list,P_list,T_list,
                                q_T1, q_T2)

        return q1_sol, q2_sol
    
# %%
# PredLinIAST2D
# %%
#with open('data.pkl', 'rb') as file:
#    IASTdata = pickle.load(file)

#y1_ran = np.linspace(0,1,50+1)
#P_ran = np.linspace(0,25,10+1)

#q1_arr_data = IASTdata['q1']
#q2_arr_data = IASTdata['q2']

class PredQuaIAST2D:
    def __init__(self, file_name='genIASTdata2D.pkl'):
        with open(file_name, 'rb') as file:
            IASTdata = pickle.load(file)
        y1_ran = IASTdata['y1']
        P_ran = IASTdata['P']
        q1_arr_data = IASTdata['q1']
        q2_arr_data = IASTdata['q2']
        
        self.y1 = y1_ran
        self.P = P_ran 
        self.q1 = q1_arr_data
        self.q2 = q2_arr_data
        del(IASTdata)
        
    def predict(self, y1_targ,P_targ,):
        y1_ran = self.y1
        P_ran = self.P
        q1_arr_data = self.q1
        q2_arr_data = self.q2

        y_diff = y1_targ- y1_ran[:-1]
        P_diff = P_targ - P_ran[:-1]

        i_1 = np.argmin(y_diff**2)
        i_2 = np.argmin(P_diff**2)

        if y_diff[i_1] < 0:
            i_1 = i_1 - 1
        if P_diff[i_2] < 0:
            i_2 =i_2 - 1
        if i_1 == len(y_diff)-1:
            i_1 = i_1 - 2
        if i_2 == len(P_diff)-1:
            i_2 = i_2 - 2
        y1_1 = y1_ran[i_1]
        y1_2 = y1_ran[i_1+1]
        y1_3 = y1_ran[i_1+2]
        y_list = [y1_1, y1_2, y1_3]

        P_1 = P_ran[i_2]
        P_2 = P_ran[i_2+1]
        P_3 = P_ran[i_2+2]
        P_list = [P_1, P_2, P_3]
        # q1: interLinIAST2D
        q11 = q1_arr_data[i_1, i_2]
        q12 = q1_arr_data[i_1, i_2+1]
        q13 = q1_arr_data[i_1, i_2+2]
        q21 = q1_arr_data[i_1+1, i_2]
        q22 = q1_arr_data[i_1+1, i_2+1]
        q23 = q1_arr_data[i_1+1, i_2+2]
        q31 = q1_arr_data[i_1+2, i_2]
        q32 = q1_arr_data[i_1+2, i_2+1]
        q33 = q1_arr_data[i_1+2, i_2+2]

        q1_sol = interQuaIAST2D(y1_targ,P_targ,
                            y_list,P_list,
                            q11,q12,q13,
                            q21,q22,q23,
                            q31,q32,q33)
        
        # q2: interLinIAST2D
        q11 = q2_arr_data[i_1, i_2]
        q12 = q2_arr_data[i_1, i_2+1]
        q13 = q2_arr_data[i_1, i_2+2]
        q21 = q2_arr_data[i_1+1, i_2]
        q22 = q2_arr_data[i_1+1, i_2+1]
        q23 = q2_arr_data[i_1+1, i_2+2]
        q31 = q2_arr_data[i_1+2, i_2]
        q32 = q2_arr_data[i_1+2, i_2+1]
        q33 = q2_arr_data[i_1+2, i_2+2]

        q2_sol = interQuaIAST2D(y1_targ,P_targ,
                            y_list,P_list,
                            q11,q12,q13,
                            q21,q22,q23,
                            q31,q32,q33)
        return q1_sol, q2_sol
    


# %%
# TEST CODES 
#[TEST 1] iso2pi
#[TEST 2] x2err
#[TEST 3] Optimization example
#[TEST 4] x2q
#[TEST 5] gridupdate
#[TEST 6] gridsearch
#[TEST 7] IAST_bi
#[TEST 8] interQuad & interQua2D

# %%
## TEST CODE 1: iso2pi

# %%
# TESTING "iso2pi" function
if __name__ == '__main__':
    qm1 = 3
    b1 = 0.5
    def iso_test(P):
        numer = qm1*b1*P
        denom = 1+b1*P
        q = numer/denom
        return q

    P_o_test = 5
    pi_ov_RT_test05 = iso2pi(iso_test, P_o_test, 5)
    pi_ov_RT_test10 = iso2pi(iso_test, P_o_test, 10)
    pi_ov_RT_test20 = iso2pi(iso_test, P_o_test, 20)
    pi_ov_RT_test30 = iso2pi(iso_test, P_o_test, 30)
    print('pi05=',pi_ov_RT_test05)
    print('pi10=',pi_ov_RT_test10)
    print('pi20=',pi_ov_RT_test20)
    print('pi30=',pi_ov_RT_test30)

# %%
## TEST CODE 2: x2err
# %%
# TESTING "x2err" function
if __name__ == "__main__":
    qm1 = 2
    qm2 = 1
    b1 = 0.2
    b2 = 0.4
    def iso_test1(P):
        numer = qm1*b1*P
        denom = 1+b1*P
        q = numer/denom
        return q
        
    def iso_test2(P):
        numer = qm2*b2*P
        denom = 1+b2*P
        q = numer/denom
        return q

    y1_test = 0.5
    P_test = 2
    x_est_test = 0.3
    mse_test = x2err_reg(iso_test1, iso_test2, 
                    y1_test, P_test, x_est_test)
    print(mse_test)

# %%
## TEST CODE 3: Optimization example
# %%

# Optimizing test with x2err_reg
if __name__ == "__main__":
    y1_test = 0.5
    P_test = 2
    err_test = lambda x: x2err_reg(iso_test1, iso_test2,
                            y1_test, P_test, x[0])
    x_est0 = [0.3]
    opt_res = minimize(err_test, x_est0)
    print(opt_res)

# %%
## TEST CODE 4: x2q

# %%
## TESTing x2q
if __name__ == '__main__':
    y1_test = 0.5
    P_test = 2
    err_test = lambda x: x2err_reg(iso_test1, iso_test2,
                            y1_test, P_test, x[0])
    x_est0 = [0.3]
    opt_res = minimize(err_test, x_est0)
    x1_sol_test = opt_res.x[0]
    q1_test, q2_test = x2q(iso_test1, iso_test2, 
                        y1_test, P_test, x1_sol_test)

    print('q1 = ')
    print(q1_test)
    print('q2 = ')
    print(q2_test)

# %%
## TEST CODE 5: gridupdate
# %%
if __name__ == "__main__":
    gridupdate(err_test, 0.1, 0.01, 20)

# %%
## TEST CODE 
# %%
if __name__ == "__main__":
    x_min_grid, fun_val = gridsearch(err_test, 0.4, 0.01)
    print('[Result: gridsearch 7 iterations]')
    print('x=', '{0:.5f}'.format(x_min_grid))

# %%
# TEST 6: IAST_bi
# %%
if __name__ == '__main__':
    # TEST IAST_bi
    y1_test = 0.1
    P_test = 2
    [q1,q2], fval_test = IAST_bi(iso_test1, iso_test2,
                                y1_test, P_test)

    print('q1 = ')
    print(q1)
    print('q2 = ')
    print(q2)
    print('pi_error = ')
    print(fval_test)
# %%
# TEST 7: Test interQuad & Test interQuaIAST2D
# %%
if __name__ =='__main__':
    # 1D data quadratic spline
    x_ran = np.linspace(0,4)
    y_ran = x_ran**2 - 4*x_ran + 1
    ii_list = [1, 22, 35]
    x_pick = []
    y_pick = []
    for ii in ii_list:
        x_pick.append(x_ran[ii])
        y_pick.append(y_ran[ii])
    x_target = 1.4
    # Here is Prediction
    y_predict = interQuad(x_target, x_pick, y_pick)
    # Plot !
    import matplotlib.pyplot as plt
    plt.plot(x_ran, y_ran, label = 'Equation')
    plt.plot(x_pick, y_pick, 'bo', label = 'Sample')
    plt.plot(x_target, y_predict, 'ro', label = 'Predict')
    plt.legend(fontsize = 13)
    # 2D data quadratic spline
    x_ran = np.linspace(0,4)
    y_ran = np.linspace(0,4)
    z_ran = np.zeros([len(x_ran), len(y_ran)])
    for ii, xx in enumerate(x_ran):
        for jj, yy in enumerate(y_ran):
            z_tmp = xx**2 -xx+ yy**2-4*yy-4
            z_ran[ii,jj]=z_tmp
    #z_ran = np.array(z_list)
    ii1_list = [3,10, 14]
    ii2_list = [3,10, 14]
    x_pick = []
    y_pick = []
    
    for ii,jj in zip(ii1_list,ii2_list):
        x_pick.append(x_ran[ii])
        y_pick.append(y_ran[jj])
    
    z_mat = np.zeros([3,3])
    for ien, ii in enumerate(ii1_list):
        for jen, jj in enumerate(ii2_list):
            z_mat[ien,jen] = z_ran[ii,jj]
    # x1 = ii1_list[0] = 3
    f11, f21, f31 = z_mat[0,:]
    # x1 = ii1_list[1] = 10
    f12, f22, f32 = z_mat[1,:]
    # x1 = ii1_list[2] = 14
    f13, f23, f33 = z_mat[2,:]
    
    x_target = 1.4
    y_target = 1.2
    f_predict = interQuaIAST2D(x_target, y_target, x_pick, y_pick,
                               f11,f12,f13,f21,f22,f23,f31,f32,f33)
    x_me, y_me = np.meshgrid(x_ran, y_ran)
    fig, ax = plt.subplots(subplot_kw={'projection':'3d'})
    surf =ax.plot_surface(x_me, y_me, z_ran)
    ax.plot3D(x_target,y_target, f_predict, 'ro')
    ax.view_init(-15, -65)
    