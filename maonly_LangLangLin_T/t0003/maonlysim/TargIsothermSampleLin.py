
from IASTrigCal import PredQuaIAST3D, genIASTdata3D
import os
import numpy as np
# %% 
# Generate data if does not exist
# %%
# Generate IAST data (Lookup table)
# %% 
f_nam = "IAST_da_3D.pkl"
if not os.path.exists(f_nam):
    qm1 = 4
    b1 = 0.2
    qm2 = 1.5
    b2 = 0.5
    dH1 = 15e3
    dH2 = 4e3
    T_ref = 300
    R_gas = 8.3145
    def iso1(P,T):
        P_norm = P*np.exp(dH1/R_gas*(1/T - 1/T_ref))
        numer = qm1*b1*P_norm
        denom = 1+b1*P_norm
        q_ret = numer/denom
        return q_ret

    def iso2(P,T):
        P_norm = P*np.exp(dH2/R_gas*(1/T - 1/T_ref))
        numer = qm2*b2*P_norm
        denom = 1+b2*P_norm
        q_ret = numer/denom
        return q_ret

    y1_dom = np.linspace(0,1,101)
    P_dom_low = np.linspace(0,1,26)
    P_dom_high = np.linspace(1,30,60)
    P_dom = np.concatenate([P_dom_low, P_dom_high[1:]])
    T_dom = np.linspace(283,383, 21)
    # Lookup Table with temperature
    genIASTdata3D(iso1, iso2, y1_dom, P_dom,T_dom, file_name=f_nam)
# %%
# Interpolation model (Lokuptable)

q_inter = PredQuaIAST3D(f_nam)

# %%
# DO NOT CHANGE THE FUCNTION NAME! (isomix)
# DO NOT CHANGE THE FUCNTION NAME! (isomix)
# DO NOT CHANGE THE FUCNTION NAME! (isomix)

def iso_mix(P1, P2, T):
    P_ov = P1+P2+1E-5
    y1 = P1/P_ov
    q1, q2 = q_inter.predict(y1, P_ov, T)
    return q1, q2

f_IAST = lambda p1, p2, T: np.array(list(map(iso_mix, p1, p2, T)))
