
from IASTrigCal import PredLinIAST2D, genIASTdata2D
import os
import numpy as np
# %% 
# Generate data if does not exist
# %%
# Generate IAST data (Lookup table)
# %% 
f_nam = "IAST_da.pkl"
if not os.path.exists(f_nam):
    qm1 = 4
    b1 = 0.2
    qm2 = 1.5
    b2 = 0.5

    def iso1(P):
        numer = qm1*b1*P
        denom = 1+b1*P
        q_ret = numer/denom
        return q_ret

    def iso2(P):
        numer = qm2*b2*P
        denom = 1+b2*P
        q_ret = numer/denom
        return q_ret
    #y1_dom = np.linspace(0,1,101)
    #P_dom_low = np.linspace(0,1,26)
    #P_dom_high = np.linspace(1,30,60)
    y1_dom = np.linspace(0,1,51)
    P_dom_low = np.linspace(0,1,21)
    P_dom_high = np.linspace(1,30,30)
    P_dom = np.concatenate([P_dom_low, P_dom_high[1:]])
    genIASTdata2D(iso1, iso2, y1_dom, P_dom, file_name=f_nam)

# %%
# Interpolation model (Lokuptable)
q_inter = PredLinIAST2D(f_nam)

# %%
# DO NOT CHANGE THE FUCNTION NAME! (isomix)
# DO NOT CHANGE THE FUCNTION NAME! (isomix)
# DO NOT CHANGE THE FUCNTION NAME! (isomix)

def iso_mix(P1, P2):
    P_ov = P1+P2+1E-7
    y1 = P1/P_ov
    q1, q2 = q_inter.predict(y1, P_ov)
    return q1, q2

f_IAST = lambda p1, p2: np.array(list(map(iso_mix, p1, p2)))
