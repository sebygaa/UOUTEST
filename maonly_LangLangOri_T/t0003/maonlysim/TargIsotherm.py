
from IASTrigCal import IAST_bi
import os
import numpy as np
# %% 
# Generate data if does not exist
# %%
# Generate IAST data (Lookup table)
# %% 
def Lang(P, iso_params):
    numer = iso_params[0]*iso_params[1]*P
    denom =  1 + iso_params[1]*P
    return numer / denom

iso_par1 = [3,1]
iso_par2 = [1,0.5]

Lang1 = lambda P: Lang(P, iso_par1)
Lang2 = lambda P: Lang(P, iso_par2)

def iso_mix(P1, P2):
    P_ov = P1 + P2
    y1 = P1/P_ov
    [q1,q2], fval = IAST_bi(Lang1, Lang2, y1, P_ov)
    return q1, q2
# %%
# DO NOT CHANGE THE FUCNTION NAME! (isomix)
# DO NOT CHANGE THE FUCNTION NAME! (isomix)
# DO NOT CHANGE THE FUCNTION NAME! (isomix)
f_IAST = lambda p1, p2: np.array(list(map(iso_mix, p1, p2)))