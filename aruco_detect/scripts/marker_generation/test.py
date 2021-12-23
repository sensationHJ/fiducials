import numpy as np
from func_dh_table import Tmat


aa = np.array([[ -np.pi/2, 0 , 0, -np.pi/2],[0, 1.77, 0, 0.08]])

Tmat(aa)