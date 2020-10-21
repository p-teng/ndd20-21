import numpy as np
from math import log2, ceil

from joblib import Parallel, delayed

from proglearn.sims import *
import functions.xor_nxor_functions as fn

X, Y = generate_gaussian_parity(750, angle_params=0)
Z, W = generate_gaussian_parity(750, angle_params=np.pi/2)

# plot and format:
fn.plot_xor_nxor(X, Y, 'Gaussian XOR')
fn.plot_xor_nxor(Z, W, 'Gaussian N-XOR')