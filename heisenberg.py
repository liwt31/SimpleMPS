# SimpleMPS
# Density matrix renormalization group (DMRG) in matrix product state (MPS)

# This file contains the definition of matrix product operators in a Heisenberg model.
# For theoretical backgrounds, see the [reference]:
# Ulrich Schollwöck, The density-matrix renormalization group in the age of matrix product states,
# Annals of Physics, 326 (2011), 96-192

import numpy as np

# Pauli matrices

Sp = np.complex128([[0, 1], [0, 0]])
Sm = np.complex128([[0, 0], [1, 0]])

Sx = np.complex128([[0, .5], [.5, 0]])
Sy = np.complex128([[0, -.5j], [.5j, 0]])
Sz = np.complex128([[.5, 0], [0, -.5]])

S0 = np.zeros((2, 2), dtype=np.complex128)
S1 = np.eye((2), dtype=np.complex128)

# coupling constant
J = Jz = 1

# Planck constant
h = 1

# MPO line by line
_W1 = [S1,    S0,     S0,     S0,    S0]
_W2 = [Sp,    S0,     S0,     S0,    S0]
_W3 = [Sm,    S0,     S0,     S0,    S0]
_W4 = [Sz,    S0,     S0,     S0,    S0]
_W5 = [-h*Sz, J/2*Sm, J/2*Sp, Jz*Sz, S1]

# W shape: 5, 5, 2, 2
W = np.complex128([_W1, _W2, _W3, _W4, _W5]).real
# prohibit writing for safety concerns
W.flags.writeable = False
# the first MPO, only contains the last row
W_1 = W[-1]
W_1 = W_1.reshape((1,) + W_1.shape)
# the last MPO, only contains the first column
W_L = W[:, 0]
W_L = W_L.reshape((W_L.shape[0],) + (1,) + W_L.shape[1:])


def build_W_list(SITE_NUM):
    """
    build MPO list for MPS.
    :param SITE_NUM: the total number of sites
    :return MPO list
    """
    return [W_1] + [W] * (SITE_NUM - 2) + [W_L]

