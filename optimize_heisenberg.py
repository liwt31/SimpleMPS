# SimpleMPS
# Density matrix renormalization group (DMRG) in matrix product state (MPS)

# This file contains the definition of matrix product state
# For theoretical backgrounds, see the [reference]:
# Ulrich Schollwöck, The density-matrix renormalization group in the age of matrix product states,
# Annals of Physics, 326 (2011), 96-192

from mps import TensorProductState
from heisenberg import build_W_list

# number of sites
SITE_NUM = 20

BOND_ORDER = 16

TensorProductState(BOND_ORDER, build_W_list(SITE_NUM)).optimize()
