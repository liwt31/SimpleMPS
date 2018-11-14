# -*- encoding: utf-8

# SimpleMPS
# Density matrix renormalization group (DMRG) in matrix product state (MPS)

# This file contains the definition of pauli matrices
# For theoretical backgrounds, see the [reference]:
# Ulrich Schollwöck, The density-matrix renormalization group in the age of matrix product states,
# Annals of Physics, 326 (2011), 96-192

import numpy as np

# Pauli matrices

# S^+
Sp = np.float64([[0, 1], [0, 0]])
# S^-
Sm = np.float64([[0, 0], [1, 0]])

Sx = np.float64([[0, 0.5], [0.5, 0]])
Sz = np.float64([[0.5, 0], [0, -0.5]])

# zero matrix block
S0 = np.zeros((2, 2))
# identity matrix block
S1 = np.eye((2))
