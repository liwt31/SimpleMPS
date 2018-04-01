# -*- encoding: utf-8

# SimpleMPS
# Density matrix renormalization group (DMRG) in matrix product state (MPS)

# This file contains the definition of matrix product operators in Ising model,
# then performs ground state search based on MPS.
# For theoretical backgrounds, see the [reference]:
# Ulrich Schollwöck, The density-matrix renormalization group in the age of matrix product states,
# Annals of Physics, 326 (2011), 96-192

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)

from mps import MatrixProductState, build_mpo_list

# import pauli matrices
from paulimat import *

def construct_single_mpo(J=1, h=1):
    """
    Construct single site mpo based on Heisenberg model
    :param J: coupling constant
    :param h: Strength of external field
    :return: constructed single site mpo with shape (3, 3, 2, 2)
    """
    # MPO line by line
    mpo_block1 = [S1,    S0,     S0]
    mpo_block2 = [Sz,    S0,     S0]
    mpo_block3 = [-h*Sx, -J*Sz, S1]

    # MPO shape: 3, 3, 2, 2
    mpo = np.float64([mpo_block1, mpo_block2, mpo_block3])
    # prohibit writing just in case
    mpo.flags.writeable = False
    return mpo


# number of sites
SITE_NUM = 20

# only one of the following 2 parameters are needed

# maximum bond order in MPS
BOND_ORDER = 16

# the threshold for error when compressing MPS
ERROR_THRESHOLD = 1e-7

# study phase diagram in a j and h grid
LOGSPACE = list(np.logspace(-2, 2, 20))
MINUS_LOGSPACE = list(-np.logspace(-2, 2, 20)[::-1])
J_GRID = MINUS_LOGSPACE + [0] + LOGSPACE
H_GRID = MINUS_LOGSPACE + [0] + LOGSPACE


if __name__ == '__main__':
    mz_result = np.zeros((len(J_GRID), len(H_GRID)))
    mx_result = np.zeros((len(J_GRID), len(H_GRID)))
    for j_idx, j in enumerate(J_GRID):
        for h_idx, h in enumerate(H_GRID):
            param_str = 'j=%g, h=%g' % (j, h)
            logging.info('Start: ' + param_str)
            mpo_list = build_mpo_list(construct_single_mpo(j, h), SITE_NUM)
            mps = MatrixProductState(mpo_list, error_threshold=ERROR_THRESHOLD)
            energies = mps.search_ground_state()
            # The MPOs are not regularized by `SITE_NUM`
            # Resulting mz and mx should be divided by `SITE_NUM`
            mz = mps.expectation(build_mpo_list(np.float64([[S1, S0], [Sz, S1]]), SITE_NUM))
            mx = mps.expectation(build_mpo_list(np.float64([[S1, S0], [Sx, S1]]), SITE_NUM))
            mz_result[j_idx][h_idx] = mz
            mx_result[j_idx][h_idx] = mx
            logging.info('End %s. Energy: %g, Mz: %g, Mx: %g' % (param_str, energies[-1], mz, mx))
    np.savez('ising', mz_result=mz_result, mx_result=mx_result, j_grid=J_GRID, h_grid=H_GRID)


