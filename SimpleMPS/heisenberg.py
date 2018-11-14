# -*- encoding: utf-8

# SimpleMPS
# Density matrix renormalization group (DMRG) in matrix product state (MPS)

# This file contains the definition of matrix product operators in Heisenberg model,
# then performs ground state search based on MPS.
# For theoretical backgrounds, see the [reference]:
# Ulrich Schollwöck, The density-matrix renormalization group in the age of matrix product states,
# Annals of Physics, 326 (2011), 96-192

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from mps import MatrixProductState, build_mpo_list

# import pauli matrices
from paulimat import *


def construct_single_mpo(J=1, Jz=1, h=1):
    """
    Construct single site mpo based on Heisenberg model
    :param J: coupling constant
    :param Jz: coupling constant
    :param h: strength of external field
    :return: constructed single site mpo with shape (5, 5, 2, 2)
    """
    # MPO line by line
    mpo_block1 = [S1, S0, S0, S0, S0]
    mpo_block2 = [Sp, S0, S0, S0, S0]
    mpo_block3 = [Sm, S0, S0, S0, S0]
    mpo_block4 = [Sz, S0, S0, S0, S0]
    mpo_block5 = [-h * Sz, J / 2 * Sm, J / 2 * Sp, Jz * Sz, S1]

    # MPO shape: 5, 5, 2, 2
    mpo = np.float64([mpo_block1, mpo_block2, mpo_block3, mpo_block4, mpo_block5])
    # prohibit writing just in case
    mpo.flags.writeable = False

    return mpo


# number of sites
SITE_NUM = 20

# only one of the following 2 parameters are needed

# maximum bond order in MPS
MAX_BOND_ORDER = 16

# the threshold for error when compressing MPS
ERROR_THRESHOLD = 1e-7

if __name__ == "__main__":
    mpo_list = build_mpo_list(construct_single_mpo(), SITE_NUM)
    mps = MatrixProductState(mpo_list, max_bond_order=MAX_BOND_ORDER)
    # use threshold as criterion for compression
    # mps = MatrixProductState(mpo_list, error_threshold=ERROR_THRESHOLD)
    print(mps.search_ground_state())
