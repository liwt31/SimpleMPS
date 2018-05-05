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


def construct_single_mpo(j=1., h=1.):
    """
    Construct single site mpo based on transverse field Ising model
    :param j: coupling constant
    :param h: Strength of external field
    :return: constructed single site mpo with shape (3, 3, 2, 2)
    """
    # MPO line by line
    mpo_block1 = [S1,      S0,      S0]
    mpo_block2 = [Sz,      S0,      S0]
    mpo_block3 = [-h * Sx, -j * Sz, S1]

    # MPO shape: 3, 3, 2, 2
    mpo = np.float64([mpo_block1, mpo_block2, mpo_block3])
    # prohibit writing just in case
    mpo.flags.writeable = False
    return mpo


def construct_mz_mpo_list(site_num):
    """
    Construct the $M_z$ MPO list needed for calculating the expectation value of $M_z$ for a MPS.
    :param site_num: number of sites
    :return: $M_z$ mpo list
    """
    mpo_list = build_mpo_list(np.float64([[S1, S0], [Sz, S1]]), site_num, regularize=True)
    return mpo_list


def construct_mx_mpo_list(site_num):
    """
    Construct the $M_x$ MPO list needed for calculating the expectation value of $M_x$ for a MPS
    :param site_num: number of sites
    :return: $M_x$ mpo list
    """
    mpo_list = build_mpo_list(np.float64([[S1, S0], [Sx, S1]]), site_num, regularize=True)
    return mpo_list


# number of sites
SITE_NUM = 20

# only one of the following 2 parameters are needed

# maximum bond order in MPS
BOND_ORDER = 16

# the threshold for error when compressing MPS
ERROR_THRESHOLD = 1e-7


def calc_phase_diagram():
    """
    Calculating the phase diagram of transverse field Ising model in $h$ and $J$ grid.
    $h$ and $J$ both range from -100 to 100. The result is saved in an npz file.
    :return: None.
    """
    logspace = list(np.logspace(-2, 2, 20))
    minus_logspace = list(-np.logspace(-2, 2, 20)[::-1])
    j_grid = minus_logspace + [0] + logspace
    h_grid = minus_logspace + [0] + logspace
    mz_result = np.zeros((len(j_grid), len(h_grid)))
    mx_result = np.zeros((len(j_grid), len(h_grid)))
    for j_idx, j in enumerate(j_grid):
        for h_idx, h in enumerate(h_grid):
            param_str = 'j = %g, h = %g' % (j, h)
            logging.info('Start: ' + param_str)
            mpo_list = build_mpo_list(construct_single_mpo(j, h), SITE_NUM)
            mps = MatrixProductState(mpo_list, error_threshold=ERROR_THRESHOLD)
            energies = mps.search_ground_state()
            mz = mps.expectation(construct_mz_mpo_list(SITE_NUM))
            mx = mps.expectation(construct_mx_mpo_list(SITE_NUM))
            mz_result[j_idx][h_idx] = mz
            mx_result[j_idx][h_idx] = mx
            logging.info('End %s. Energy: %g, M_z: %g, M_x: %g' % (param_str, energies[-1], mz, mx))
    np.savez('ising_phase_diagram', mz_result=mz_result, mx_result=mx_result, j_grid=j_grid, h_grid=h_grid)


def calc_mx_vs_h():
    """
    Calculate the relation between $M_x$ and $h$ when $J$ is fixed to 1 and $h$ varies from 0 to 2
    to study the phase transition. Energies are also calculated as a reference.
    The result is saved in an npz file.
    :return: None
    """
    h_list = np.linspace(0, 2, 20)
    energy_list = []
    mx_list = []
    for h in h_list:
        logging.info('Start: h = %g' % h)
        mpo_list = build_mpo_list(construct_single_mpo(j=1, h=h), SITE_NUM)
        mps = MatrixProductState(mpo_list, error_threshold=ERROR_THRESHOLD)
        energies = mps.search_ground_state()
        mx = mps.expectation(construct_mx_mpo_list(SITE_NUM))
        average_energy = energies[-1] / SITE_NUM
        energy_list.append(average_energy)
        mx_list.append(mx)
        logging.info('End h = %g. Energy: %g, M_x: %g' % (h, average_energy, mx))
    np.savez('ising_mx_vs_h', h_list=h_list, energy_list=energy_list, mx_list=mx_list)


if __name__ == '__main__':
    #calc_mx_vs_h()
    calc_phase_diagram()

