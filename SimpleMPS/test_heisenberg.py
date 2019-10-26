# -*- coding: utf-8 -*-

import pytest

from mps import build_mpo_list, MatrixProductState
from heisenberg import construct_single_mpo, SITE_NUM, MAX_BOND_DIMENSION


def test_search():
    mpo_list = build_mpo_list(construct_single_mpo(), SITE_NUM)
    mps = MatrixProductState(mpo_list, max_bond_dimension=MAX_BOND_DIMENSION)
    # use threshold as criterion for compression
    # mps = MatrixProductState(mpo_list, error_threshold=ERROR_THRESHOLD)
    energies = mps.search_ground_state()

    assert pytest.approx(energies[-1], -10, abs=0.1)