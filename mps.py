# -*- encoding: utf-8

# SimpleMPS
# Density matrix renormalization group (DMRG) in matrix product state (MPS)

# This file contains the definition of matrix product state
# For theoretical backgrounds, see the [reference]:
# Ulrich Schollwöck, The density-matrix renormalization group in the age of matrix product states,
# Annals of Physics, 326 (2011), 96-192

# The implementation is based on doubly linked list

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from functools import reduce

import numpy as np
from numpy.linalg import svd
from scipy.sparse.linalg import eigs


class MatrixState(object):
    """
    Matrix state for a single site. A 3-degree tensor with 2 bond degrees to other matrix states and a physical degree.
    A matrix product operator (MPO) is also included in the matrix state.
    A sentinel matrix state could be initialized for an imaginary state
    which provides convenience for doubly linked list implementation.
    """
    def __init__(self, bond_order1, bond_order2, mpo):
        """
        Initialize a matrix state with shape (bond_order1, phys_d, bond_order2) and an MPO attached to the state.
        The MPO is usually the Hamiltonian.
        If a sentinel `MatrixState` is required, set bond_order1, phys_d or bond_order2 to 0 or None.
        MPO should be a 4-degree tensor with 2 bond degrees at first and 2 physical degrees at last.
        """
        # if is a sentinel matrix state
        if not (bond_order1 and bond_order2):
            self._matrix = self.mpo = np.ones((0, 0, 0))
            self.left_ts = self.right_ts = None
            self.F_cache = self.L_cache = self.R_cache = np.ones((1,) * 6)
            self.is_sentinel = True
            return
        self.is_sentinel = False
        phys_d = mpo.shape[2]
        # random initialization of the state tensor
        self._matrix = np.random.random((bond_order1, phys_d, bond_order2))
        self.mpo = mpo
        # the pointer to the matrix state on the left
        self.left_ts = None
        # the pointer to the matrix state on the right
        self.right_ts = None
        # cache for F, L and R to accelerate calculations
        # for the definition of these parameters, see the [reference]: Annals of Physics, 326 (2011), 145-146
        # because of the cache, any modifications to self._matrix should be properly wrapped.
        # modifying self._matrix directly may lead to unexpected results.
        self.F_cache = None
        self.L_cache = self.R_cache = None

    @property
    def matrix(self):
        return self._matrix

    @matrix.setter
    def matrix(self, new_matrix):
        # bond order may have reduced due to low local degree of freedom
        # but the order of the physical degree must not change
        assert self.phys_d == new_matrix.shape[1]
        self._matrix = new_matrix
        # forbid writing for safety concerns
        self._matrix.flags.writeable = False
        # disable the cache for F, L, R
        self.clear_cache()

    @property
    def bond_order1(self):
        """
        :return: the order of the first bond degree
        """
        return self.matrix.shape[0]

    @property
    def phys_d(self):
        """
        :return: the order fo the physical index
        """
        assert self.matrix.shape[1] == self.mpo.shape[2] == self.mpo.shape[3]
        return self.matrix.shape[1]

    @property
    def bond_order2(self):
        """
        :return: the order of the second bond degree
        """
        return self.matrix.shape[2]

    def left_canonicalize(self):
        """
        Perform left canonical decomposition on this site
        """
        if not self.right_ts:
            return
        u, s, v = svd(self.matrix.reshape(self.bond_order1 * self.phys_d, self.bond_order2), full_matrices=False)
        self.matrix = u.reshape((self.bond_order1, self.phys_d, -1))
        self.right_ts.matrix = np.tensordot(np.dot(np.diag(s), v), self.right_ts.matrix, axes=[1, 0])

    def left_canonicalize_all(self):
        """
        Perform left canonical decomposition on this site and all sites on the right
        """
        if not self.right_ts:
            return
        self.left_canonicalize()
        self.right_ts.left_canonicalize_all()

    def right_canonicalize(self):
        """
        Perform right canonical decomposition on this site
        """
        if not self.left_ts:
            return
        u, s, v = svd(self.matrix.reshape(self.bond_order1, self.phys_d * self.bond_order2), full_matrices=False)
        self.matrix = v.reshape((-1, self.phys_d, self.bond_order2))
        self.left_ts.matrix = np.tensordot(self.left_ts.matrix, np.dot(u, np.diag(s)), axes=[2, 0])

    def right_canonicalize_all(self):
        """
        Perform right canonical decomposition on this site and all sites on the left
        """
        if not self.left_ts:
            return
        self.right_canonicalize()
        self.left_ts.right_canonicalize_all()

    def test_left_unitary(self):
        """
        Helper function to test if this site is left normalized
        Only for test. Not used in release version
        """
        m = self.matrix
        summation = sum([np.dot(m[:, i, :].transpose().conj(), m[:, i, :]) for i in range(self.phys_d)])
        print('Test left unitary: %s' % np.allclose(summation, np.eye(self.bond_order2)))

    def test_right_unitary(self):
        """
        Helper function to test if this site is right normalized
        Only for test. Not used in release version
        """
        m = self.matrix
        summation = sum([np.dot(m[:, i, :], m[:, i, :].transpose().conj()) for i in range(self.phys_d)])
        print('Test right unitary: %s' % np.allclose(summation, np.eye(self.bond_order1)))

    def calc_F(self, mpo=None):
        """
        calculate F for this site.
        graphical representation (* for MPS and # for MPO,
        numbers represents a set of imaginary bond orders used for comments below):
                                  1 --*-- 5
                                      | 4
                                  2 --#-- 3
                                      | 4
                                  1 --*-- 5
        :parameter mpo: an external MPO to calculate. Used in expectation calculation.
        :return the calculated F
        """
        # whether use self.mpo or external MPO
        use_self_mpo = mpo is None
        if use_self_mpo:
            mpo = self.mpo
        # return cache immediately if the value has been calculated before and self.matrix has never changed
        if use_self_mpo and self.F_cache is not None:
            return self.F_cache
        # Do the contraction from top to bottom.
        # suppose self.matrix.shape = 1,4,5, self.mpo.shape = 2,3,4,4 (left, right, up, down)
        # up_middle is of shape (1, 5, 2, 3, 4)
        up_middle = np.tensordot(self.matrix.conj(), mpo, axes=[1, 2])
        # return value F is of shape (1, 5, 2, 3, 1, 5). In the graphical representation,
        # the position of the degrees of the tensor is from top to bottom and left to right
        F = np.tensordot(up_middle, self.matrix, axes=[4, 1])
        if use_self_mpo:
            pass
            self.F_cache = F
        return F

    def calc_L(self):
        """
        calculate L in a recursive way
        """
        # the left state is a sentinel, return F directly.
        if not self.left_ts:
            return self.calc_F()
        # return cache immediately if available
        if self.L_cache is not None:
            return self.L_cache
        # find L from the state on the left
        last_L = self.left_ts.calc_L()
        # calculate F in this state
        F = self.calc_F()
        """
        do the contraction. Note the sequence of the indexes in self.calc_F.
        graphical representation (* for MPS and # for MPO, numbers represents the index of the degree in tensor.shape):
          0 --*-- 1          0 --*-- 1                   0 --*-- 3                   0 --*-- 1          
              |                  |                           |                           |     
          2 --#-- 3     +    2 --#-- 3  --tensordot-->   1 --#-- 4    --reshape-->   2 --#-- 3                
              |                  |                           |                           |     
          4 --*-- 5          4 --*-- 5                   2 --*-- 5                   4 --*-- 5          
        
        """
        L = np.tensordot(last_L, F, axes=[[1, 3, 5], [0, 2, 4]]).transpose((0, 3, 1, 4, 2, 5))
        self.L_cache = L
        return L

    def calc_R(self):
        """
        calculate R in a recursive way
        """
        # mirror to self.calc_L. Explanation omitted.
        if not self.right_ts:
            return self.calc_F()
        if self.R_cache is not None:
            return self.R_cache
        last_R = self.right_ts.calc_R()
        F = self.calc_F()
        R = np.tensordot(F, last_R, axes=[[1, 3, 5], [0, 2, 4]]).transpose((0, 3, 1, 4, 2, 5))
        self.R_cache = R
        return R

    def clear_cache(self):
        """
        clear cache for F, L and R when self.matrix has changed
        """
        self.F_cache = None
        # clear R cache for all matrix state on the left because their R involves self.matrix
        self.left_ts.clear_R_cache()
        # clear L cache for all matrix state on the right because their L involves self.matrix
        self.right_ts.clear_L_cache()

    def clear_L_cache(self):
        """
        clear all cache for L in matrix states on the right in a recursive way
        """
        # stop recursion if the end of the MPS is met
        if self.L_cache is None or not self:
            return
        self.L_cache = None
        self.right_ts.clear_L_cache()

    def clear_R_cache(self):
        """
        clear all cache for R in matrix states on the left in a recursive way
        """
        # stop recursion if the end of the MPS is met
        if self.R_cache is None or not self:
            return
        self.R_cache = None
        self.left_ts.clear_R_cache()

    def calc_variational_tensor(self):
        """
        calculate the variational tensor for the ground state search. L * MPO * R
        graphical representation (* for MPS and # for MPO):
                                   --*--     --*--
                                     |         |
                                   --#----#----#--
                                     |         |
                                   --*--     --*--
                                     L   MPO   R
        """
        """
        do the contraction for L and MPO
        graphical representation (* for MPS and # for MPO, numbers represents the index of the degree in tensor.shape):
          0 --*-- 1                                    0 --*-- 1                
              |                | 2                         |    | 6      
          2 --#-- 3    +   0 --#-- 1  --tensordot-->   2 --#----#-- 5                 
              |                | 3                         |    | 7      
          4 --*-- 5                                    3 --*-- 4                
              L                MPO                       left_middle
        """
        left_middle = np.tensordot(self.left_ts.calc_L(), self.mpo, axes=[3, 0])
        """
        do the contraction for L and MPO
        graphical representation (* for MPS and # for MPO, numbers represents the index of the degree in tensor.shape):
          0 --*-- 1             0 --*-- 1                   0 --*-- 1 8 --*-- 9      
              |    | 6              |                           |    | 6  |   
          2 --#----#-- 5   +    2 --#-- 3  --tensordot-->   2 --#----#----#-- 10       
              |    | 7              |                           |    | 7  |   
          3 --*-- 4             4 --*-- 5                   3 --*-- 4 11--*-- 12 
            left_middle             R                       raw variational tensor
        Note the order of 0, 2, 3, 9, 10, 12 are all 1, so the dimension could be reduced
        """
        raw_variational_tensor = np.tensordot(left_middle, self.right_ts.calc_R(), axes=[5, 2])
        shape = (self.bond_order1, self.bond_order1, self.phys_d, self.phys_d, self.bond_order2, self.bond_order2)
        # reduce the dimension and rearrange the degrees to 1, 8, 6, 4, 11, 7 in the above graphical representation
        return raw_variational_tensor.reshape(shape).transpose((0, 2, 4, 1, 3, 5))

    def variational_update(self, direction):
        """
        Update the matrix of this state to search ground state by variation method
        :param direction: the direction to update. 'right' means from left to right and 'left' means from right to left
        :return the energy of the updated state.
        """
        assert direction == 'left' or direction == 'right'
        dim = self.bond_order1 * self.phys_d * self.bond_order2
        # reshape variational tensor to a square matrix
        variational_tensor = self.calc_variational_tensor().reshape(dim, dim)
        # find the smallest eigenvalue and eigenvector. Note the value returned by `eigs` are complex numbers
        eig_val, eig_vec = eigs(variational_tensor, 1, which='SR')
        # reshape the eigenvector back to a matrix state
        self.matrix = eig_vec.real.reshape(self.bond_order1, self.phys_d, self.bond_order2)
        # perform normalization
        if direction == 'right':
            self.left_canonicalize()
        if direction == 'left':
            self.right_canonicalize()
        print('Ground state energy: %.10f' % eig_val.real)
        return eig_val.real

    def insert_ts_before(self, ts):
        """
        insert a matrix state before this matrix state. Standard doubly linked list operation.
        """
        left_ts = self.left_ts
        left_ts.right_ts = ts
        ts.left_ts, ts.right_ts = left_ts, self
        self.left_ts = ts

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        return 'MatrixState (%d, %d, %d)' % (self.bond_order1, self.phys_d, self.bond_order2)

    def __nonzero__(self):
        return self.__bool__()

    def __bool__(self):
        """
        :return: True if this state is not a sentinel state and vice versa.
        """
        return not self.is_sentinel


class MatrixProductState(object):
    """
    A doubly linked list of `MatrixState`. The matrix product state of the whole wave function.
    """
    def __init__(self, bond_order, W_list):
        """
        Initialize a MatrixProductState with given bond order.
        :param bond_order: the bond order required. The higher bond order, the higher accuracy and compuational cost
        :param W_list: the list for MPOs. The site num depends on the length of the list
        """
        self.bond_order = bond_order
        self.site_num = len(W_list)
        self.W_list = W_list
        # establish the sentinels for the doubly linked list
        self.tensor_state_head = MatrixState(0, 0, 0)
        self.tensor_state_tail = MatrixState(0, 0, 0)
        self.tensor_state_head.right_ts = self.tensor_state_tail
        self.tensor_state_tail.left_ts = self.tensor_state_head
        # initialize the matrix states with random numbers.
        M_list = [MatrixState(1, bond_order, W_list[0])] + \
                 [MatrixState(bond_order, bond_order, W_list[i + 1]) for i in range(self.site_num - 2)] + \
                 [MatrixState(bond_order, 1, W_list[-1])]
        # insert matrix states to the doubly linked list
        for ts in M_list:
            self.tensor_state_tail.insert_ts_before(ts)
        # perform the initial normalization
        self.tensor_state_head.right_ts.left_canonicalize_all()
        # test for the unitarity
        #for ts in self.iter_ts_left2right():
        #    ts.test_left_unitary()

    def iter_ts_left2right(self):
        """
        matrix state iterator. From left to right
        """
        ts = self.tensor_state_head.right_ts
        while ts:
            yield ts
            ts = ts.right_ts
        raise StopIteration

    def iter_ts_right2left(self):
        """
        matrix state iterator. From right to left
        """
        ts = self.tensor_state_tail.left_ts
        while ts:
            yield ts
            ts = ts.left_ts
        raise StopIteration

    def optimize(self):
        """
        optimize the energy of the MPS by variation method
        :return the energies of each step during the optimization
        """
        energies = []
        # stop when the energies does not change anymore
        while len(energies) < 2 or not np.isclose(energies[-1], energies[-2]):
            for ts in self.iter_ts_right2left():
                energies.append(ts.variational_update('left'))
            for ts in self.iter_ts_left2right():
                energies.append(ts.variational_update('right'))
        return energies

    def expectation(self, W_list):
        F_list = [ts.calc_F(mpo) for mpo, ts in zip(W_list, self.iter_ts_left2right())]

        def contractor(tensor1, tensor2):
            return np.tensordot(tensor1, tensor2, axes=[[1, 3, 5], [0, 2, 4]]).transpose((0, 3, 1, 4, 2, 5))
        expectation = reduce(contractor, F_list).reshape(1)
        return expectation[0]


def build_mpo_list(single_mpo, site_num):
    """
    build MPO list for MPS.
    :param single_mpo: a numpy ndarray with ndim=4.
    The first 2 dimensions reprsents the square shape of the MPO and the last 2 dimensions are physical dimensions.
    :param site_num: the total number of sites
    :return MPO list
    """
    argument_error = ValueError("The definition of MPO is incorrect. Datatype: %s, shape: %s." 
                                "Please make sure it's a numpy array and check the dimensions of the MPO."
                                % (type(single_mpo), single_mpo.shape))
    if not isinstance(single_mpo, np.ndarray):
        raise argument_error
    if single_mpo.ndim != 4:
        raise argument_error
    if single_mpo.shape[2] != single_mpo.shape[3]:
        raise argument_error
    if single_mpo.shape[0] != single_mpo.shape[1]:
        raise argument_error
    # the first MPO, only contains the last row
    mpo_1 = single_mpo[-1]
    mpo_1 = mpo_1.reshape((1, ) + mpo_1.shape)
    # the last MPO, only contains the first column
    mpo_L = single_mpo[:, 0]
    mpo_L = mpo_L.reshape((mpo_L.shape[0], ) + (1, ) + mpo_L.shape[1:])
    return [mpo_1] + [single_mpo] * (site_num - 2) + [mpo_L]