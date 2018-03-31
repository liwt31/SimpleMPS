# SimpleMPS
Matrix product state (MPS) based density matrix renormalization group (DMRG) to solve Heisenberg model: a Python implementation.

---
# What is SimpleMPS
Density matrix renormalization group (DMRG) is a powerful method to simulate one-dimensional strongly correlated quantum systems. During recent years, combining DMRG with matrix product state (MPS) has allowed further understanding of the DMRG method. SimpleMPS aims to provide a demo for MPS based DMRG to solve the [Heisenberg model](https://en.wikipedia.org/wiki/Heisenberg_model_(quantum)). The ground state of the Heisenberg model could be obtained iteratively (see the image below, h=1, J=Jz=1, 20 sites).

![energy profile](energy.jpg)

The implementation is largely inspired by [The density-matrix renormalization group in the age of matrix product states](https://arxiv.org/abs/1008.3477v2). Understanding the first 6 parts of the article is crucial to understanding the code. 
# Files
* `mps.py` Implemented the MPS based DMRG method
* `heisenberg.py` The hamitonian and matrix product operator of the Heisenberg model
* `optimize_heisenberg.py` Run the optimization. 

# How to use
* `python optimize_heisenberg.py` to see the output of the energy during each iteration.
  * There are 2 parameters hard-coded in `optimize_heisenberg.py`: `SITE_NUM` which is the number of sites in the model and `BOND_ORDER` which is the order of the bond degree in matrix product state (the higher bond order, the higher accuracy and compuational cost).
* Modify the code and do anything you want!