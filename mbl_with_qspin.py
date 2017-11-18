#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 17 18:30:48 2017

@author: Alexandre Dauphin
"""

import numpy as np
import quspin
from quspin.operators import hamiltonian # Hamiltonians and operators
from quspin.basis import spin_basis_1d # Hilbert space spin basis
from numpy.random import ranf,seed # pseudo random numbers
import numpy as np # generic math functions
from matplotlib import pyplot as plt
from tqdm import tqdm

def ratio(mat):
    emin,emax=mat.eigsh(k=2,which="BE",maxiter=1E4,return_eigenvectors=False)
    e=(emin+emax)/2
    kk=10
    val=H_XXZ.eigsh(k=kk+2,sigma=e,maxiter=1E4,return_eigenvectors=False)
    val=np.sort(val)
    r=0
    for i in np.arange(1,kk+1):
        delta_n = val[i]-val[i-1]
        delta_n1 = val[i+1]-val[i]
        r=r+min(delta_n, delta_n1)/max(delta_n, delta_n1)
    r=r/kk
    return r

##### define model parameters #####
L=12 # system size
Jxy=1.0#np.sqrt(2.0) # xy interaction
Jzz_0=1.0 # zz interaction
hz=0.#1.0/np.sqrt(3.0) # z external field

basis = spin_basis_1d(L,pauli=False,Nup=L//2) # zero magnetisation sector

# define operators with OBC using site-coupling lists
J_zz = [[Jzz_0,i,i+1] for i in np.arange(L-1)] # OBC
J_xy = [[Jxy/2.0,i,i+1] for i in np.arange(L-1)] # OBC

#For PBC
#J_zz.append([Jzz_0,L-1,0])
#J_xy.append([Jxy/2.0,L-1,0])

# static and dynamic lists
static = [["+-",J_xy],["-+",J_xy],["zz",J_zz]]
dynamic=[]

# compute the time-dependent Heisenberg Hamiltonian
H_XXZ = hamiltonian(static,dynamic,basis=basis,dtype=np.float64)

# compute disordered z-field Hamiltonian
no_checks={"check_herm":False,"check_pcon":False,"check_symm":False}

hmbl=0.
unscaled_fields=-1+2*ranf((basis.L,))
h_z=[[unscaled_fields[i],i] for i in range(basis.L)]
disorder_field = [["z",h_z]]
Hz=hamiltonian(disorder_field,[],basis=basis,dtype=np.float64,**no_checks)
H_MBL=H_XXZ+hmbl*Hz
vecmbl=np.arange(0,4,0.1)
r=np.zeros_like(vecmbl)
var_r=np.zeros_like(vecmbl)
nreal=100

i=-1
for hmbl in tqdm(vecmbl):
    i=i+1
    for j in np.arange(nreal):
        # draw random field uniformly from [-1.0,1.0] for each lattice site
        unscaled_fields=-1+2*ranf((basis.L,))
        
        h_z=[[unscaled_fields[i],i] for i in range(basis.L)]
        disorder_field = [["z",h_z]]
        
        Hz=hamiltonian(disorder_field,[],basis=basis,dtype=np.float64,**no_checks)
        H_MBL=H_XXZ+hmbl*Hz
        rr=ratio(H_MBL)
        r[i]=r[i]+rr
        var_r[i]=var_r[i]+rr**2
        
r=r/nreal
var_r=var_r/nreal
var_r=var_r-r**2
err=np.sqrt(var_r)/(2*np.sqrt(nreal))
plt.errorbar(vecmbl,r,yerr=err,fmt='o-')


