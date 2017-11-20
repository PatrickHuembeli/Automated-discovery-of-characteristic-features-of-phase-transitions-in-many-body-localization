from quspin.operators import hamiltonian # Hamiltonians and operators
from quspin.basis import spin_basis_1d # Hilbert space spin basis
import numpy as np # generic math functions
import h5py
from keras.utils.io_utils import HDF5Matrix

#np.random.seed(seed = 1)
##### define model parameters #####
L=12 # system size
Jxy= 1.0 # xy interaction
Jzz_0=1.0# zz interaction
hmax = 0.9

def Hamilt_qspin(Jxy, Jzz_0, hmax, N):
    #basis = spin_basis_1d(L,pauli=False)
    no_checks={"check_herm":False,"check_pcon":False,"check_symm":False}
    basis = spin_basis_1d(N,pauli=False,Nup=N//2) # zero magnetisation sector
    #pauli false indicates that we use 1/2 sigma convention
    #Nup is the number of spins pointing up.
    hz= np.random.uniform(-hmax, hmax, N)
    # define operators with OBC using site-coupling lists
    J_zz = [[Jzz_0,i,(i+1)%N] for i in range(N)] # PBC
    J_xy = [[Jxy/2.0,i,(i+1)%N] for i in range(N)] # PBC
    h_z=[[hz[i],i] for i in range(N)]

    # static and dynamic lists
    static = [["+-",J_xy],["-+",J_xy],["zz",J_zz]]
    disorder_field = [["z",h_z]]
    dynamic = []
    # compute the time-dependent Heisenberg Hamiltonian
    H_XXZ = hamiltonian(static,dynamic,basis=basis,dtype=np.float64,**no_checks)
    Hz =hamiltonian(disorder_field,[],basis=basis,dtype=np.float64,**no_checks)
    Htot = H_XXZ + Hz
    Hqspin = Htot.todense() #makes it to a normal numpy matrix
    return Hqspin


def statistics(n, E):
    delta_n = (E[n]-E[n-1])
    delta_np1 = (E[n+1]-E[n])
    #print(delta_n, delta_np1, min(delta_n, delta_np1)/max(delta_n, delta_np1))
    return min(delta_n, delta_np1)/max(delta_n, delta_np1)
          

def find_epsilon_index(Energies, epsilon):
    # find epsilon = 0.5
    # 0.5 = (E-Emax)/(Emin-Emax) --> E = 0.5*(Emin+Emax)
    targetE = epsilon*(Energies[0]-Energies[-1])+Energies[-1]
    return np.abs(np.array(Energies) - targetE).argmin()


def mean_energy(samples, energies, hmax):         
    epsilon_index = find_epsilon_index(energies, 0.5)
    deltas = [] 
    for n in range(max(1,epsilon_index-25),min(len(energies)-1,epsilon_index+25)): #for n in range(1,2**N-1): # 
        deltas.append(statistics(n,energies))
    mean = np.mean(deltas)
    return mean, epsilon_index
    
hmax_list = [0.9, 5.0] #These are two values deep in the phases with good values
hmax_list = np.linspace(1.0, 4.8, 20)

samples = 10
labels = []
state_list = []
liste = []
for hmax in hmax_list:
    stat = 0.0
    for i in range(samples):
        print(hmax, i)
        H = Hamilt_qspin(Jxy, Jzz_0, hmax, L)
        energies, states = np.linalg.eig(H)
        sort_indices = energies.argsort()
        energies = energies[sort_indices]
        states = states[sort_indices]
        #energies = np.sort(energies)
        mean, epsilon_index = mean_energy(samples, energies, hmax)
        stat += mean
        for i in states[max(1,epsilon_index-25): min(len(energies)-1,epsilon_index+25)]:
            psi = np.asarray(i)
            psi = psi.reshape(psi.shape[1],)
            state_list.append(psi)
            if hmax<2.0:
                labels.append([1,0])
            else:
                labels.append([0,1])
    liste.append(stat/samples)
    
#import matplotlib.pyplot as plt
#plt.close()
#plt.plot(np.arange(0.2, 5.0, 0.2), liste)
#plt.savefig('N12_test')
folder = 'MBL_data/'
filename = folder + 'TARGET_Set_09_to_50_2' + '_N' + str(L) + '.h5'

f = h5py.File(filename, 'w')
# Creating dataset to store features
X_dset = f.create_dataset('my_data', (len(labels), state_list[0].shape[0]), dtype='f')
X_dset[:] = state_list
# Creating dataset to store labels
y_dset = f.create_dataset('my_labels', (len(labels), 2), dtype='i')
y_dset[:] = labels
f.close()


"""
# -----------------------------------------------------------------------------
# QUTIP Hamiltonian
from scipy.sparse import csr_matrix, rand
#import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import qutip as qt
import scipy.sparse.linalg as sl

plt.close()

# Hamiltonian from Many-body localization edge in the random-ﬁeld Heisenberg chain
# https://arxiv.org/pdf/1411.0660.pdf
# In the paper they set J=Jz=1
def Hamiltonian_Heisenberg_chain(hmax, sigmax, sigmay, sigmaz, N, J, Jz):
    hz = np.random.uniform(-hmax, hmax, N)
    hz = hzfix
    #hz = [2]*N
    H = np.sum(Jz*(sigmaz[j]*sigmaz[(j+1)%N]) + J*(sigmax[j]*sigmax[(j+1)%N]\
    + sigmay[j]*sigmay[(j+1)%N]) + hz[j]*sigmaz[j] for j in range(N)) #- hz[N-1]*sigmaz[N-1]
    return H
   
N = L
qubitfactor = 1/2.0
sigmax = [qubitfactor*qt.tensor([qt.qeye(2)]*(j-1) + [qt.sigmax()] + [qt.qeye(2)]*(N-j))
          for j in range(1, N+1)]
sigmay = [qubitfactor*qt.tensor([qt.qeye(2)]*(j-1) + [qt.sigmay()] + [qt.qeye(2)]*(N-j))
          for j in range(1, N+1)]
sigmaz = [qubitfactor*qt.tensor([qt.qeye(2)]*(j-1) + [qt.sigmaz()] + [qt.qeye(2)]*(N-j))
          for j in range(1, N+1)]
    
H = Hamiltonian_Heisenberg_chain(hmax, sigmax, sigmay, sigmaz, N, 0.0, 0.0)
"""
