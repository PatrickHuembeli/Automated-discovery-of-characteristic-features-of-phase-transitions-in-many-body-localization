from scipy.sparse import csr_matrix, rand
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import qutip as qt
import scipy.sparse.linalg as sl


# Hamiltonian from Many-body localization edge in the random-ﬁeld Heisenberg chain
# https://arxiv.org/pdf/1411.0660.pdf
# In the paper they set J=Jz=1
def Hamiltonian_Heisenberg_chain(hmax, sigmax, sigmay, sigmaz, N, J, Jz):
    hz = np.random.uniform(-hmax, hmax, N)
    #hz = [2]*N
    H = sum(Jz*(sigmaz[j]*sigmaz[(j+1)%N]) + J*(sigmax[j]*sigmax[(j+1)%N]\
    + sigmay[j]*sigmay[(j+1)%N]) - hz[j]*sigmaz[j] for j in range(N)) #- hz[N-1]*sigmaz[N-1]
    return H


def statistics(n, E):
    delta_n = (E[n]-E[n-1])
    delta_np1 = (E[n+1]-E[n])
    print(delta_n, delta_np1, min(delta_n, delta_np1)/max(delta_n, delta_np1))
    return min(delta_n, delta_np1)/max(delta_n, delta_np1)
          
samples = 1

def find_epsilon_index(Energies, epsilon):
    # find epsilon = 0.5
    # 0.5 = (E-Emax)/(Emin-Emax) --> E = 0.5*(Emin+Emax)
    targetE = epsilon*(Energies[0]-Energies[-1])+Energies[-1]
    return np.abs(np.array(Energies) - targetE).argmin()


def mean_energy(samples):
    h_list = [0.001] + np.arange(0.1, 4.0, 0.3).tolist()
    h_list = [0.001]
    mean_list = []
    for hmax in h_list:
        stat = 0.0
        for i in range(samples):
            print(hmax, i)          
            H = Hamiltonian_Heisenberg_chain(hmax, sigmax, sigmay, sigmaz, N, J, Jz)
            m ,v = H.eigenstates()
            epsilon_index = find_epsilon_index(m, 0.5)
            deltas = [] 
            for n in range(max(1,epsilon_index-25),min(len(m)-1,epsilon_index+25)): #for n in range(1,2**N-1): # 
                deltas.append(statistics(n,m))
            mean = np.mean(deltas)
            stat += mean
            print(deltas, mean)
            #print('Js', Jz, J)
            #print('epsilon_index', epsilon_index, a)
            #print(hmax, mean)
        mean_list.append(stat/samples)
    return mean_list
    



def entanglement_entropy(hmax):
    liste = []
    for N in [4, 6, 8, 10]:
        print(N)
        sigmax = [1/2.0*qt.tensor([qt.qeye(2)]*(j-1) + [qt.sigmax()] + [qt.qeye(2)]*(N-j))
              for j in range(1, N+1)]
        sigmay = [1/2.0*qt.tensor([qt.qeye(2)]*(j-1) + [qt.sigmay()] + [qt.qeye(2)]*(N-j))
                  for j in range(1, N+1)]
        sigmaz = [1/2.0*qt.tensor([qt.qeye(2)]*(j-1) + [qt.sigmaz()] + [qt.qeye(2)]*(N-j))
                  for j in range(1, N+1)]
        H = Hamiltonian_Heisenberg_chain(hmax, sigmax, sigmay, sigmaz, N, J, Jz)
        m ,v = H.eigenstates()
        epsilon_index = 2**N//2 # Just take middle of spectrum as in paper
        rho = v[epsilon_index]
        liste.append(qt.entropy_vn(rho, base=np.e, sparse=False))
    return liste
    



    
N = 8
J = 1.0
Jz = 1.0
hmax = 1.0
qubitfactor = 1/2.0


sigmax = [qubitfactor*qt.tensor([qt.qeye(2)]*(j-1) + [qt.sigmax()] + [qt.qeye(2)]*(N-j))
          for j in range(1, N+1)]
sigmay = [qubitfactor*qt.tensor([qt.qeye(2)]*(j-1) + [qt.sigmay()] + [qt.qeye(2)]*(N-j))
          for j in range(1, N+1)]
sigmaz = [qubitfactor*qt.tensor([qt.qeye(2)]*(j-1) + [qt.sigmaz()] + [qt.qeye(2)]*(N-j))
          for j in range(1, N+1)]
          
          
a = mean_energy(samples)
#np.save('7mean_energy_N12', a)

"""
# Fidelity check
delta_hmax = 0.01
liste = []
for hmax in np.arange(0.1, 4.0, 0.1):
    meanfid = 0.0
    for i in range(samples):
        H = Hamiltonian_Heisenberg_chain(hmax, sigmax, sigmay, sigmaz, N, J, Jz)
        w1,v1 = H.eigenstates()
        H = Hamiltonian_Heisenberg_chain((hmax+delta_hmax), sigmax, sigmay, sigmaz, N, J, Jz)
        w2,v2 = H.eigenstates()
        fidel = qt.fidelity(v1[0], v2[0])
        meanfid +=fidel
    print(meanfid/samples)
    liste.append(meanfid/samples)
"""



#liste = check_phase_trans(samples)
    
"""
for i in range(samples):          
    H = Hamiltonian_Heisenberg_chain(hmax, sigmax, sigmay, sigmaz, N, J, Jz)
    m ,v = H.eigenstates()
    def statistics(n, E):
        delta_n = (E[n]-E[n-1])
        delta_np1 = (E[n+1]-E[n])
        return min(delta_n, delta_np1)/max(delta_n, delta_np1)
    
    a = [] 
    for n in range(1,2**N-1):
        a.append(statistics(n,m))
    mean = np.mean(a)
    stat += mean
    # print(stat/(i+1))
    print(mean)
    
    
    
    # SAVE STATES
    for state in v[N//4:3*N//4]:
        states.append(state)
        if stat > 0.49:
            labels.append([1,0])
        else:
            labels.append([0,1])
    
filename = 'MBL_states.h5'
f = h5py.File(filename, 'w')

X_dset = f.create_dataset('my_data', (len(labels), N**2, 1), dtype='f')
X_dset[:] = states

y_dset = f.create_dataset('my_labels', (len(labels), 2), dtype='i')
y_dset[:] = labels
f.close()
"""

