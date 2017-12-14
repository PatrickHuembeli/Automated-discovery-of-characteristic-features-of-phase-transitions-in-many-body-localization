from quspin.operators import hamiltonian # Hamiltonians and operators
from quspin.basis import spin_basis_1d # Hilbert space spin basis
import numpy as np # generic math functions
import h5py
from keras.utils.io_utils import HDF5Matrix

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


def mean_energy(samples, energies, hmax, epsilon):         
    epsilon_index = find_epsilon_index(energies, epsilon)
    deltas = [] 
    for n in range(max(1,epsilon_index-25),min(len(energies)-1,epsilon_index+25)): #for n in range(1,2**N-1): # 
        deltas.append(statistics(n,energies))
    mean = np.mean(deltas)
    return mean, epsilon_index

def calc_samples(epsilon, hmax, margin , stat):
    H = Hamilt_qspin(Jxy, Jzz_0, hmax, L)
    energies, states = np.linalg.eig(H)
    sort_indices = energies.argsort()
    energies = energies[sort_indices]
    states = states[sort_indices]
    mean, epsilon_index = mean_energy(samples, energies, hmax, epsilon)
    stat += mean
    for i in states[max(1,epsilon_index-margin): min(len(energies)-1,epsilon_index+margin)]:
        psi = np.asarray(i)
        psi = psi.reshape(psi.shape[1],)
        state_list.append(psi)
        if hmax<2.0:
            labels.append([1,0])
        else:
            labels.append([0,1])
    return stat
    
    
def save_H5(filename):
    f = h5py.File(filename, 'w')
    # Creating dataset to store features
    X_dset = f.create_dataset('my_data', (len(labels), state_list[0].shape[0]), dtype='f')
    X_dset[:] = state_list
    # Creating dataset to store labels
    y_dset = f.create_dataset('my_labels', (len(labels), 2), dtype='i')
    y_dset[:] = labels
    f.close()
    
L=12 # system size
Jxy= 1.0 # xy interaction
Jzz_0=1.0# zz interaction

samples = 10
margin = 25
labels = []
state_list = []
labels_fullspec = []
state_list_fullspec = []
labels_2D = []
state_list_2D = []
folder = 'Data/'

source = False
target = True
evaluation = False

if source:
    # SOURCE
    labels = []
    state_list = []
    hmax_list = np.append(np.linspace(0.8, 0.85, 2), np.linspace(5.1, 5.2, 2))
    epsilon_list = epsilon_list = np.linspace(0.1, 0.9, 9)
    filename = folder + '00SOURCE_N' + str(L) + '.h5'
    for epsilon in epsilon_list:
        liste = []
        for hmax in hmax_list:
            stat = 0.0
            for i in range(samples):
                print('SOURCE, ', 'epsilon:', epsilon, 'hmax:', hmax, i)
                stat = calc_samples(epsilon, hmax, margin, stat)
            liste.append(stat/samples)
        np.save('SOURCE_statistics_N_'+str(L)+'epsilon_'+str(epsilon), liste)
    save_H5(filename)

    
if target:
    # TARGET
    labels = []
    state_list = []
    hmax_list = np.linspace(0.9, 5.0, 20)
    epsilon_list = [0.5]
    filename = folder + '00TARGET_N' + str(L) + '.h5'
    for epsilon in epsilon_list:
        liste = []
        for hmax in hmax_list:
            stat = 0.0
            for i in range(samples):
                print('TARGET, ', 'epsilon:', epsilon, 'hmax:', hmax, i)
                stat = calc_samples(epsilon, hmax, margin, stat)
            liste.append(stat/samples)
        np.save('TARGET_statistics_N_'+str(L)+'epsilon_'+str(epsilon), liste)
    save_H5(filename)
    
    
if evaluation:
    # EVALUATION SET
    labels = []
    state_list = []
    folder = 'EVALUATION_FILES/'
    hmax_list = np.linspace(0.9, 5.0, 20)
    epsilon_list = epsilon_list = np.linspace(0.1, 0.9, 9)
    for epsilon in epsilon_list:
        liste = []
        for hmax in hmax_list:
            stat = 0.0
            labels = []
            state_list = []
            labels_fullspec = []
            for i in range(samples):
                print('EVALUATE, ', 'epsilon:', epsilon, 'hmax:', hmax, i)
                stat = calc_samples(epsilon, hmax, margin, stat)
            liste.append(stat/samples)
            #filename = folder + '1000_samp_EVALUATE_N' + str(L) + 'eps'+ str(epsilon) + 'hmax' + str(hmax) +'.h5'
            print(len(state_list))
            #save_H5(filename)
        np.save('EVALUATION_statistics_N_'+str(L)+'_sampels_'+str(samples)+'epsilon_'+str(epsilon), liste)
        
        
        
        
