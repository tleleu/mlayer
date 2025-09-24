import numpy as np

import itertools

import scipy.sparse as sp

import networkx as nx

from scipy.sparse import coo_matrix, csr_matrix

import random

def generate_SK(N):
    J = np.random.uniform(-1,1,(N,N))
    J = np.sign(J)
    J = np.triu(J) - np.diag(np.diag(J))
    J = J + J.T
    np.fill_diagonal(J, 0)
    return J

def random_Bethe(J):

    J = J.tocoo()  # Convert to COO format for easy row-column access

    # Generate a random {-1, 1} value for each unique (i, j) pair where i < j
    random_values = np.random.choice([-1, 1], size=len(J.data) // 2)

    # Modify values while ensuring symmetry
    modified_data = np.zeros_like(J.data)

    for idx in range(0, len(J.data), 2):  # Iterate over unique (i, j) pairs
        i, j = J.row[idx], J.col[idx]

        if i != j:  # Ignore diagonal elements
            a = random_values[idx // 2]  # Get a random ±1 multiplier
            modified_data[idx] = J.data[idx] * a  # J[i, j]
            modified_data[idx + 1] = J.data[idx + 1] * a  # J[j, i] (same multiplier)
        else:
            modified_data[idx] = J.data[idx]  # Keep diagonal elements unchanged

    # Create a new symmetric sparse matrix
    J_modified = coo_matrix((modified_data, (J.row, J.col)), shape=J.shape)

    return J_modified.tocsr()  # Convert back to CSR format for efficiency

def generate_random_regular_graph(N, d, sparse=False):
    # Check if the degree and number of nodes are compatible
    if N * d % 2 != 0:
        raise ValueError("N * d must be even for a regular graph")

    G = nx.random_regular_graph(d, N)
    
    if sparse==False:
        J = nx.adjacency_matrix(G).todense()
    else:
        J = nx.adjacency_matrix(G).tocsr()

    return J

def generate_permutation(C):
    M = C.shape[0]  # Size of the matrix
    permutation = np.zeros(M, dtype=int)  # To store the permutation
    assigned = np.full(M, False)  # To track which elements have been assigned

    for a in range(M):
        # Row C[a, :] gives the probabilities of permuting a to any b
        row_probabilities = C[a, :].copy()

        # Set the probabilities of already assigned elements to 0
        row_probabilities[assigned] = 0

        # Normalize the row so the sum of the probabilities is 1
        row_probabilities /= np.sum(row_probabilities)

        # Sample b from the row based on the given probabilities
        b = np.random.choice(M, p=row_probabilities)

        # Assign the permutation of a -> b
        permutation[a] = b

        # Mark b as assigned
        assigned[b] = True

    return permutation

def apply_generalized_diagonals_with_cycle(matrix_size, max_diagonal):
    # Create a square matrix filled with zeros
    C = np.zeros((matrix_size, matrix_size), dtype=int)
    
    # Apply the diagonals dynamically with regular boundary conditions
    for offset in range(0, max_diagonal + 1):
        np.fill_diagonal(C[offset:], 1)        # Diagonal below the main diagonal
        np.fill_diagonal(C[:, offset:], 1)     # Diagonal above the main diagonal
    
    # Apply periodic (cycle) boundary conditions by wrapping around diagonals
    for offset in range(1, max_diagonal + 1):
        for i in range(matrix_size):
            # Wrap around from top-right to bottom-left
            C[i, (i + offset) % matrix_size] = 1
            # Wrap around from bottom-left to top-right
            C[(i + offset) % matrix_size, i] = 1
    
    return C

def Mlayer(J, M, permute=True, GoG=False, do_anneal=False, typeperm='asym'):
    
    if M==1:
        return sp.csr_matrix(J)

    N = J.shape[0]
    
    Q = np.triu(J)
    
    nzi = np.nonzero(Q)
    
    irow = []
    icol = []
    qdata = []
    for i, j in zip(nzi[0], nzi[1]):
        irow.append(i)
        icol.append(j)
        qdata.append(Q[i, j])
        
    def flatten_index(a, i, N):
        return a * N + i
    
    if GoG == True:
        
        #C = apply_generalized_diagonals_with_cycle(M, 3)
        #C = C + 0.00000001
        #C = C/np.max(C)
        
        def gaussian(x, mu, sigma):
            return np.exp(-((x - mu) ** 2) / (2 * sigma ** 2))
        def generate_gaussian_matrix(M, d):
            matrix = np.zeros((M, M))
            x = np.arange(M)
            
            for i in range(M):
                for j in range(M):
                    matrix[i, j] = gaussian(j, i, d)  # Gaussian centered at diagonal (i, i)
            
            return matrix
        d = 2
        C = generate_gaussian_matrix(M, d)
        C = C + 0.00000001
        C = C/np.max(C)
    
    if permute == False:
            
        irow_M = []
        icol_M = []
        qdata_M = []
        for i, j, q in zip(irow,icol,qdata):
            for a in range(M):
                irow_M.append(flatten_index(a, i, N))
                icol_M.append(flatten_index(a, j, N))
                qdata_M.append(Q[i, j])
                
                irow_M.append(flatten_index(a, j, N))
                icol_M.append(flatten_index(a, i, N))
                qdata_M.append(Q[i, j])
                
    else:
        
        irow_M = []
        icol_M = []
        qdata_M = []
        for i, j, q in zip(irow,icol,qdata):
            
            #ref
            if typeperm == 'asym':
                if GoG == False:
                    perm1 = np.random.permutation(M)
                    perm2 = np.random.permutation(M)
                    
                else:
                    perm1 = generate_permutation(C)
                    perm2 = generate_permutation(C)
                    
                    
                for a in range(M):
                    
                    b1 = perm1[a]
                    b2 = perm2[a]
                    
                    irow_M.append(flatten_index(b1, i, N))
                    icol_M.append(flatten_index(a, j, N))
                    if do_anneal==False:
                        qdata_M.append(Q[i, j])
                    else:
                        rdm = np.random.choice([-1, 1])
                        qdata_M.append(Q[i, j]*rdm)
                    
                    irow_M.append(flatten_index(b2, j, N))
                    #irow_M.append(flatten_index(b1, j, N))
                    icol_M.append(flatten_index(a, i, N))
                    if do_anneal==False:
                        qdata_M.append(Q[i, j])
                    else:
                        qdata_M.append(Q[i, j]*rdm)
                  
            # testing symmetric M-layer
            if typeperm == 'sym':
           
                if GoG == False:
                    perm1 = np.random.permutation(M)
                    perm2 = np.random.permutation(M)
                    
                else:
                    perm1 = generate_permutation(C)
                    perm2 = generate_permutation(C)
                    
                    
                for a in range(M):
                    
                    b1 = perm1[a]
                    b2 = perm2[a]
                    
                    if do_anneal==False:
                        irow_M.append(flatten_index(b1, i, N))
                        icol_M.append(flatten_index(a, j, N))
                        qdata_M.append(Q[i, j])
                        irow_M.append(flatten_index(a, j, N))
                        icol_M.append(flatten_index(b1, i, N))
                        qdata_M.append(Q[i, j])
                    else:
                        rdm = np.random.choice([-1, 1])
                        irow_M.append(flatten_index(b1, i, N))
                        icol_M.append(flatten_index(a, j, N))
                        qdata_M.append(Q[i, j]*rdm)
                        irow_M.append(flatten_index(a, j, N))
                        icol_M.append(flatten_index(b1, i, N))
                        qdata_M.append(Q[i, j]*rdm)
                  
                  
            #double permutation
            if 0:
            
                if GoG == False:
                    perm1 = np.random.permutation(M)
                    perm2 = np.random.permutation(M)
                    perm3 = np.random.permutation(M)
                    perm4 = np.random.permutation(M)
                else:
                    perm1 = generate_permutation(C)
                    perm2 = generate_permutation(C)
                    perm3 = generate_permutation(C)
                    perm4 = generate_permutation(C)
                
                for a in range(M):
                    
                    a1 = perm1[a]
                    a2 = perm2[a]
                    b1 = perm3[a]
                    b2 = perm4[a]
                    
                    irow_M.append(flatten_index(b1, i, N))
                    icol_M.append(flatten_index(a1, j, N))
                    if do_anneal==False:
                        qdata_M.append(Q[i, j])
                    else:
                        rdm = np.random.choice([-1, 1])
                        qdata_M.append(Q[i, j]*rdm)
                    
                    irow_M.append(flatten_index(b2, j, N))
                    icol_M.append(flatten_index(a2, i, N))
                    if do_anneal==False:
                        qdata_M.append(Q[i, j])
                    else:
                        qdata_M.append(Q[i, j]*rdm)
        
        
    JM = sp.csr_matrix((qdata_M, (irow_M, icol_M)), shape=(N*M, N*M))
    
    if 0:
        import matplotlib.pyplot as plt
        
        # Plotting the 2D matrix
        #plt.imshow(JM.todense(), cmap='viridis', interpolation='nearest')
        plt.imshow(np.abs(JM.todense()), cmap='viridis', interpolation='nearest')
        
        # Adding a colorbar to show the scale
        plt.colorbar(label='Value')
        
    if 0:
        #tests
        J_ = np.abs(np.array(JM.todense()))
        
        print(f'Nb of connections',np.sum(np.abs(J_)))
        print(f'Nb of asymmetric connections (0 if symmetric)',np.sum(np.abs(J_-J_.T)/2))
        print(f'Nb of symmetric connections (0 if asymmetric)',np.sum(np.abs(J_)*np.abs(J_).T))
        
        print(f'Nb of symmetric connections J (0 if asymmetric)',np.sum(np.abs(J)*np.abs(J).T))
        
    return JM


def calculate_delta_E(spins, J):
    # Convert spins to float for matrix multiplication
    spins_float = spins.astype(float)
    
    # Calculate the interaction term with broadcasting J across K instances
    interaction_term = np.matmul(J, spins_float.T)  # KxN x NxN -> KxN
    delta_E = 2 * spins_float.T * interaction_term
    
    return delta_E.T

def is_local_minimum(spins, J):
    # Convert spins to float for matrix multiplication
    spins_float = spins.astype(float)
    
    # Calculate the interaction term
    if sp.issparse(J) and isinstance(J, sp.csr_matrix):
        # Use sparse matrix multiplication for CSR matrix
        interaction_term = J @ spins_float.T  # Optimized for sparse matrix multiplication
    else:
        # Fallback to dense matrix multiplication if J is not sparse
        interaction_term = np.matmul(J, spins_float.T)  # Standard dense multiplication
    
    delta_E = 2 * spins_float.T * interaction_term

    # If all delta_E >= 0, we are at a local minimum
    return np.all(delta_E >= 0, axis=0)

def check_local_minima_replicas(final_spins, J, M):
    K = final_spins.shape[0]
    N = J.shape[0]

    # Reshape final_spins to (K, M, N)
    spins_reshaped = final_spins.reshape(K, M, N)
    
    # Initialize the array to store local minima status
    is_minima = np.zeros((K, M), dtype=bool)
    
    # Check if each replica is a local minimum
    for m in range(M):
        is_minima[:, m] = is_local_minimum(spins_reshaped[:, m, :], J)
    
    return is_minima

def calculate_energy(spins, J):
    # Convert spins to float for calculations
    spins_float = spins.astype(float)
    
    # Calculate interaction energy: -1/2 * sum_ij J_ij * S_i * S_j
    if sp.issparse(J) and isinstance(J, sp.csr_matrix):
        # Use sparse matrix multiplication for CSR matrix
        total_energy = -0.5 * np.sum(spins_float.T * (J @ spins_float.T), axis=0)
    else:
        # Fallback to dense matrix multiplication if J is not sparse
        total_energy = -0.5 * np.sum(spins_float.T * np.matmul(J, spins_float.T), axis=0)
    
    return total_energy

def calculate_energy_replicas(final_spins, J, M):
    K = final_spins.shape[0]
    N = J.shape[0]

    # Reshape final_spins to (K, M, N)
    spins_reshaped = final_spins.reshape(K, M, N)
    
    # Initialize the array to store energy values
    energies = np.zeros((K, M))
    
    # Calculate energy for each replica
    for m in range(M):
        energies[:, m] = calculate_energy(spins_reshaped[:, m, :], J)
    
    return energies / N
