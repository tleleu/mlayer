import numpy as np
import scipy.sparse as sp
from scipy.sparse import coo_matrix, csr_matrix
import random

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
