import numpy as np

import itertools

import scipy.sparse as sp

from instance import create_Bethe

def sec_min_1(arr):
    arr = np.array(arr)
    
    # This will store the second minimum for each row
    second_min_values = np.full(arr.shape[0], np.nan)  # Initialize with NaNs
    
    for i in range(arr.shape[0]):
        unique_row = np.unique(arr[i])  # Get unique elements in each row
        if len(unique_row) >= 2:  # Ensure at least two distinct elements exist
            second_min_values[i] = np.sort(unique_row)[1]  # Get the second minimum
    
    return second_min_values

#dp = count_decimal_places(a)
def count_decimal_places(number):
    """Returns the number of decimal places in a float."""
    # Convert to string and split on the decimal point
    str_number = str(number)
    if '.' in str_number:
        return len(str_number.split('.')[1])
    else:
        return 0
    
def create_instance(N, Jtype='SK'):
    
    if Jtype == 'SK':
        return create_instance_SK(N)
    if Jtype == 'RR':
        k=2
        J = create_Bethe(N, k + 1)
        J = np.array(J.todense(),dtype=float)
        return J, 0, 0, 0, 0, []
    #if Jtype == 'wishart':
    #    return create_instance_Wishart(N)

def generate_L(d):
    L = np.array(list(itertools.product([-1, 1], repeat=d)))
    return L

def brute_force(W,L):
    
    N = len(W[0,:])
    
    #L = np.array(list(itertools.product([-1, 1], repeat=d)))
    P = (L @ W) * L
    lm = np.sum(P>0,1)==N #check is state is local minima or not
    H = -0.5 * np.sum( P ,1)
    i0 = np.argmin(H)
    H0= H[i0]
    H1 = sec_min_1(np.expand_dims(H,0))
    s0 = L[i0,:]

    return H, s0, H0, H1, lm

def create_instance_SK(N):

    # Custom connectivity matrix J (NxN) for each spin interaction
    if 0:
        J = np.random.randn(N, N)
        J = (J + J.T) / np.sqrt(2 * N)  # Make J symmetric
        J = np.round(J,5)
    else:
        J = np.random.uniform(-1,1,(N,N))
        J = np.sign(J)
        J = np.triu(J) - np.diag(np.diag(J))
        J = J + J.T

    # Remove the diagonal elements by setting them to zero
    np.fill_diagonal(J, 0)
    
    # Ground-state energy unknown for SK
    H0 = np.array(0.0)
    H1 = np.array(0.0)
    aH0 = []
    
    # Ground-state configuration
    gs = np.zeros(N)
    
    #brute force if small graph
    if 0:
        if N<=22:
            L = generate_L(N)
            aH0, gs, H0, H1, lm0 = brute_force(J,L)
    
    dp = count_decimal_places(H0)
    
    return J, H0 / N, H1 / N, gs, dp, aH0

def generate_random_graph(n, d):
    # Calculate the edge probability based on the desired average degree
    p = d / (n - 1)

    # Initialize an adjacency matrix with all zeros
    adj_matrix = np.zeros((n, n), dtype=int)

    # Populate the adjacency matrix based on the probability p
    for i in range(n):
        for j in range(i + 1, n):
            if np.random.random() < p:
                # Add an edge between i and j with probability p
                adj_matrix[i][j] = 1
                adj_matrix[j][i] = 1  # Ensure the graph is undirected

    return adj_matrix


def generate_permutation_old(C):
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

def generate_permutation(C, *, eliminate_drift=False, rng=None):
    """
    Sequential without-replacement sampler.
    If eliminate_drift=True, apply:
      - random row order,
      - random cyclic rotation of rows+cols (for ring kernels),
      - 50% orientation flip via C^T.
    """
    rng = np.random.default_rng(rng)
    M = C.shape[0]

    def core(Cmat, row_order):
        perm = np.empty(M, dtype=int)
        assigned = np.zeros(M, dtype=bool)
        for a in row_order:
            probs = Cmat[a].astype(float).copy()
            probs[assigned] = 0.0
            s = probs.sum()
            if s <= 0.0:
                # uniform over remaining columns
                choices = np.flatnonzero(~assigned)
                b = rng.choice(choices)
            else:
                probs /= s
                b = rng.choice(M, p=probs)
            perm[a] = b
            assigned[b] = True
        return perm

    if not eliminate_drift:
        return core(C, np.arange(M))

    # --- drift elimination ---
    # 1) random orientation flip with prob 1/2
    use_CT = rng.random() < 0.5
    C_eff = C.T if use_CT else C

    # 2) random cyclic rotation (preserves ring-locality, kills global direction)
    k = rng.integers(M) if M > 1 else 0
    C_rot = np.roll(np.roll(C_eff, -k, axis=0), -k, axis=1)

    # 3) random row order
    row_order = rng.permutation(M)

    # sample on rotated, possibly transposed kernel
    perm_rot = core(C_rot, row_order)

    # undo rotation
    # perm_rot maps a' -> b' in rotated indices; map back: a=(a'+k)%M, b=(b'+k)%M
    perm = np.empty(M, dtype=int)
    for a_prime, b_prime in enumerate(perm_rot):
        a = (a_prime + k) % M
        b = (b_prime + k) % M
        perm[a] = b

    # if we sampled on C^T, we built b->a; invert to get a->b
    if use_CT:
        inv = np.empty(M, dtype=int)
        inv[perm] = np.arange(M)
        perm = inv

    return perm

def directed_permutation_sequential(C, kappa=0.0, eliminate_drift=False, rng=None):
    rng = np.random.default_rng(rng)
    M = C.shape[0]
    idx = np.arange(M)
    sdist = (idx[:, None] - idx[None, :])
    sdist = ( (sdist + (M//2)) % M ) - (M//2)

    def core(Cmat, order):
        perm = np.empty(M, dtype=int)
        assigned = np.zeros(M, dtype=bool)
        for a in order:
            probs = Cmat[a].astype(float) * np.exp(kappa * sdist[a])
            probs[assigned] = 0.0
            s = probs.sum()
            if s <= 0:
                choices = np.flatnonzero(~assigned)
                b = rng.choice(choices)
            else:
                probs /= s
                b = rng.choice(M, p=probs)
            perm[a] = b
            assigned[b] = True
        return perm

    if not eliminate_drift:
        return core(C, np.arange(M))

    # eliminate global artifacts but keep directional tilt
    use_CT = rng.random() < 0.5
    Ce = C.T if use_CT else C
    k = rng.integers(M) if M > 1 else 0
    Ce = np.roll(np.roll(Ce, -k, axis=0), -k, axis=1)
    order = rng.permutation(M)

    p_rot = core(Ce, order)

    # undo rotation
    perm = np.empty(M, dtype=int)
    for ap, bp in enumerate(p_rot):
        a = (ap + k) % M
        b = (bp + k) % M
        perm[a] = b

    if use_CT:
        inv = np.empty(M, dtype=int)
        inv[perm] = np.arange(M)
        perm = inv
    return perm

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

def generate_tree_adjacency_matrix(M, d):
    if M < 1 or d < 1:
        raise ValueError("M and d must be positive integers.")
    
    # Create an adjacency matrix initialized with zeros
    adjacency_matrix = np.zeros((M, M), dtype=int)

    # Initialize the tree with the root node (node 0)
    current_node = 0
    node_queue = [current_node]  # Queue to track nodes to which new nodes will be attached
    next_node = 1

    # While there are nodes to attach and unconnected nodes
    while next_node < M:
        # Get the next parent node from the queue
        parent = node_queue.pop(0)
        
        # Attach up to d new nodes to this parent
        for _ in range(d):
            if next_node < M:
                adjacency_matrix[parent][next_node] = 1
                adjacency_matrix[next_node][parent] = 1
                node_queue.append(next_node)
                next_node += 1
            else:
                break

    return adjacency_matrix

def generate_scale_free_adjacency_matrix(M, m):
    """
    Generate an adjacency matrix for a scale-free graph using the Barabási-Albert (BA) model.
    
    Parameters:
    M: Number of nodes in the graph.
    m: Number of edges to attach from a new node to existing nodes.
    
    Returns:
    adjacency_matrix: The adjacency matrix representing the graph.
    """
    if M < 1 or m < 1 or m >= M:
        raise ValueError("Ensure that M > 1, m >= 1, and m < M.")
    
    # Initialize the adjacency matrix with zeros
    adjacency_matrix = np.zeros((M, M), dtype=int)
    
    # Start with a fully connected small network of `m` nodes
    for i in range(m):
        for j in range(i + 1, m):
            adjacency_matrix[i][j] = 1
            adjacency_matrix[j][i] = 1
    
    # Preferential attachment: add new nodes one by one
    for new_node in range(m, M):
        # Calculate the total number of edges to choose preferentially
        total_edges = np.sum(adjacency_matrix)
        
        # Probability of connecting to each existing node is proportional to its degree
        degrees = np.sum(adjacency_matrix, axis=0)
        probabilities = degrees / total_edges
        
        # Select `m` unique existing nodes based on their degree
        connected_nodes = set()
        while len(connected_nodes) < m:
            selected_node = np.random.choice(range(new_node), p=probabilities[:new_node])
            connected_nodes.add(selected_node)
        
        # Connect the new node to the selected nodes
        for node in connected_nodes:
            adjacency_matrix[new_node][node] = 1
            adjacency_matrix[node][new_node] = 1
    
    return adjacency_matrix

#def Mlayer(J, M, permute=True):
def Mlayer_dense(J, M, permute=True):
    
    if M==1:
        return J

    N = J.shape[0]
    
    # Initialize a tensor of zeros
    JM = np.zeros((M, N, M, N))
    
    # 1 - Replicate the coupling within layers
    for a in range(M):
        JM[a, :, a, :] = J
        
    if 1: #test
        #C = np.eye(M)    
        #C = np.random.rand(M, M)
        #C = (C+C.T)/2
        #C = C/np.max(C)
        
        #C = np.eye(M) # without coupling
        #C = np.ones((M,M)) #unifom M-layer
        #C = np.array(generate_random_graph(M, 3)).astype(float) + 0.001
        
        C = apply_generalized_diagonals_with_cycle(M, 3)  #Seems to be working well for SK
        
        #C = generate_tree_adjacency_matrix(M, 3)
        
        #C = generate_scale_free_adjacency_matrix(M, 5)
            
        C = C + 0.0000001
        
        C = C/np.max(C)
        
    # 2 - Iterate over all pairs of spins and permute the connections
    if permute:
        for i in range(N):
            
         
            for j in range(i + 1, N):
                
                if 1: #test
                    perm1 = generate_permutation(C)
                    perm2 = generate_permutation(C)
                    #perm1 = np.random.permutation(M)
                    #perm2 = np.random.permutation(M)

                    cW1 = JM[:,i,:,j]
                    JM[:,i,:,j] = JM[:,i,perm1,j]
                    JM[:,j,:,i] = JM[:,j,perm2,i]
                    
                if 0: #reference: kind of working but not symmetric
                    perm1 = np.random.permutation(M)
                    perm2 = np.random.permutation(M)
    
                    cW1 = JM[:,i,:,j]
                    JM[:,i,:,j] = JM[:,i,perm1,j]
                    JM[:,j,:,i] = JM[:,j,perm2,i]
                    
                    #cW1 = JM[:,i,:,j]
                    #JM[:,i,:,j] = cW1[perm1]
                    #JM[:,j,:,i] = cW1[perm2]
                    
                if 0:
                    #p_perm = np.random.uniform(0,1)
                    p_perm = 1.0
                    perm = np.random.permutation(M)
                    r = np.random.uniform(0,1,M)>=p_perm #proba of perm
                    idx = list(range(M))
                    perm = r*idx + (1-r)*perm
                    cW1 = JM[perm,i,:,j]
                    JM[:,i,:,j] = cW1
                    JM[:,j,:,i] = cW1.T
                 
                if 0:
                    perm = np.random.permutation(M)
                    cW1 = JM[perm,i,:,j]    #WTF!
                    JM[:,i,:,j] = cW1
                    JM[:,j,:,i] = cW1
                    
                if 0:
                    perm = np.random.permutation(M)
                    cW1 = JM[:,i,perm,j]
                    cW2 = JM[perm,j,:,i]
                    JM[:,i,:,j] = cW1
                    JM[:,j,:,i] = cW2
                    
            #add RSA
            if 0:
                JM[:,i,:,i] = 0.01
            
    # 3 - Reshape to form the final (M*N, M*N) connectivity matrix
    JM = JM.reshape(N * M, N * M)
    
    return JM


def create_mixing_Q_step(M,width):
    C = apply_generalized_diagonals_with_cycle(M, width)
    C = C + 0.00000001
    C = C/np.max(C)
    return C

#def Mlayer_sparse(J, M, permute=True):
def Mlayer(J, M, permute=True, GoG=False, typeperm='asym', C=[]):
    
    def generate_permutation_(C):
        return generate_permutation(C,eliminate_drift=False)
        #return directed_permutation_sequential(C, kappa=4.0, eliminate_drift=False, rng=None)
        
    
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
                    
                if 1:
                    if GoG == False:
                        perm1 = np.random.permutation(M)
                        perm2 = np.random.permutation(M)
                    else:
                        perm1 = generate_permutation_(C)
                        perm2 = generate_permutation_(C)
                        
                        if False:
                            import utils
                            per_row, mean_shift, direction, strength, vals, probs = utils.shift_metrics_from_perm(perm1)
                            utils.plot_shift_distribution(vals, probs, outpath="energy/shift.pdf")
                            
                            s1, c1 = utils.best_shift(perm1)            # NOT a list-comp
                            print("best_shift:", s1, "coherence:", c1)
                        
                    for a in range(M):
                        
                        b1 = perm1[a]
                        b2 = perm2[a]
                        
                        irow_M.append(flatten_index(b1, i, N))
                        icol_M.append(flatten_index(a, j, N))
                        qdata_M.append(Q[i, j])
                        
                        irow_M.append(flatten_index(b2, j, N))
                        #irow_M.append(flatten_index(b1, j, N))
                        icol_M.append(flatten_index(a, i, N))
                        qdata_M.append(Q[i, j])
                else: #test
                
                    if GoG == False:
                        perm1 = np.random.permutation(M)
                        perm2 = np.random.permutation(M)
                        perm3 = np.random.permutation(M)
                        perm4 = np.random.permutation(M)
                    else:
                        perm1 = generate_permutation_(C)
                        perm2 = generate_permutation_(C)
                        perm3 = generate_permutation_(C)
                        perm4 = generate_permutation_(C)
                    
                    for a in range(M):
                        
                        a1 = perm1[a]
                        a2 = perm2[a]
                        b1 = perm3[a]
                        b2 = perm4[a]
                        
                        irow_M.append(flatten_index(b1, i, N))
                        icol_M.append(flatten_index(a1, j, N))
                        qdata_M.append(Q[i, j])
                        
                        irow_M.append(flatten_index(b2, j, N))
                        icol_M.append(flatten_index(a2, i, N))
                        qdata_M.append(Q[i, j])
                        
            # testing symmetric M-layer
            if typeperm == 'sym':
           
                if GoG == False:
                    perm1 = np.random.permutation(M)
                    perm2 = np.random.permutation(M)
                    
                else:
                    perm1 = generate_permutation_(C)
                    perm2 = generate_permutation_(C)
                    
                    
                for a in range(M):
                    
                    b1 = perm1[a]
                    b2 = perm2[a]
                    
                    irow_M.append(flatten_index(b1, i, N))
                    icol_M.append(flatten_index(a, j, N))
                    qdata_M.append(Q[i, j])
                    irow_M.append(flatten_index(a, j, N))
                    icol_M.append(flatten_index(b1, i, N))
                    qdata_M.append(Q[i, j])    
        
    JM = sp.csr_matrix((qdata_M, (irow_M, icol_M)), shape=(N*M, N*M))
    
    if 0:
        import matplotlib.pyplot as plt
        
        # Plotting the 2D matrix
        plt.imshow(JM.todense(), cmap='viridis', interpolation='nearest')
        
        # Adding a colorbar to show the scale
        plt.colorbar(label='Value')
            
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


def load_matrix_from_file(filename, N):
    # Initialize an N x N matrix with zeros
    matrix = np.zeros((N, N))
    
    # Open the file and read each line
    with open(filename, 'r') as file:
        for line in file:
            # Split the line into i, j, and aij
            i, j, aij = map(int, line.split())
            # Place the value aij in the matrix at position (i, j)
            matrix[i-1][j-1] = aij
            
    matrix = matrix + matrix.T
       
    return matrix

def load_numbers_from_file(filename):
    # Initialize an empty list to store the numbers
    numbers = []
    
    # Open the file and read each line
    with open(filename, 'r') as file:
        for line in file:
            # Convert the line to a float (or int if you prefer) and add it to the list
            number = float(line.strip())
            numbers.append(number)
    
    # Convert the list to a NumPy array
    numbers_array = np.array(numbers)
    
    return numbers_array
 
    