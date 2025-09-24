#####################################################################
#
# Timothee Leleu
#
# March 2025
#
# Library for generating different types of instances
#
#
#####################################################################

from scipy.sparse import csr_matrix
import networkx as nx
import numpy as np

# ---------------------------------------------------------------------------
# Bethe lattice (random regular graph)
# Note: BP should give e = -1.2777
# ---------------------------------------------------------------------------

def create_Bethe(N, d):
    
    if N * d % 2 != 0:
        raise ValueError("N * d must be even for a regular graph")

    G = nx.random_regular_graph(d, N)
    J = nx.adjacency_matrix(G).tocsr()

    rows, cols = J.nonzero()
    mask = rows < cols  # Only upper triangle (to avoid double counting)
    
    for i, j in zip(rows[mask], cols[mask]):
        sign = np.random.choice([-1, 1])
        J[i, j] = sign
        J[j, i] = sign

    return J
    
# ---------------------------------------------------------------------------
# SK
# ---------------------------------------------------------------------------

def create_SK(N):

    J = np.random.uniform(-1,1,(N,N))
    J = np.sign(J)
    J = np.triu(J) - np.diag(np.diag(J))
    J = J + J.T
    np.fill_diagonal(J, 0)
    J = csr_matrix(J)
    
    return J