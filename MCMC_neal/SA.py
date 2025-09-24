#from pyqubo import Spin
#from pyqubo import Array, Placeholder, solve_qubo
import neal
import neal.simulated_annealing as sa

#import dimod

import lib

import numpy as np

def run_SA(N,Q,T,K,beta,SAcode='py',anneal=False,x0=[]):
    if SAcode == 'neal':
        return run_SA_neal(N,Q,T,K,beta,anneal=anneal,x0=x0)
    if SAcode == 'py':
        return run_SA_py(N,Q,T,K,beta)
   
    
def run_SA_py(N,Q,T,K,beta):
   
    x = np.random.normal(0,1,size=(N,K))
    x = np.sign(x)
    u = np.matmul(Q,x)
    if beta > 50:
        P = (1+np.sign(u))/2
    else:
        P = 1/(1+np.exp(-u*beta))
    H = -0.5*np.sum(x*u,axis=0)
    
    aH = []
    
    for t in range(T):
            
        rid = np.random.randint(0, N, size=K)

        rdm = np.random.uniform(0,1,size=K)
        P_temp = P[rid,np.arange(len(rid))]
        x_temp = P_temp>rdm
        x[rid,np.arange(len(rid))] = 2*x_temp-1

        u = Q @ x
        if beta > 50:
            P = (1+np.sign(u))/2
        else:
            P = 1/(1+np.exp(-u*beta))

        
    H = -0.5 * np.sum(x * np.matmul(Q, x) , 0)
       
    return H / N, x.T


def run_SA_py_(N, Q, T, K, beta):
    x = np.random.normal(0, 1, size=(N, K))
    x = np.sign(x)
    u = np.matmul(Q, x)
    P = (1 + np.sign(u)) / 2
    H = -0.5 * np.sum(x * u, axis=0)
    
    aH = []
    
    for t in range(T):
        u = Q @ x
        P = (1 + np.sign(u)) / 2
        
        # Identify indices where P < 0.5
        valid_indices = np.where(P < 0.5)[0]
        
        # Select random indices from valid_indices
        if len(valid_indices) > 0:
            rid = np.random.choice(valid_indices, size=K, replace=True)
        else:
            # If no valid indices, fall back to random selection (although this case might be rare)
            #rid = np.random.randint(0, N, size=K)
            break

        rdm = np.random.uniform(0, 1, size=K)
        P_temp = P[rid, np.arange(len(rid))]
        x_temp = P_temp > rdm
        x[rid, np.arange(len(rid))] = 2 * x_temp - 1

    H = -0.5 * np.sum(x * np.matmul(Q, x), axis=0)
    
    return H / N, x.T


    
def run_SA_neal(N,Q,T,K,beta,anneal=False,x0=[]):

    nzi = np.nonzero(Q)
    
    irow = []
    icol = []
    qdata = []
    for i, j in zip(nzi[0], nzi[1]):
        irow.append(i)
        icol.append(j)
        qdata.append(Q[i, j])

    ldata = [0]*N
    #ldata = np.sign(np.random.uniform(-1,1,size=(N)))*10
    
    if len(x0)==0:
        x0 = np.sign(np.random.uniform(-1,1,size=(K,N))).astype(dtype=np.int8)
 
    num_sweeps_per_beta = 1
    beta_schedule = np.ones(T)*beta
    seed = np.random.randint(2**31)
    interrupt_function=None
    
    samples, energies = sa.simulated_annealing(
        K, np.array(ldata), np.array(irow), np.array(icol), np.array(qdata),
        num_sweeps_per_beta, beta_schedule,
        seed, x0, interrupt_function)
        
    return energies, samples

if __name__ == "__main__":
    
    T = 100
    K = 2
    N = 11
    
    Q, H0, gs, dp = lib.create_instance(N,Jtype='wishart')
 
    energies, solutions = run_SA(N,Q,T,K)

    p0 = np.mean(np.abs(energies-H0)<10**(-dp))
    
    print(energies,H0)
    print(p0)