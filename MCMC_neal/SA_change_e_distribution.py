import lib
import SA

import numpy as np
from tqdm import tqdm
import os

import networkx as nx

folder_name = "SA_change_e_distribution"
if not os.path.exists(folder_name):
    os.makedirs(folder_name)
    
import matplotlib.pyplot as plt
    
def simulate_ising_model(J, K, N, M, num_steps, beta):

    # Run the simulation on GPU
    if 1:
    
        # Create the new graph using Mlayer
        JM = lib.Mlayer(J, M, permute=True, GoG=True, typeperm='asym')
        #JM = lib.Mlayer(J, M, permute=True, GoG=False, typeperm='asym')
        #JM = lib.Mlayer(J, M, permute=True, GoG=False, typeperm='sym')
        #JM = lib.Mlayer(J, M, permute=True, GoG=True, typeperm='sym')
        #JM = lib.Mlayer_dense(J, M, permute=True)
        
        timeseq, final_spins = SA.run_SA(N*M,JM,num_steps,K,beta,SAcode=SAcode)
        #energies_replica, final_spins = SA.run_SA(N*M,JM,num_steps,K,beta,SAcode=SAcode,anneal=True)
    else:
        JM = lib.Mlayer(J, M, permute=False)
        x0 = np.sign(np.random.uniform(-1,1,size=(K,N*M))).astype(dtype=np.int8)
        for rep in range(1):
            timeseq, final_spins = SA.run_SA(N*M,JM,num_steps,K,beta,SAcode=SAcode,x0=x0)
            x0 = final_spins
            JM = lib.Mlayer(J, M, permute=True)
            
    if 0: #check correction between combinations
        spins_reshaped = final_spins.reshape(K, M, N)
        np.mean(spins_reshaped,1)
        np.sum(np.abs(np.mean(spins_reshaped,1)),1) #check diversity of spin confs
        
        
    energies_replica = lib.calculate_energy(final_spins,JM)/M/N
  
    # Verify if the final configuration is a local minimum
    is_minimum_replica = lib.is_local_minimum(final_spins, JM)
    #print(energies_replica,H0)

    # Calculate the energy for each replica
    energies = lib.calculate_energy_replicas(final_spins, J, M)
    
    # Check if each replica is a local minimum
    is_minimum_vector = lib.check_local_minima_replicas(final_spins, J, M)
    
    return final_spins, is_minimum_replica, energies_replica, energies, is_minimum_vector, timeseq, JM
    
if __name__ == "__main__":
      
    # Parameters
    if 0: #SK
    
        N_vector = [50]*1 # Number of spins
        M_vector = [1,2,5,10,20]
        K = 50  # Batch size per instance
        num_steps_per_M = 100
        num_J = 1
        Jtype = 'SK'
        
        ebins = np.linspace(-0.75,-0.45,50)
        
    if 1: #random regular
    
        N_vector = [30]*1 # Number of spins
        #M_vector = [1,2,5,10,20,50,100,200]
        M_vector = [1,5,20,100,200,400]
        K = 50  # Batch size per instance
        num_steps_per_M = 100
        num_J = 1
        Jtype = 'RR'
        
        ebins = np.linspace(-1.3,-0.8,50)
        

    folder_name = "SA_change_e_distribution_test"
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)  
   
    SAcode = 'neal' # 'py', 'neal'
    
    beta = 20000
    
    idx = 0#10 #instance index if load

    if True:
    
        res = []
        aaH0 = []
        
        for iN, N in enumerate(N_vector):
            
            print(f"Instance index {iN}")
            
            inst = lib.create_instance(N, Jtype=Jtype)
      
            timeseq_mat = []
            energies_mat = []
            JM_mat = []
    
            for M in tqdm(M_vector):
            
                H0_vec = []
                H1_vec = []
                
                timeseq_vec = []
                energies_vec = []
                JM_vec = []
    
                for r in range(num_J):
                    
                    J, H0, H1, gs, dp, aH0 = inst
                    H0_vec.append(H0)
                    H1_vec.append(H1)
                    
                    #if len(N_vector)>1: #load instance
                    #    file_path = os.path.join('SA_find_GS', f"J_{N}_{iN}.txt")
                    #else:
                    #    iN = idx
                    #    file_path = os.path.join('SA_find_GS', f"J_{N}_{iN}.txt")
                    #J = np.loadtxt(file_path)
                    
                    if False: #load same as Belief propagation
                        from scipy.sparse import coo_matrix
                        rows = []
                        cols = []
                        data = []
                        
                        with open(f"./../3-DCM_clean/DCM_Mlayer_conv/J_{N}.txt", "r") as f:
                            for line in f:
                                i, j, v = line.strip().split()
                                rows.append(int(i))
                                cols.append(int(j))
                                data.append(float(v))  # or int(v) if it's always int
                        
                        # Reconstruct sparse matrix
                        J_coo = coo_matrix((data, (rows, cols)), shape=(N, N)).tocsr()
                        J = np.array(J_coo.todense())
                       
                    if SAcode=='neal':
                        num_steps = num_steps_per_M * M  # Max number of steps
                    else:
                        num_steps = num_steps_per_M * N * M  # Max number of steps
                    
                    # Run the Ising model simulation
                    final_spins, is_minimum_replica, energies_replica, energies, is_minimum_vector, timeseq, JM = simulate_ising_model(J, K, N, M, num_steps, beta)
                
                    residual = energies - H0
                    isoptimal = (energies*N == H0*N)
                    isfirst = (energies*N == H1*N)
                    
                    timeseq_vec.append(timeseq.tolist())
                    energies_vec.append(energies.tolist())
                    
                    JM_vec.append(JM)
                    
                    #save to files (avoid memory blow-up)
                    file_path = os.path.join(folder_name, f"timeseq_{N}_{M}_{iN}_{r}.txt")
                    np.savetxt(file_path,timeseq.tolist())
                    file_path = os.path.join(folder_name, f"energies_{N}_{M}_{iN}_{r}.txt")
                    np.savetxt(file_path,energies.tolist())
                    #file_path = os.path.join(folder_name, f"JM_{N}_{M}_{iN}_{r}.txt")
                    #np.savetxt(file_path,JM)
                    
                #timeseq_mat.append(timeseq_vec)
                #energies_mat.append(energies_vec)
                #JM_mat.append(JM_vec)
             
            #res.append((N,energies_mat,H0_vec,H1_vec,timeseq_mat,JM_mat))

                
    ## FIGS
        
    if 1:
            
        ## FIG 1
        
        mv = []
        mv0 = []
        mean_de_mat = []
        std_de_mat = []
        for iN, N in enumerate(N_vector):
            
            if len(N_vector)==1:
                iN = idx
            elif 0:
                if iN==idx:
                    iN = iN + 1
                    continue
            elif 0:
                if iN==1:
                    iN = iN + 1
                    continue   
            elif 0:
                if iN==8 or iN==13:
                    iN = iN + 1
                    continue
                
            elif 1:
                if iN==0:
                    iN = iN + 1
                    continue
                
            plt.figure(figsize=(12, 4))
            
            plt.subplot(1,2,1)

            mean_energy_vec = []
            H_min_vec = []
            for M in M_vector:
                
                energies_mat = []
                
                for r in range(num_J):
                    
                    file_path = os.path.join(folder_name, f"energies_{N}_{M}_{iN}_{r}.txt")
                    energies_mat.append(np.loadtxt(file_path))
                    
                r = 2 #for the plot
                if Jtype == 'SK':
                    fac = np.sqrt(N)
                else:
                    fac = 1
                    
                energies = np.squeeze(energies_mat[0]).flatten()/fac
                
                hist, bin_edges = np.histogram(energies, bins=ebins)
                hist = hist / hist.max()  # Normalize by the maximum value

                #plt.bar(bin_edges[:-1], hist, width=np.diff(bin_edges), alpha=0.5, label=f'M={M}', align="edge")
                #plt.plot(bin_edges[:-1], hist, label=f'M={M}')
                plt.step(bin_edges[:-1], hist, where='mid', label=f'M={M}')
                
                mean_energy_vec.append(np.mean(energies_mat))
                H_min_vec.append(np.min(energies_mat))
                
            #first calculate min of H found (putative ground-state)
            H_min = np.min(H_min_vec)
            mean_e_vec = np.array(mean_energy_vec)/fac
            mean_de_vec = (mean_energy_vec-H_min)/fac
            mean_de_mat.append(mean_de_vec.tolist())
            
            Hval = np.sort(np.unique(np.array(energies_mat).flatten())) #based on last M value
            
            if 0: # plot evolution of energy levels
                
                iN = idx
            
                p_vec = []
                for M in M_vector:
                    
                    energies_mat = []
                    
                    for r in range(num_J):
                        
                        file_path = os.path.join(folder_name, f"energies_{N}_{M}_{iN}_{r}.txt")
                        energies_mat.append(np.loadtxt(file_path))
                        
                    p0 =  np.mean(np.array(energies_mat)*N==Hval[0]*N)
                    p1 =  np.mean(np.array(energies_mat)*N==Hval[1]*N)
                    p2 =  np.mean(np.array(energies_mat)*N==Hval[2]*N)
                    #p3 =  np.mean(np.array(energies_mat)==Hval[3])
                    
                    #p_vec.append([p0,p1,p2,p3])
                    p_vec.append([p0,p1,p2])
                    
                plt.figure(figsize=(8, 6))
                for i in range(3):
                    plt.plot(M_vector,np.array(p_vec)[:,i],label=f"p{i}")
                plt.legend()
                plt.xscale('log')
                
            
            #calculate variance of de
            std_de_vec = []
            for M in M_vector:
                
                energies_mat = []
                
                for r in range(num_J):
                    
                    file_path = os.path.join(folder_name, f"energies_{N}_{M}_{iN}_{r}.txt")
                    energies_mat.append(np.loadtxt(file_path))
                    
                std_inv = (np.array(energies_mat)-H_min)/fac
                std_inv[std_inv==0] = np.nan
                #std_inv = 1/std_inv
                std_de_vec.append(np.nanstd(std_inv))

            std_de_mat.append(std_de_vec)
            
            plt.legend()
                
            plt.title(f"Energy per site of $H_M$ (N={N}, id={iN})")
            plt.xlabel('E/N')
            plt.ylabel('P(E/N)')
        
            file_path = os.path.join(folder_name, f"fig1_{N}_{iN}.pdf")
            plt.savefig(file_path)
        
            iN = iN + 1
                
        ## FIG 2
        
        plt.subplot(1,2,2)
                    
        #plt.figure(figsize=(8, 6))
        #plt.plot(M_vector,mean_energy_vec,'-d')
        iN = 0
        for mean_de_vec, std_de_vec in zip(mean_de_mat, std_de_mat):
            
            #plt.plot(M_vector,np.array(mean_de_vec),'-d',label=f"{iN}")
            #plt.plot(M_vector,1/np.array(mean_de_vec),'-d',label=f"{iN}")
            plt.errorbar(M_vector,np.array(mean_de_vec),np.array(std_de_vec)*1.96/np.sqrt(K*num_J),label=f"instance {iN}",lw=0.5)
            #plt.errorbar(M_vector,1/np.array(mean_de_vec),std_de_vec,label=f"{iN}")
            iN = iN + 1
            
        plt.plot(M_vector,np.mean(mean_de_mat,0),'d-k',lw=3,label='mean over instances')
            
        plt.title(f"Residual energy $H_M$ (N={N})")
        plt.xlabel('M')
        plt.ylabel(r'$<\Delta E/N>$')
        plt.xscale('log')
        plt.yscale('log')
        #plt.ylim(0,np.nanmax(1/np.array(de_mat)))
        #plt.ylim(0,300)
        plt.ylim(0.001,1)
        plt.xlim(np.min(M_vector),np.max(M_vector))
        #plt.legend(ncols=3)
        
        if 0:
            sp = np.polyfit(np.log(M_vector)[4:],np.log(np.mean(mean_de_mat,0))[4:],1)
            
            plt.plot(M_vector,np.exp(sp[0]*np.log(M_vector)+sp[1]),'--k',lw=1)
             
        file_path = os.path.join(folder_name, f"fig2_{N}d.pdf")
        plt.savefig(file_path)
        
    if True:
        
        plt.figure(figsize=(8, 4))
        
        plt.subplot(1,1,1)
        
        plt.plot(M_vector,mean_e_vec*fac,'o-')
        
        plt.ylabel('<E/N>')
        plt.xlabel('M')
    
        