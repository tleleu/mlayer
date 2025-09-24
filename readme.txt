file					description
#############################################################################################################

/RDE
/RDE/complexity_B2.py			1-RSB cavity RDE structured M-layer, free energy, complexity vs. B
/RDE/RDE_fixe10.py			cavity dynamics (fixed)
/RDE/G90.py				comparison to Mezard 2003 theory
/RDE/plot_complexity.py			plot complexity from complexity_B2.py

/RDE/overlap.py				computes block-block overlap as a function of sigma using cavity RDE
/RDE/contraction.py			theoretical value for contraction threshold
/RDE/overlap_pop.py			idem as overlap but vs. population size (check finite size)

/RDE/isotropy.py			isotropy from theory
/RDE/isotropy_empirical.py		isotropy empirical from cavity RDE directly
/RDE/plot_isotropy.py			plot isotropy theory vs. empirical

/RDE/misc


/MCMC
/MCMC/MCMC.py				old version run MCMC

/MCMC/energy.py				residual energy vs. sigma and M 
/MCMC/plot_MCMCenergy.py		plot residual vs. sigma / M

/MCMC/threshold.py			threshold of spectral gap for residual energy and inter-layer overlap 
/MCMC/instance.py			generation of instance
/MCMC/plotQ.py				picture of the block structure mixing matrix
/MCMC/plot_contraction.py		plot superimposed contraction MCMC and cavity
/MCMC/stats.py				computes statistics on spin level

/MCMC/mlayer.py				generation of mlayer (naive)
/MCMC/mlayer2.py			generation of mlayer (more rigorous)

/MCMC/ising_MCMC_sparse.pyx		accelerated zero temperature spin-level MCMC
/MCMC/setup_sparse.py			compilation ising_MCMC_sparse.pyx
/MCMC/sa_sparse_ising.pyx		accelerated SA
/MCMC/setup_sa_sparse_ising.py		compilation sa_sparse_ising.pyx



/_draft					reference code
/_draft/MCMC				python MCMC code for Mlayer
/_draft/BP				DCM BP applied to M-layer (old theory)				




/MCMC_neal/				reference code for SA using neal
/MCMC_neal/SA.py			interface with neal

/MCMC_neal/instance.py			misc
/MCMC_neal/lib.py			misc

/MCMC_neal/energy			same as /MCMC/energy but using neal
/MCMC_neal/energy2			idem but with tunable kernel

/MCMC_neal/SA_change_e_distribution	original scaling with neal (works well)
/MCMC_neal/scaling			new scaling with neal (using mlayer2)

/MCMC_neal/temp_compare			checking the difference between the two mlayer kernels



other codes of interest:
#############################################################################################################

2_code_simplified/HiLD2025/		plot for conference paper with asymmetric BP and backaction

2_code_simplified/BP/			sanity check for RDE 1RSB, BP on Mlayer, backaction
						> The idea here was to apply BP (or DCM) directly to Mlayer
						> Based on DCM ideas, or Monatanari style message passing
2_code_simplified/BP/wrap_BP		BP RDE directly on asymmetric M-layer
2_code_simplified/BP/wrap_BP_backaction	BP RDE on original graph with correction due to asymmetry

2_code_simplified/MCMC/			MCMC of Mlayer in python
2_code_simplified/MCMC_neal/		faster code for MCMC in c++
2_code_simplified/utils/		utilities for Mlayer

2_code/					initial tests on Mlayer
						> initial tests
						> reproduce Mezard 2003 cavity theory
						> reproducing DCM theory as well

