import numpy as np
import matplotlib.pyplot as plt      


def compute_values(mu):
    mu = np.asarray(mu)
    numerator_p0 = 24 + 3 * np.exp(2 * mu) - (8 + np.exp(mu)) * np.sqrt(8 + np.exp(2 * mu))
    denominator_p0 = 4 * (np.exp(mu) - 1) ** 2
    p0 = numerator_p0 / denominator_p0
    numerator_q = p0 * (3 * p0 - 2)
    denominator_q = np.exp(mu) - 4 * p0 - 2 * p0 * np.exp(mu)
    q = numerator_q / denominator_q
    p1 = (2 * p0 * q) / (p0 + 2 * q) ** 2
    p2 = q ** 2 / (p0 + 2 * q) ** 2
    return p0, p1, p2, q


def c0_(mu):
    e_mu = np.exp(mu)
    return (e_mu+2-np.sqrt((e_mu-4)**2 + 8*(e_mu-1)))/2/(e_mu-1)

def c1_(mu):
    return (1-c0_(mu))/2

def I1_(mu):
    c0 = c0_(mu)
    c1 = c1_(mu)
    e_mu = np.exp(mu)
    return c0**3+6*c0*c1**2+6*e_mu*(c0**2*c1+c1**3)+6*np.exp(2*mu)*c0*c1**2+2*np.exp(3*mu)*c1**3
    
def I2_(mu):
    c0 = c0_(mu)
    c1 = c1_(mu)
    e_mu = np.exp(mu)
    return e_mu*(c0**2+2*c1**2)**2+8*e_mu*c0**2*c1**2+8*np.exp(2*mu)*c0*c1*(c0**2+2*c1**2)+8*np.exp(2*mu)*c0*c1**3

def I3_(mu):
    c0 = c0_(mu)
    c1 = c1_(mu)
    e_mu = np.exp(mu)
    return 4*np.exp(3*mu)*c1**2*(c0**2+2*c1**2)+8*np.exp(3*mu)*c0**2*c1**2+2*np.exp(3*mu)*c1**4+8*np.exp(4*mu)*c0*c1**3+2*np.exp(5*mu)*c1**4
    
def phi_G90(mu):
    I1 = I1_(mu)
    I2 = I2_(mu)
    I3 = I3_(mu)
    return 2/mu*np.log(I1) - 3/2/mu*np.log(I2+I3)

if __name__ == "__main__":
    
    Ny = np.linspace(0.0001,1,100)
    
    phi_the = []
    for y in Ny:
        phi_the.append(phi_G90(y))
        
    fig, ax = plt.subplots(1, 2, figsize=(15, 5))
    ax[0].plot(Ny,phi_the,'-k')
    
    ymax = np.max(Ny)
    ymin = np.min(Ny)
    ax[0].set_ylim(-1.278, -1.27); ax[0].set_xlim(ymin, ymax)
    ax[0].set(xlabel="Parisi parameter y", ylabel="Energy per site e")
    
    # Mezard2003 eq. 44: d(mu phi) / dmu = epsilon, d(phi)/dmu = Sigma(epsilon)/mu^2
    
    mu_phi = phi_the * Ny
    
    epsilon = np.gradient(mu_phi, Ny)
    Sigma = Ny**2 * np.gradient(phi_the, Ny)
    
    ax[1].plot(epsilon,Sigma,'-k')
    
    ax[1].set_ylim(0, 0.0008); ax[1].set_xlim(-1.278, -1.27)
    ax[1].set(xlabel="Energy per site e", ylabel="complexity Σ")
   
    
    