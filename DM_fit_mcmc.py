# import pygrc as gr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import emcee
import corner
# from IPython.display import display, Math

file = "C:/Users/admin/OneDrive/Desktop/Other/Oxford UROP 2024/SPARC_Lelli2016c.mrt.txt"
SPARC_c = [ "T", "D", "e_D", "f_D", "Inc", "e_Inc",
            "L", "e_L", "Reff", "SBeff", "Rdisk",
            "SBdisk", "MHI", "RHI", "Vflat", "e_Vflat", "Q", "Ref."]
table = pd.read_fwf(file, skiprows=98, names=SPARC_c)
# print(table["Ref."]["CamB"])

# galaxy = [ "NGC5055", "NGC5585",
#             "NGC6015", "NGC6946", "NGC7331" ]
galaxy = [ "NGC6946" ]

columns = [ "Rad", "Vobs", "errV", "Vgas",
            "Vdisk", "Vbul", "SBdisk", "SBbul" ]

# Define constants
G = 4.300e-6    # Gravitational constant G (kpc (km/s)^2 solarM^(-1))

# Equation for dark matter halo velocity
def halo_v(r, rho0, rc):
    x = r / rc
    # v = np.sqrt(4*np.pi*G*rho0*rc**2*(1 - rc/r * np.arctan(r/rc))) # Pseudo-isothermal profile
    v = np.sqrt(4*np.pi*G*rho0*rc**3*(np.log(1+x) - x/(1+x))/r) # NFW profile
    # v = v.fillna(0)
    return v

for g in galaxy:
    print("=================================")
    print("Fitting halo to galaxy "+g+":")
    print("---------------------------------")
    
    D = table["D"][g] # Distance to galaxy (error = e_D)
    e_D = table["e_D"][g]
    Inc = table["Inc"][g] # Inclination in degrees (error = e_Inc)
    e_Inc = table["e_Inc"][g]
    
    file_path = "C:/Users/admin/OneDrive/Desktop/Other/Oxford UROP 2024/data/"+g+"_rotmod.dat"
    rawdata = np.loadtxt(file_path)
    data = pd.DataFrame(rawdata, columns=columns)
    bulged = np.any(data["Vbul"]>0) # Check whether galaxy has bulge.
    print("Bulged = "+str(bulged))
    r = data["Rad"]
    
    # Total velocity; wDM = True to include dark matter halo component, wDM = false otherwise.
    def Vpred(theta, wDM=True):
        if bulged:
            d, inc, rho0, rc, pdisk, pbul = theta
        else:
            d, inc, rho0, rc, pdisk = theta
            pbul = 0.0
        v_sq = data["Vgas"]**2 + data["Vdisk"]**2*pdisk + data["Vbul"]**2*pbul
        v_sq *= d / D
        if wDM:
            v_sq += halo_v(r, rho0, rc)**2
        v = np.sqrt(v_sq)
        v *= np.sin(np.deg2rad(Inc)) / np.sin(np.deg2rad(inc))
        return v
    
    def lnlike(theta, r, v, v_err):
        return -0.5 * np.sum(((v - Vpred(theta))/v_err) ** 2)
    
    def lnprior(theta):
        if bulged:
            d, inc, rho0, rc, pdisk, pbul = theta
            if pbul < 0.0:
                return -np.inf       
        else:
            d, inc, rho0, rc, pdisk = theta
        
        # All parameters are non-negative.
        if rho0 < 0.0 or rc < 0.0 or pdisk < 0.0:
            return -np.inf
        
        # Truncated Gaussians for D (distance) and inc (inclination)
        if d < 0.0 or inc < 15.0 or inc > 150.0:
            return -np.inf
        prior_D = np.log(1.0/(np.sqrt(2*np.pi)*e_D))-0.5*(d-D)**2/e_D**2
        prior_Inc = np.log(1.0/(np.sqrt(2*np.pi)*e_Inc))-0.5*(inc-Inc)**2/e_Inc**2
        
        # Gaussian prior on pdisk and pbul, with center 0.5, 0.7 respectively.
        pdisk_mu = 0.5
        pbul_mu = 0.7
        p_sigma = 0.1
        prior_pdisk = np.log(1.0/(np.sqrt(2*np.pi)*p_sigma))-0.5*(pdisk-pdisk_mu)**2/p_sigma**2
        
        prior = prior_D + prior_Inc + prior_pdisk
        if bulged:
            prior_pbul = np.log(1.0/(np.sqrt(2*np.pi)*p_sigma))-0.5*(pbul-pbul_mu)**2/p_sigma**2
            prior += prior_pbul
        
        return prior
        
    def lnprob(theta, r, v, v_err):
        lp = lnprior(theta)
        if not np.isfinite(lp):
            return -np.inf
        return lp + lnlike(theta, r, v, v_err)
    
    def MCMCfit(p0,nwalkers,niter,ndim,lnprob,data):
        sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=data)
    
        print("Running burn-in...")
        p0, _, _ = sampler.run_mcmc(p0, 500, progress=True)
        sampler.reset()
    
        print("Running production...")
        pos, prob, state = sampler.run_mcmc(p0, niter, progress=True)
    
        return sampler, pos, prob, state
    
    fit_data = ( r, data["Vobs"], data["errV"] )
    nwalkers = 128
    niter = 500
    initial = np.array([D, Inc, 2e+6, 15, 0.5])
    if bulged:
        initial = np.append(initial, [0.7])
    print("initial = "+str(initial))
    ndim = len(initial)
    p0 = [np.array(initial) + 1e-7 * np.random.randn(ndim) for i in range(nwalkers)]
    sampler, pos, prob, state = MCMCfit(p0,nwalkers,niter,ndim,lnprob,fit_data)
    samples = sampler.flatchain
    
    labels = ['d', 'inc', 'rho0', 'rc', 'pdisk']
    if bulged:
        labels.append('pbul')
        
    # fig_test, axes = plt.subplots(ndim, sharex=True)
    # samples_test = sampler.get_chain()
    # for i in range(ndim):
    #     ax = axes[i]
    #     ax.plot(samples_test[:, :, i], "k", alpha=0.3)
    #     ax.set_xlim(0, len(samples_test))
    #     ax.set_ylabel(labels[i])
    #     ax.yaxis.set_label_coords(-0.1, 0.5)
    
    # axes[-1].set_xlabel("step number")
    # fig_test.savefig("C:/Users/admin/OneDrive/Desktop/Other/Oxford UROP 2024/plots/DM_NFW/MCMC/time_series_"+g+".png", dpi=300, bbox_inches="tight")
    
    # tau = sampler.get_autocorr_time()
    # print(tau)
    
    theta_max = samples[np.argmax(sampler.flatlnprobability)]
    print('Theta max: ', theta_max)
    
    def test_plotter(sampler):
        print("Generating test plot...")
        plt.ion()
        plt.errorbar(r, data["Vobs"], yerr=data["errV"], fmt=".", capsize=3, label="Total curve - observed")
        for theta in samples[np.random.randint(len(samples), size=100)]:
            plt.plot(r, Vpred(theta), color="r", alpha=0.1)
        plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
        plt.xlabel('Radius (kpc)')
        plt.ylabel('Velocities (km/s)')
        plt.legend()
        plt.show()
        
    # test_plotter(sampler)
    
    def sample_walkers(nsamples, flattened_chain):
        models = []
        draw = np.floor(np.random.uniform(0,len(flattened_chain),size=nsamples)).astype(int)
        thetas = flattened_chain[draw]
        for i in thetas:
            mod = Vpred(i)
            models.append(mod)
        spread = np.std(models,axis=0)
        med_model = np.median(models,axis=0)
        return med_model, spread
    
    med_model, spread = sample_walkers(100, samples)
    
    # for i in range(ndim):
    #     mcmc = np.percentile(samples[:, i], [16, 50, 84])
    #     q = np.diff(mcmc)
    
    """
    Plotting galaxy rotation curves directly from data with variables:
    Vobs (overall observed, corrected for inclination), Vgas, Vdisk, Vbul.
    """
    plt.title(g)
    plt.errorbar(r, data["Vobs"], yerr=data["errV"], fmt=".", capsize=3, label="Total curve - observed")
    plt.xlabel('Radius (kpc)')
    plt.ylabel('Velocities (km/s)')
    
    rho0 = theta_max[2]
    rc = theta_max[3]
    pdisk = theta_max[4]
    if bulged:
        pbul = theta_max[5]
    
    plt.plot(r, data["Vgas"], label="Gas")
    plt.plot(r, data["Vdisk"]*np.sqrt(pdisk), label="Stellar disc")
    if bulged:
        plt.plot(r, data["Vbul"]*np.sqrt(pbul), label="Bulge")
    plt.plot(r, Vpred(theta_max, wDM=False), linestyle="dashed", color="grey", label="Total curve W/O DM")
    plt.plot(r, halo_v(r, rho0, rc), label="Dark matter halo - best fit")
    plt.plot(r, Vpred(theta_max), color="black", label="Total curve - best fit")
    plt.fill_between(r, med_model-spread, med_model+spread, color='yellow', alpha=0.5, label=r'$1\sigma$ posterior spread')
    plt.legend(bbox_to_anchor=(1,1), loc="upper left")
    
    plt.savefig("C:/Users/admin/OneDrive/Desktop/Other/Oxford UROP 2024/plots/DM_NFW/MCMC/test_"+g+".png", dpi=300, bbox_inches="tight")
    plt.show()
    plt.close()
    
    fig = corner.corner(samples, show_titles=True, labels=labels, plot_datapoints=True, quantiles=[0.16, 0.5, 0.84], smooth=1)
    fig.savefig("C:/Users/admin/OneDrive/Desktop/Other/Oxford UROP 2024/plots/DM_NFW/MCMC/test_"+g+"_corner.png", dpi=300, bbox_inches="tight")
    # plt.close(fig)
    
    print("=================================")
