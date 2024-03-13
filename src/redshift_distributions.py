import numpy as np
from scipy.integrate import simps
from scipy.stats import rv_histogram
import warnings


def p_ph_z_p(z_p, z, f_out = 0., sigma_b = 0.05, sigma_o = 0.05, c_b = 1.0, c_o = 1.0, z_b = 0.0, z_o = 0.1):
    # Convolve the true redshift distribution with the probability function of a photometric redshift given the true redshift.
    # This is based on https://arxiv.org/abs/1910.09273 Eq (115).
    # For LSST a simple Gaussian with z_p as mean and sigma_z should be used (so just f_out=0.0 instead of 0.1).
    return (1-f_out) / (np.sqrt(2*np.pi)*sigma_b*(1+z)) * \
            np.exp(-0.5*((z-c_b*z_p-z_b)/(sigma_b*(1+z)))**2) + \
            f_out/(np.sqrt(2*np.pi)*sigma_o*(1+z)) * \
            np.exp(-0.5*((z-c_o*z_p-z_o)/(sigma_o*(1+z)))**2)


def nz_phot(z, nz, zi_m, zi_p, sigma_z=0.05):
    z_phot = np.linspace(zi_m, zi_p, num=128)
    nominator = simps(nz * p_ph_z_p(z_p=z_phot.reshape(len(z_phot),1),
                                    z=z.reshape(1, len(z)), sigma_b=sigma_z),
                      z_phot, axis=0)
    denominator = simps(nominator, z)
    return nominator/denominator


def get_redshift_distributions(z_arr, N_bins, z0, a, sigma_z, zbin_edges=None):
    if np.any(z_arr<=0):
        warnings.warn('The redshift array provided contains 0 or lower values.'
                      'Bin edges will be below 0!')
    delta_z_step = z_arr[1]-z_arr[0]
    z_bins = np.append(z_arr, z_arr[-1]+delta_z_step)-delta_z_step/2.
    #from scipy.special import gamma
    #norm = 3 / (z0**3 * gamma((3+a)/a)) # Binned distributions are normalised below.
    pz = z_arr ** 2 * np.exp(-(z_arr / z0) ** a)

    if zbin_edges is None:
        # Find a number of redshift bins that contains equal number of galaxies.
        # Create a histogram object with z_bins as bins and pz as value between bins.
        hist_dist = rv_histogram((pz, z_bins))
        quantiles = np.linspace(0, 1, N_bins + 1)
        zbin_edges = hist_dist.ppf(quantiles).round(2)

    dndz_bin = np.empty((N_bins, len(z_arr)))
    for i in range(N_bins):
        dndz_bin[i] = nz_phot(z_arr, pz, zi_m=zbin_edges[i], zi_p=zbin_edges[i+1], sigma_z=sigma_z)
    return dndz_bin, zbin_edges
