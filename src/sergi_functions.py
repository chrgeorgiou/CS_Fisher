import numpy as np
import pyccl as ccl
from scipy.interpolate import interp1d
from matplotlib.patches import Ellipse
from scipy.stats import chi2
import warnings
from .utils import sigma8_derivative, names_to_latex

# ----------------------------------------------------------------------------------------------------------------------------------------------
# Do we want windowed or unwindowed Pk??? Note add the negligible Pks

def get_pk_halomod(cosmo: ccl.Cosmology,
                   A_IA: float,
                   k: np.ndarray = None,
                   a_arr: np.ndarray = None,
                   b: float = -2,
                   a1h: float = 0.001,
                   hm_def: str = '200m',   
                   C1rhocrit: float = 5e-14*ccl.physical_constants.RHO_CRITICAL) -> tuple:

    """
    Computes the Intrinsic-Intrinsic (II) and Gravitational-Intrinsic (GI) power spectra
    using a halo model formalism with a satellite galaxy shear HOD model for intrinsic alignments.
    
    Based on Chr. Georgiou's example: "Intrinsic alignments power spectra using the halo model"
    (https://github.com/LSSTDESC/CCLX/blob/master/Halo%20model%20for%20IA.ipynb).

    Args:
        cosmo (ccl.Cosmology): CCL Cosmology object.
        A_IA (float): Amplitude parameter for intrinsic alignments (NLA normalization).
        k (np.ndarray): Wavenumber array [h/Mpc], must be log-spaced. 
                        If None, defaults to 128 points between 1e-3 and 1e3 [h/Mpc].
        a_arr (np.ndarray): Scale factor array (corresponding to redshifts of interest).
                            If None, defaults to 32 points linearly spaced from a=0.1 to a=1.
        b (float, optional): Slope parameter for the satellite shear HOD profile. Defaults to -2.
        a1h (float, optional): Amplitude parameter for the 1-halo satellite shear term. Defaults to 0.001.
        hm_def (str, optional): Halo mass definition (e.g., '200m', 'vir'). Defaults to '200m'.
        C1rhocrit (float, optional): Normalization factor for the IA power spectrum.
                                     Defaults to 0.0134.

    Returns:
        tuple:
            pk_II_total (ccl.pk2d.Pk2D): Total II power spectrum (1-halo + 2-halo terms).
            pk_GI_total (ccl.pk2d.Pk2D): Total GI power spectrum (1-halo + 2-halo terms).

    Notes:
        - The centra-satelite, central-central contributions from the II power spectra are not 
          considered in their 1-halo term...
        - The 1-halo central term from the GI power spectra is not considered...
    """
    if k is None:
        k = np.geomspace(1E-3, 1e3, 128) 
    lk = np.log(k)

    if a_arr is None:
        a_arr = np.linspace(0.1, 1, 32)

    #Initialize halo model quantities:

    # the Duffy 2008 concentration-mass relation,
    cM = ccl.halos.ConcentrationDuffy08(mass_def=hm_def)
    # the Tinker 2010 halo mass function,
    nM = ccl.halos.MassFuncTinker10(mass_def=hm_def)
    # the Tinker 2010 halo bias,
    bM = ccl.halos.HaloBiasTinker10(mass_def=hm_def)
    # and the halo model calculator object.
    hmc = ccl.halos.HMCalculator(mass_function=nM, halo_bias=bM, mass_def=hm_def)
    # generate satellite shear HOD
    sat_gamma_HOD = ccl.halos.SatelliteShearHOD(concentration=cM, mass_def=hm_def, a1h=0.001, b=-2)

    # COMPUTE II P(K)
    pk_II_1h_ss = ccl.halos.halomod_Pk2D(cosmo, hmc, sat_gamma_HOD, get_2h = False, a_arr=a_arr, lk_arr=lk)
    pk_II_2h_ss = ccl.halos.halomod_Pk2D(cosmo, hmc, sat_gamma_HOD, get_1h=False, a_arr=a_arr, lk_arr=lk)
    
    # Compute the 2-halo c-s term:
    C = A_IA * C1rhocrit * cosmo['Omega_m'] / cosmo.growth_factor(a_arr)
    C_pk_lin = ccl.pk2d.Pk2D(a_arr=a_arr, lk_arr=lk,
                             pk_arr=C.reshape(-1,1)*cosmo.linear_matter_power(np.e**lk, a_arr),
                             is_logp=False)
    pk_b_gamma = -1 * ccl.pk2d.Pk2D(a_arr=a_arr, lk_arr=lk,
                                    pk_arr=ccl.halos.halomod_bias_1pt(cosmo, hmc, np.e**lk, a_arr,
                                                                      sat_gamma_HOD), is_logp=False)
    pk_II_2h_cs = C_pk_lin * pk_b_gamma
    
    pk_II_2h_cc = ccl.pk2d.Pk2D(a_arr=a_arr, lk_arr=lk,
                                pk_arr=C.reshape(-1,1)**2*cosmo.linear_matter_power(np.e**lk, a_arr),
                                is_logp=False)
    pk_II_total = pk_II_1h_ss + pk_II_2h_ss + pk_II_2h_cs + pk_II_2h_cc # + pk_II_1h_cs + pk_II_1h_cc (first negligible? and second doesnt exist?)

    # COMPUTE GI P(K)
    # NFW profile for matter (G)
    NFW =  ccl.halos.HaloProfileNFW(mass_def=hm_def, concentration=cM, truncated=True, fourier_analytic=True)
    
    pk_GI_1h_s = ccl.halos.halomod_Pk2D(cosmo, hmc, NFW, prof2 = sat_gamma_HOD, get_2h = False, a_arr=a_arr, lk_arr=lk)
    pk_GI_2h_s = ccl.halos.halomod_Pk2D(cosmo, hmc, NFW, prof2 = sat_gamma_HOD, get_1h = False, a_arr=a_arr, lk_arr=lk)
    pk_GI_2h_c = -1*C_pk_lin
    pk_GI_total = pk_GI_1h_s + pk_GI_2h_s + pk_GI_2h_c # + pk_GI_1h_c (doesnt exist I think? need to justify and above too)

    return pk_II_total, pk_GI_total

# ----------------------------------------------------------------------------------------------------------------------------------------------     # specificly say that k array has to be log?? New parameters need to be added to configuration or elsewhere
 # like b, a1h, hm_def, C1rhocrit, (k and a_arr ??) or do we want them to be always te same?
            
def get_Cell_data_vector_halomod(cosmo: ccl.Cosmology,
                         z: np.ndarray, dndz: np.ndarray,
                         pk_II: ccl.pk2d.Pk2D, pk_GI: ccl.pk2d.Pk2D,
                         ell: np.ndarray = None,
                         include_GI: bool = True,
                         include_II: bool = True,
                         include_GG: bool = True) -> dict:
    """
    Computes the total cosmic shear angular power spectra (C_ell) including gravitational 
    lensing (GG), intrinsic-intrinsic alignments (II), and gravitational-intrinsic 
    alignments (GI), using a halo model-based prescription for intrinsic alignments.

    Args:
        cosmo (ccl.Cosmology): A CCL cosmology object.
        z (np.ndarray): Redshift values where the n(z) is defined.
        dndz (np.ndarray): Redshift distribution(s). Shape should be (n_bins, len(z)) or (len(z),).
        pk_II, pk_GI (ccl.pk2d.Pk2D): II and GI contributions of the power spectrum computed using the
                                   get_pk_halomod function.
        ell (np.ndarray): Multipole values at which to compute C_ell.

        include_XX (bool): Whether to include the XX (GI, II, GG) term. Default: True.

    Returns:
        dict: Dictionary of angular power spectra keyed by redshift bin pair labels ('z0-z0', 'z0-z1', ...).
              Each value is the total C_ell (GG + GI + II) for that bin pair.
    """
        
    dndz_use = np.atleast_2d(dndz) # if 1D array, gets converted to (1,N) (N len of dndz)
    n_z_bins = dndz_use.shape[0] # number of bins
    
    b_IA = np.ones(len(z)) # A_IA = 1 in the NLA model

    cl_total = {}

    for z1 in range(n_z_bins):
        for z2 in range(n_z_bins):
            if z2 < z1: continue
            # Tracers without shear and A_ia contribution since the alignment signal is embedded in the power spectrum.
            ia_tracer1 = ccl.WeakLensingTracer(cosmo, dndz = (z, dndz_use[z1]), has_shear=False, ia_bias = (z, b_IA), use_A_ia=False) 
            ia_tracer2 = ccl.WeakLensingTracer(cosmo, dndz = (z, dndz_use[z2]), has_shear=False, ia_bias = (z, b_IA), use_A_ia=False)
            # Weak gravitational lensing tracers for the GG and GI terms.
            wl_tracer1 = ccl.WeakLensingTracer(cosmo,dndz = (z, dndz_use[z1]))
            wl_tracer2 = ccl.WeakLensingTracer(cosmo,dndz = (z, dndz_use[z2]))

            cl_IG = ccl.angular_cl(cosmo, ia_tracer1, wl_tracer2, ell, p_of_k_a = pk_GI) if include_GI else 0 
            cl_GI = ccl.angular_cl(cosmo, wl_tracer1, ia_tracer2, ell, p_of_k_a = pk_GI) if include_GI else 0 
            cl_II = ccl.angular_cl(cosmo, ia_tracer1, ia_tracer2, ell, p_of_k_a = pk_II) if include_II else 0
            cl_GG = ccl.angular_cl(cosmo, wl_tracer1, wl_tracer2, ell) if include_GG else 0 # add pk default
            cl_total[f'z{z1}-z{z2}'] = cl_GG + cl_GI + cl_IG + cl_II 

    return cl_total

# ----------------------------------------------------------------------------------------------------------------------------------------------

def get_pk_halomod_windowed(cosmo: ccl.Cosmology,
                   A_IA: float,
                   k: np.ndarray = None,
                   a_arr: np.ndarray = None,
                   b: float = -2,
                   a1h: float = 0.001,
                   hm_def: str = '200m',   
                   C1rhocrit: float = 5e-14*ccl.physical_constants.RHO_CRITICAL) -> tuple:
    
    if k is None:
        k = np.geomspace(1E-3, 1e3, 128) 
    lk = np.log(k)
    
    if a_arr is None:
        a_arr = np.linspace(0.1, 1, 32)
    
    C = A_IA * C1rhocrit * cosmo['Omega_m'] / cosmo.growth_factor(a_arr)
    
    k1h = 4*cosmo['h'] 
    k2h = 6*cosmo['h'] 
    
    #Compute II and GI P(k) terms with NLA 
    
    pk_II_NLA_windowed = ccl.pk2d.Pk2D(a_arr=a_arr, lk_arr=lk,
                                       pk_arr=C.reshape(-1,1)**2*cosmo.nonlin_matter_power(np.e**lk, a_arr)*np.exp(-(k/k2h)**2).reshape(1,-1),
                                       is_logp=False)
    pk_GI_NLA_windowed = ccl.pk2d.Pk2D(a_arr=a_arr, lk_arr=lk,
                                       pk_arr=-C.reshape(-1,1)*cosmo.nonlin_matter_power(np.e**lk, a_arr)*np.exp(-(k/k2h)**2).reshape(1,-1),
                                       is_logp=False)
    
    #Initialize halo model quantities:
    
    # the Duffy 2008 concentration-mass relation,
    cM = ccl.halos.ConcentrationDuffy08(mass_def=hm_def)
    # the Tinker 2010 halo mass function,
    nM = ccl.halos.MassFuncTinker10(mass_def=hm_def)
    # the Tinker 2010 halo bias,
    bM = ccl.halos.HaloBiasTinker10(mass_def=hm_def)
    # and the halo model calculator object.
    hmc = ccl.halos.HMCalculator(mass_function=nM, halo_bias=bM, mass_def=hm_def)
    # generate satellite shear HOD
    sat_gamma_HOD = ccl.halos.SatelliteShearHOD(concentration=cM, mass_def=hm_def, a1h=0.001, b=-2)
    
    # COMPUTE II P(K) term with halo model 
    pk_II_1h_ss = ccl.halos.halomod_Pk2D(cosmo, hmc, sat_gamma_HOD, get_2h = False, a_arr=a_arr, lk_arr=lk)
    
    pk_II_1h_ss_windowed = ccl.pk2d.Pk2D(a_arr=a_arr, lk_arr=lk,
                                         pk_arr=pk_II_1h_ss(k, a_arr)*(1-np.exp(-(k/k1h)**2)).reshape(1,-1),
                                         is_logp=False)
    
    # NFW profile for matter (G)
    NFW =  ccl.halos.HaloProfileNFW(mass_def=hm_def, concentration=cM, truncated=True, fourier_analytic=True)
    
    # COMPUTE GI P(K) term with halo model 
    pk_GI_1h_s = ccl.halos.halomod_Pk2D(cosmo, hmc, NFW, prof2 = sat_gamma_HOD, get_2h = False, a_arr=a_arr, lk_arr=lk)
    
    pk_GI_1h_s_windowed = ccl.pk2d.Pk2D(a_arr=a_arr, lk_arr=lk,
                                         pk_arr=pk_GI_1h_s(k, a_arr)*(1-np.exp(-(k/k1h)**2)).reshape(1,-1),
                                         is_logp=False)
    
    # NLA for large scales and halo-model for small scales?
    pk_II_windowed = pk_II_NLA_windowed + pk_II_1h_ss_windowed
    pk_GI_windowed = pk_GI_NLA_windowed + pk_GI_1h_s_windowed
    
    return pk_II_windowed, pk_GI_windowed


# ----------------------------------------------------------------------------------------------------------------------------------------------
# General unpacking function

def unpack_array_params(param_dict: dict) -> dict:
    '''
    Unpacks parameters in a dictionary if any of the fiducial entries 
    are lists or arrays (e.g., for parameters like 'm_nu':[0.01, 0.02, 0.03] with individual components).
    If just one shift provided for something like m_nu = [m1, m2, m3], assumes same shift for each unpacked parameter.

    Args:
        param_dict (dict): Dictionary like cosmo/astro/redshift_params containing keys like 'name', 'fiducial', 
                           'shift', 'latex', and optionally others (e.g., 'mass_split').

    Returns:
        new_dict (dict): The same dictionary with any list-valued parameters unpacked into 
              individual entries (e.g., 'm_nu1', 'm_nu2', 'm_nu3').
    '''
    new_dict = {k: [] for k in ['name', 'fiducial', 'shift', 'latex']}

    for i, name in enumerate(param_dict['name']):
        fiducial = param_dict['fiducial'][i]
        shift = param_dict['shift'][i]
        latex = param_dict['latex'][i]

        # If the fiducial is a list or tuple, unpack it
        if isinstance(fiducial, (list, tuple)):
            for j, val in enumerate(fiducial):
                new_dict['name'].append(f"{name}{j+1}")
                new_dict['fiducial'].append(val)
                if isinstance(shift, (list, tuple)):
                    new_dict['shift'].append(shift[j])
                elif shift is not None:
                    new_dict['shift'].append(shift)
                else:
                    new_dict['shift'].append(None)
                new_dict['latex'].append(f"{latex}_{j+1}")
        else:
            # Otherwise, just append the regular scalar
            new_dict['name'].append(name)
            new_dict['fiducial'].append(fiducial)
            new_dict['shift'].append(shift)
            new_dict['latex'].append(latex)

    # Preserve any extra fields, like 'mass_split'
    for key in param_dict:
        if key not in new_dict:
            new_dict[key] = param_dict[key]

    return new_dict
# ----------------------------------------------------------------------------------------------------------------------------------------------
# Neutrino focused packing and unpacking functions --> OPTIONALLY DO ALL THIS READING THE MASS_SPLIT !!!
''' Type of input:
{'name': ['Omega_m', 'Omega_b', 'h', 'A_s', 'n_s', 'w0', 'wa', 'm_nu'], 'fiducial': [0.315, 0.049, 0.674, 2.0989e-09, 0.965, -1, 0, [0.01, 0.02]], 'shift': [0.003, None, 0.002, 0.002, None, None, None, [0.33, 0.44]], 'latex': ['$\\Omega_\\mathrm{m}$', '$\\Omega_\\mathrm{b}$', '$h$', '$A_\\mathrm{s}$', '$n_\\mathrm{s}$', '$w_0$', '$w_a$', '$m_\\mathrm{nu}$'], 'mass_split': 'equal'}
'''

def unpack_m_nu(cosmo_params):
    """
    Unpacks parameters in a dictionary if any of the fiducial entries 
    are lists or arrays (e.g., for parameters like 'm_nu':[0.01, 0.02, 0.03] with individual components).
    If just one shift provided for something like m_nu = [m1, m2, m3], assumes same shift for each unpacked parameter.

    ...
    
    """
    if 'm_nu' in cosmo_params['name']:
        idx = cosmo_params['name'].index('m_nu')
        fid_val = cosmo_params['fiducial'][idx]
        shift_val = cosmo_params['shift'][idx]
        latex_val = cosmo_params['latex'][idx]

        if not isinstance(fid_val, (list, tuple)):
            return cosmo_params

        # Remove the original m_nu entry
        for key in ['name', 'fiducial', 'shift', 'latex']:
            cosmo_params[key].pop(idx)

        # Insert m_nu_1, m_nu_2, m_nu_3 to sublist in dictionary, starting from the same index as m_nu
        for i in len(m_nu):
            cosmo_params['name'].insert(idx + i, f'm_nu_{i+1}')
            cosmo_params['fiducial'].insert(idx + i, fid_val[i])
            cosmo_params['shift'].insert(idx + i, shift_val if isinstance(shift_val, (float, int)) else shift_val[i])
            cosmo_params['latex'].insert(idx + i, f'{latex_val}_{{{i+1}}}')

    return cosmo_params
    

def pack_m_nu(param_dict):
    """
    Packs m_nu_1, m_nu_2, m_nu_3 from a dictionary into m_nu = [m1, m2, m3].
    Assumes input keys are 'm_nu_1', 'm_nu_2', 'm_nu_3'.

    ...

    
    """
    mnu_keys = ['m_nu_1', 'm_nu_2', 'm_nu_3']

    # Cheking if the keys are in the input dict
    if all(k in param_dict for k in mnu_keys): # generates boolean either true or false, all() returns True only if every item is True
        mnu_vals = [param_dict[k] for k in mnu_keys]
        for k in mnu_keys:
            param_dict.pop(k)
        param_dict['m_nu'] = mnu_vals

    return param_dict
    

# ----------------------------------------------------------------------------------------------------------------------------------------------
# Sergi
def compute_Omega_nu(m_nu, h: float, mass_split: str, C_nu: float = 93.14) -> float:
    """
    Computes Omega_nu from neutrino mass and cosmological parameters.

    Args:
        m_nu (float or list of float): Total neutrino mass in eV (float) or list of individual masses (if mass_split == 'list').
        h (float): Reduced Hubble constant (H0 / 100), where H0 is in km/s/Mpc.
        mass_split (str): Neutrino mass split type. Must be one of ['equal', 'list', 'sum', 'single', 'normal', 'inverted'].
        C_nu (float): Conversion factor between neutrino mass and Omega_nu. Default is 93.14 (from Planck units).

    Returns:
        float: Computed value of Omega_nu.
    """
    if mass_split == 'list' or isinstance(m_nu, (list, tuple)):
        M_nu = sum(m_nu)
    else:
        M_nu = m_nu

    return M_nu / (C_nu * h**2)

# ----------------------------------------------------------------------------------------------------------------------------------------------

def bias_vector(C_ell_obs: dict, C_ell_lens: dict, cov: np.ndarray, d_C_ells: np.ndarray):    
    # Calculate C_ell systematic as difference between observed and lensing one
    C_ell_sys = {}
    for key in C_ell_obs:
        C_ell_sys[key] = C_ell_obs[key] - C_ell_lens[key]
    
    # Turn C_ell dict into array 
    C_ell_sys_array = np.array(list(C_ell_sys.values())).T # ell x zi-zj
    
    L, P, N = d_C_ells.shape # ell x parameter x zi-zj
    
    B = np.zeros(P) # bias vector will have as many components as parameters varied
    
    for j in range(P): # sum over parameters
        for ell in range(L): # sum over ells
            invcov = np.linalg.inv(cov[ell,:,:])    # inverse of cov for a given ell -->  (zi-zj x zi-zj)
            sys_vec = C_ell_sys_array[ell,:]         # for a given ell row of Cl_sys -->  (1 x zi-zj) 
            deriv  = d_C_ells[ell, j, :]          # for a given parameter and ell of dC_ells --> (zi-zj x 1) 
    
            B_j = np.dot(sys_vec, np.dot(invcov, deriv)) 
            #Alternative
            #B_j = np.dot(deriv, np.linalg.solve(cov[ell], C_ell_sys_array[ell]))  # cov x result = C_ell sys --> more stable than inversting?
            B[j] += B_j 
    
    return B


def bias_parameter(fisher_matrix: np.ndarray, bias_vector: np.ndarray):
    """
    Computes the biases of each parameter varied using the inverse Fisher matrix and bias vector.

    Args:
        fisher_matrix (np.ndarray): Fisher matrix.
        bias_vector (np.ndarray): Vector of biases B_j (same length as number of parameters).

    Returns:
        b (np.ndarray): Vector with estimated bias on parameters p_i, computed as (F^-1)_ij * B_j.
    """
    
    invfisher = np.linalg.inv(fisher_matrix)
    b = np.dot(invfisher, bias_vector) 

    #alternative:  
    #solution = np.linalg.solve(fisher_matrix, bias_vector)
    #return solution

    return b

def standard_dev(fisher_matrix: np.ndarray):
    """
    Computes the standard deviation (1sigma uncertainties) for each parameter from the Fisher matrix.

    Args:
        fisher_matrix (np.ndarray): Fisher matrix.

    Returns:
        standard_deviation (np.ndarray): Vector of standard deviations for each parameter, computed as the 
                              square roots of the diagonal elements of the inverse Fisher matrix.
    """
    return np.sqrt(np.diag(np.linalg.inv(fisher_matrix)))

  
   
        

    

    