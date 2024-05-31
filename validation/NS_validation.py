import os
# First thing, restrict to 1 CPU per thread for using MPI properly.
os.environ["OMP_NUM_THREADS"] = "1"

from nautilus import Prior, Sampler
from mpi4py.futures import MPIPoolExecutor
import mpi4py.MPI as MPI
from scipy.stats import norm #, multivariate_normal
import pyccl as ccl
from scipy.interpolate import interp1d
from itertools import pairwise
import argparse
import sys
sys.path.append('../')
from src import load_config, fisher_matrix, redshift_distributions, IO, fisher
import numpy as np


def parse_args():
    # Parse the arguments and return the Namespace.
    parser = argparse.ArgumentParser(
        description='',
        epilog='')
    parser.add_argument('-c', '--config', dest='config',
                        help='Location of the configuration yaml file.',
                        default='../configs/config_main.yaml')
    parser.add_argument('-o', '--output', dest='output_filename',
                        help='Location of the output hdf5 file.',
                        default='NS_validation.hdf5')
    parser.add_argument('-r', '--resume', action='store_true',
                        help='If set, the sampling will continue from the last iteration.',
                        default=False)
    return parser.parse_args()


# Manage the MPI processes so that they receive the configuration.
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
if rank == 0:
    # If main process, read argument parser and send it to all other processes.
    # Note: broadcasting seems to mess up the printing and maybe more.
    args = parse_args()
    for i in range(1, size):
        comm.send(args, dest=i, tag=99)
#    comm.bcast(args, root=0, )
else:
    # Receive argument parser from main process.
    args = comm.recv(source=0, tag=99)

config = load_config(args.config)
print(f'Running fisher analysis for {config.name}.')
# Load the cosmology
cosmo = ccl.Cosmology(Omega_c=config.cosmology.Omega_m-config.cosmology.Omega_b,
                      Omega_b=config.cosmology.Omega_b,
                      h=config.cosmology.h,
                      sigma8=config.cosmology.sigma8,
                      n_s=config.cosmology.n_s,
                      w0=config.cosmology.w0,
                      wa=config.cosmology.wa,
                      matter_power_spectrum='camb',
                      extra_parameters={"camb": config.baryons_dict})

# ell binning setup
ell_bins = np.geomspace(config.ell_binning.cosmic_shear.bin_start,
                        config.ell_binning.cosmic_shear.bin_end,
                        config.ell_binning.cosmic_shear.N_bins+1)
ell_bins = ell_bins[
    (ell_bins >= config.ell_binning.cosmic_shear.ell_min) &
    (ell_bins <= config.ell_binning.cosmic_shear.ell_max)
    ].astype(int)
ell_arr = np.array([(a + b) / 2 for a, b in pairwise(ell_bins)]).astype(int)
Delta_ell = np.array([b - a for a, b in pairwise(ell_bins)])

# Define the forecasting setup
sigma_e = config.forecast.e_rms
N_z_bins = config.redshift_distributions.sources.N_z_bins
z_arr = np.linspace(config.forecast.z_min, config.forecast.z_max,
                    config.forecast.N_z_values)
nz_arr, zbin_edges = (
    redshift_distributions.get_redshift_distributions(
        z_arr, N_z_bins,
        config.redshift_distributions.sources.z0,
        config.redshift_distributions.sources.a,
        config.redshift_distributions.sources.sigma_z))

# Define the fisher matrix setup
cosmo_params = {'name': list(config.cosmology.keys()),
                'fiducial': list(config.cosmology.values()),
                'shift': [config.derivatives.step_size[x] for x in list(config.cosmology.keys())],
                'latex': ['$\Omega_\mathrm{m}$', '$\Omega_\mathrm{b}$', '$h$',
                          '$\sigma_8$', '$n_\mathrm{s}$', r'$w_0$', r'$w_a$']}
astro_params = {'name': list(config.IA.keys())+list(config.baryons.keys()),
                 'fiducial': list(config.IA.values())+list(config.baryons.values()),
                 'shift': [config.derivatives.step_size[x] for x in list(config.IA.keys())+list(config.baryons.keys())],
                 'latex': ['$A_\mathrm{IA}$', '$\log T_\mathrm{AGN}$']}
redshift_params = {'name': [f'dz{i+1}' for i in range(N_z_bins)],
                   'fiducial': [0.] * N_z_bins,
                   'shift': [config.derivatives.step_size.delta_z] * N_z_bins,
                   'latex': ['$\Delta z_{%i}$' % (i + 1) for i in range(N_z_bins)]}

# Compute the fisher matrix
print(f'Computing fisher matrix.')
fm = fisher_matrix(cosmo=cosmo, z=z_arr, dndz=nz_arr, ell=ell_arr,
                   sigma_e=sigma_e,
                   n_bar=config.redshift_distributions.sources.nbar,
                   fsky=config.forecast.fsky, Delta_ell=Delta_ell,
                   n_points=config.derivatives.stencil_points, cosmo_params=cosmo_params,
                   astro_params=astro_params, redshift_params=redshift_params)

# Add Gaussian photo-z priors
mean_z = np.trapz(nz_arr * z_arr, z_arr)
scale_z = config.redshift_distributions.sources.sigma_delta_z*(1+mean_z)
fm.add_prior(redshift_params['name'], scale_z)

print(f'Writing output.')
IO.save_fisher_matrix(config, fm, 'fm_sampling_validation')

# Compute the theory data and covariance
C_ell_vec = fm.C_ell
covariance_matrix = fm.data_covariance
data = np.array(list(C_ell_vec.values())).flatten()
N_ell = len(ell_arr)
N_tracer_combinations = len(C_ell_vec.keys())
cov = np.zeros((N_ell*N_tracer_combinations, N_ell*N_tracer_combinations))
for i in range(N_tracer_combinations):
    for j in range(N_tracer_combinations):
        cov[i*N_ell:(i+1)*N_ell, j*N_ell:(j+1)*N_ell] = np.diag(covariance_matrix[:, i, j])
invcov = np.linalg.inv(cov)

# Priors
prior = Prior()
for ip, p in enumerate(config.sampling_validation.keys()):
    if config.sampling_validation[p] is None:
        continue
    elif len(config.sampling_validation[p]) == 2:
        prior.add_parameter(p, dist=tuple(config.sampling_validation[p]))
    elif p == 'delta_z':
        for i in range(N_z_bins):
            prior.add_parameter(f'dz_{i + 1}', dist=norm(loc=0.0,
                scale=config.redshift_distributions.sources.sigma_delta_z*(1 + mean_z[i])))
    else:
        raise ValueError(f'Input parameter in config file had problem: {p}')


# Likelihood
def likelihood(param_dict):
    cosmo_in_dict = {}
    bayrons_dict_in = {}

    for ip, p in enumerate(config.sampling_validation.keys()):
        if config.sampling_validation[p] is None:
            if p in config.cosmology.keys():
                cosmo_in_dict[p] = config.cosmology[p]
            elif p == 'A_IA':
                A_IA_in = None
            elif p == 'delta_z':
                dndz_in = nz_arr
            continue
        else:
            if p in config.cosmology.keys():
                cosmo_in_dict[p] = 1.*param_dict[p]
            elif p == 'logT_AGN':
                bayrons_dict_in = {'kmax': 20, "halofit_version": "mead2020_feedback",
                                   "HMCode_logT_AGN": 1.*param_dict['logT_AGN']}
            elif p == 'A_IA':
                A_IA_in = 1.*param_dict['A_IA']
            elif p == 'delta_z':
                dndz_in = np.zeros(nz_arr.shape)
                for i in range(N_z_bins):
                    interp_dndz = interp1d(z_arr, nz_arr[i], bounds_error=False, fill_value=0)
                    dndz_in[i] = interp_dndz(z_arr + 1.*param_dict[f'dz_{i+1}'])

    if cosmo_in_dict['w0']+cosmo_in_dict['wa'] > 0:
        return -np.inf
    if 'Omega_m' in cosmo_in_dict.keys():
        cosmo_in_dict['Omega_c'] = cosmo_in_dict['Omega_m'] - cosmo_in_dict['Omega_b']
        cosmo_in_dict.pop('Omega_m')
    try:
        bayrons_dict_in.update({"dark_energy_model": "ppf"})
        cosmo_in = ccl.Cosmology(**cosmo_in_dict, extra_parameters={"camb": bayrons_dict_in})
        model = fisher.get_Cell_data_vector(cosmo_in, z_arr, dndz_in, A_IA_in, ell_arr)
        model = np.array(list(model.values())).flatten()
        #return multivariate_normal.logpdf(model, mean=data, cov=cov)
        return -0.5 * np.dot(np.dot(data-model, invcov), data-model)
    except:
        print(f'Problem with parameters: {param_dict.values()}')
        return -np.inf


if __name__ == '__main__':
    filename = args.output_filename
    resume = args.resume

    n_live_points = 3000
    f_live = 0.01

    try: pool_type = MPIPoolExecutor()
    except: pool_type = None

    sampler = Sampler(prior, likelihood,
                      filepath=filename, resume=resume,
                      n_live=n_live_points,
                      pool=pool_type)
    sampler.run(verbose=True, discard_exploration=True, f_live=f_live)
