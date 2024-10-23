import os
import numpy as np
from src import fisher_matrix
from src.configs import DictAsMember


# TODO: I should change this all...
def save_fisher_matrix(config: DictAsMember, fisher: fisher_matrix, name: 'str'):
    output_dir = config.paths.output.fisher_matrix
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    np.save(os.path.join(output_dir, f'{config.name}_{name}.npy'),
            fisher.fisher_matrix)


def save_fisher_matrix_validation(config: DictAsMember,
                                  shifts_array: np.ndarray,
                                  validation_array: np.ndarray,
                                  name: str):
    output_dir = config.paths.output.fisher_matrix
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    np.savez(os.path.join(output_dir, f'{config.name}_{name}.npz'),
             shifts=shifts_array, validation=validation_array)


def load_fisher_matrix(config: DictAsMember,
                       name: str) -> fisher_matrix:
    fisher_matrix_array = np.load(os.path.join(config.paths.output.fisher_matrix,
                                               f'{config.name}_{name}.npy'))

    N_z_bins = config.redshift_distributions.sources.N_z_bins

    cosmo_params = {'name': list(config.cosmology.keys()),
                    'fiducial': list(config.cosmology.values()),
                    'shift': [config.derivatives.step_size[x] for x in list(config.cosmology.keys())],
                    'latex': ['$\Omega_\mathrm{m}$', '$\Omega_\mathrm{b}$', '$h$',
                              '$\sigma_8$', '$n_\mathrm{s}$', r'$w_0$', r'$w_a$', '']}
    astro_params = {'name': list(config.IA.keys()) + list(config.baryons.keys()),
                    'fiducial': list(config.IA.values()) + list(config.baryons.values()),
                    'shift': [config.derivatives.step_size[x] for x in list(config.IA.keys()) + list(config.baryons.keys())],
                    'latex': ['$A_\mathrm{IA}$', '$\eta$', '$\log T_\mathrm{AGN}$']}
    redshift_params = {'name': [f'dz{i+1}' for i in range(N_z_bins)],
                       'fiducial': [0.] * N_z_bins,
                       'shift': [config.derivatives.step_size.delta_z] * N_z_bins,
                       'latex': ['$\Delta z_{%i}$' % (i + 1) for i in range(N_z_bins)]}

    fisher_matrix_object = fisher_matrix(fisher_from_input=fisher_matrix_array,
                                         cosmo_params=cosmo_params,
                                         astro_params=astro_params,
                                         redshift_params=redshift_params)
    return fisher_matrix_object


def load_fisher_matrix_validation(config: DictAsMember,
                                  name: str) -> (np.ndarray, np.ndarray):
    data = np.load(os.path.join(config.paths.output.fisher_matrix,
                                f'{config.name}_{name}.npz'))
    return data['shifts'], data['validation']
