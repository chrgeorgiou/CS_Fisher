import pyccl as ccl
import numpy as np
from itertools import pairwise
from src import load_config, fisher_matrix, redshift_distributions, IO


def main(config):
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
                          extra_parameters = {"camb": config.baryons_dict})

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
                    'shift': [config.step_size[x] for x in list(config.cosmology.keys())],
                    'latex': ['$\Omega_\mathrm{m}$', '$\Omega_\mathrm{b}$', '$h$',
                              '$\sigma_8$', '$n_\mathrm{s}$', r'$w_0$', r'$w_a$']}
    astro_params = {'name': list(config.IA.keys())+list(config.baryons.keys()),
                 'fiducial': list(config.IA.values())+list(config.baryons.values()),
                 'shift': [config.step_size[x] for x in list(config.IA.keys())+list(config.baryons.keys())],
                 'latex': ['$A_\mathrm{IA}$', '$\log T_\mathrm{AGN}$']}
    redshift_params = {'name': [f'dz{i+1}' for i in range(N_z_bins)],
                       'fiducial': [0.] * N_z_bins,
                       'shift': [config.step_size.delta_z] * N_z_bins,
                       'latex': ['$\Delta z_{%i}$' % (i + 1) for i in range(N_z_bins)]}

    # Compute the fisher matrix
    print(f'Computing fisher matrix.')
    fm = fisher_matrix(cosmo=cosmo, z=z_arr, dndz=nz_arr, ell=ell_arr,
                       sigma_e=sigma_e,
                       n_bar=config.redshift_distributions.sources.nbar,
                       fsky=config.forecast.fsky, Delta_ell=Delta_ell,
                       n_points=3, cosmo_params=cosmo_params,
                       astro_params=astro_params, redshift_params=redshift_params)
    # Add Gaussian photo-z priors
    mean_z = np.trapz(nz_arr * z_arr, z_arr)
    scale_z = config.redshift_distributions.sources.sigma_delta_z*(1+mean_z)
    fm.add_prior(redshift_params['name'], scale_z)

    # Fisher matrix validation
    print(f'Producing fisher matrix validation plot.')
    shifts = np.geomspace(5e-3, 3e-1, 16)
    FoM_parameters = config.validation
    FoM_validation = fm.validate_fisher_matrix(shifts, FoM_parameters)

    print(f'Writing output.')
    IO.save_fisher_matrix(config, fm, 'fisher_matrix')
    IO.save_fisher_matrix_validation(config, shifts, FoM_validation, 'validation_fisher_matrix')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(
        description='',
        epilog='')
    parser.add_argument('-c', '--config', dest='config',
                        help='Location of the configuration yaml file.',
                        default='configs/config_main.yaml')
    args = parser.parse_args()

    config = load_config(args.config)

    main(config)
