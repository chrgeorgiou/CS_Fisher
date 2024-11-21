# Cosmic Shear Fisher
This code computes the Fisher matrix for a cosmic shear survey. It assumes equally-populated redshift bins, a constant 
intrinsic alignment amplitude through the survey and logarithmically spaced 
$`\ell`$
bins.

The user interface happens through a configuration file, located in ```configs```, which is then used in the 
```main.py``` script. The code dumps its output in a directory specified in the configuration file. This output 
can be plotted using the ```notebooks``` directory. 

## Configuration file
```yaml
name: This will be added in front of the output files.

paths:
    project: This should point to the project directory (where the src code is) 
    output:
        fisher_matrix: Output of calculated arrays (fisher matrix, validation).
        figures: Output of figures (not used now).

validation: Which parameters to use to calculate figure-of-merit for validation.

cosmology: The values of corresponding cosmological parameters.

IA:
    A_IA: The value of the NLA amplitude.
    eta: The value of the redshift dependence ((1+z)/(1+z0))**eta with z0=0.62.

derivatives:
    validation: Parameters to use to compute the figure of merit when validating derivatives.
    stencil_points: Number of points for the stencil method to compute derivatives.
    step_size:
        Omega_m: 0.002
        Omega_b: 0.002
        A_s: 0.002
            Step size for calculating derivatives of each parameter. 
            if left empty, that parameter will not be varied in the fisher matrix.
            Note that, if used, the step size of A_s should be in units of 1e+9.

ell_binning:
    cosmic_shear:
        bin_start: Start of the first ell bin.
        bin_end: End of the first ell bin.
        N_bins: Number of ell bins to consider.
        ell_min: Minimum ell, will discard bins that start lower than this value.
        ell_max: Minimum ell, will discard bins that end higher than this value.

forecast:
    fsky: Fraction of sky covered by the survey.
    e_rms: Ellipticity RMS, used for shape-noise calculation.
    year: LSST survey release (can be 1 or 10).
    z_min: Minimum redshift for the distribution.
    z_max: Maximum redshift for the distribution.
    N_z_values: Number of redshift values.

sampling_validation:
    Omega_m: [0.2, 0.45]
    Omega_b: [0.005, 0.12]
    A_s: [1.0, 3.0]
    logA_s: [-8, -6] # Don't use both! Using base-10 logarithm.
        Ranges for the uniform priors to be used during nested sampling.
        Note that, if used, the ranges of A_s should be in units of 1e+9
```

# Fisher Matrix
The fisher matrix object created will have several useful properties. It is built using
the ```fisher_matrix``` function in ```src/fisher.py```. 
It contains a list of the parameters, their fiducial values and latex representation, that can be accessed via
```fisher_matrix.parameters```, ```fisher_matrix.fiducial_parameters``` and ```fisher_matrix.latex_parameters```,
respectively. Integrated is also a function to plot the fisher ellipse, accessed with
```fisher_matrix.draw_covariance_ellipse```. If not read from input, also contains a dictionary with the theoretical
data vector that can be accessed with 
```fisher_matrix.C_ell```. Each key of the dictionary contains the theory prediction for the corresponding redshift 
bin combination. 