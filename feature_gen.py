# Script to do generate features from SN light curve data
import os
import pywt
import random
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ExpSineSquared
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.decomposition import PCA

# ignore UserWarning messages during wavelet decomposition
warnings.filterwarnings("ignore", category=UserWarning)

# Method to do GPR fitting of lightcurves, in one band
# Take in photometric time-series data:
#  - A list of observations in one filter
#
# Return:
#  - [x,y,yerr], where x is the phase
#    and y is the GPR fit of the light curve, and
#    yerr is the error in y
def gaussian_regfit(lc_tab, flt):

    # list to hold GPR fit variables
    lcfit = []

    lc = lc_tab
    # Use only data with SNR > 1
    lc = lc.loc[lc['SNR'] > 1]
    
    # Load in values from observation
    t = lc['NORM_T']
    flux = lc['FLUXCAL']
    flux_err = lc['FLUXCALERR']
    #flt = lc['FLT'].unique()[0]
    
    x_in = t
    y_in = flux
    y_err = flux_err

    x_min = min(t) - 10
    x_max = max(t) + 10

    # Mesh the input space for evaluations of the real function, the prediction and
    # its MSE
    x_space = np.atleast_2d(np.linspace(x_min, x_max, 100)).T
    x_fit = np.atleast_2d(x_in).T

    # Define RBF kernel
    k_rbf = RBF(length_scale=10, length_scale_bounds=(5., 1e2))
    # Define sine kernel
    k_sine = ExpSineSquared(length_scale=1e2, length_scale_bounds=(5., 1e2), periodicity=1e2, periodicity_bounds=(1e2, 1e4))

    # Define white noise kernel
    k_noise = WhiteKernel(noise_level=0.1, noise_level_bounds=(1e-3, 1e3))

    kernel = 1.0*k_rbf + 1.0*(k_rbf*k_sine) + k_noise

    '''
    # Things used for plotting
    kernel_label = 'RBF + RBF*Sine + noise'
    mean_colors = ['#000099', '#b30000', '#006600']
    var_colors = ['#9999ff', '#ff8080', '#66ff66']
    '''

    gpr = GaussianProcessRegressor(kernel=kernel, alpha=y_err, n_restarts_optimizer=10)

    # Fit to data using Maximum Likelihood Estimation of the parameters
    gpr.fit(x_fit, y_in)

    # Make the prediction on the meshed x-axis (ask for MSE as well)
    y_pred, y_pred_sigma = gpr.predict(x_space, return_std=True)
    
    # Store fitted lightcurve into holding list
    lcfit.append([x_space.flatten(), y_pred, y_pred_sigma])

    '''
    # Get log likelihood and hyperparameters
    log_likelihood = np.round(gpr.log_marginal_likelihood(),2)
    hyper_params = gpr.kernel_
    params = hyper_params.get_params()
        '''
    return lcfit

# Method to write GPR fit data to .csv file
def tabulate_gpr(lc_tab, obs_filter, sn_id, sn_type, save_directory):
    #print(lc_tab[0])
    # Extract information
    t = lc_tab[0]
    flux = lc_tab[1]
    flux_err = lc_tab[2]
    # Make column for SN type
    sn_typecol = [sn_type] * len(t)

    # Create pandas dataframe for GPR fit values
    gp_data = {'phase' : t, 'flux_pred' : flux, 'flux_pred_sigma' : flux_err, 'type' : sn_typecol}
    gp_df = pd.DataFrame(data=gp_data)
    filename = sn_id + '_gprfit_{0}.csv'.format(obs_filter)
    gp_df.to_csv(save_directory + filename, header=True, index=False)

# Methods to do wavelet decompostion on lightcurves

# Method to do 2-level wavelet decomposition on 2d data (epoch, mag),
# returns wavelet coefficients, a 2D array of wavelet coefficients
def get_wavedec_2d(x_in, y_in, wavelet):
    coeffs = pywt.wavedec2([x_in, y_in], wavelet, level=2)
    #print('Decomposition level: {0}'.format(len(coeffs)-1))
    cA2 = coeffs[0]
    cH2, cV2, cD2 = coeffs[1][0], coeffs[1][1], coeffs[1][2]
    cH1, cV1, cD1 = coeffs[2][0], coeffs[2][1], coeffs[2][2]
    
    coeffs_vector = [cA2, cH2, cV2, cD2, cH1, cV1, cD1]
    
    return coeffs_vector

# Method to do 2-level stationary wavelet decomposition on 1d data (flux as a function of time),
# returns wavelet coefficients
def get_stwavedec_1d(y_in, wavelet):
    coeffs = pywt.swt(y_in, wavelet, level=2)
    return coeffs

# Method to flatten vector of coefficients
def flatten_coeffs(coeff_vector):
    
    #flattened_arr = [coeff.flatten() for coeff in coeff_vector]
    flattened_arr = []
    
    for coeff in coeff_vector:
        flat_coeffs = coeff.flatten()
        flattened_arr.extend(flat_coeffs)
    
    return flattened_arr

# Method to create a dictionary of coeff labels and values and type
def mk_coeff_dict(coeff_vector, sn_id, sn_type):
    len_cv = len(coeff_vector)
    
    coeff_labels = ['coeff{0}'.format(i) for i in range(len_cv)]
    
    coeff_dict = {}
    
    for j, label in enumerate(coeff_labels):
        coeff_dict[label] = coeff_vector[j]
    
    coeff_dict['type'] = sn_type
    coeff_dict['id'] = sn_id
    
    return coeff_dict

# Path to SPCC lightcurves
lc_path = '/local/php18ufb/backed_up_on_astro3/SPCC/lightcurves_withkeys/'

# Path to Training set lightcurves (not needed?)
trainset_lcpath = '/local/php18ufb/backed_up_on_astro3/SPCC/trainset_lightcurves/'

# Path to training set gpr fits
trainset_gprpath = '/local/php18ufb/backed_up_on_astro3/SPCC/trainset_gprfit/'

# Test save directory for GPR fits
gpr_path = '/local/php18ufb/backed_up_on_astro3/SPCC/test_gprfit_350/'

##############################################################################

# get list of all lightcurves
lc_files = sorted(os.listdir(lc_path))

# Get a random sample of lightcurves for testing
#lc_tsample = random.sample(lc_files, 350)

# Load full SPCC SN catalog as a dataframe
sn_cat = pd.read_csv('spcc_sn_keycat.csv')

# Load SPCC SN catalog for the training set
train_sn_cat = pd.read_csv('SPCC_spectypes.csv')
test_sn_cat = pd.read_csv('SPCC_no_spectypes.csv')

# Get filenames of training set SNe
train_snid = train_sn_cat['SNID'].tolist()
train_lcs = [train_id + '.csv' for train_id in train_snid]

# Get filenames of test set SNe
test_snid = test_sn_cat['SNID'].tolist()
test_lcs = [test_id + '.csv' for test_id in test_snid]

# --------------------
# Testing with 10 training set supernovae

#lc_tsample = random.sample(train_lcs, 10)

# Select wavelet family
wavelet = input('Wavelet family: ')

# List of wavelet coefficients of each SN
sn_features = []
n_rej = 0   # Track no. of rejected lightcurves
for i, sn in enumerate(train_lcs):

    # Display progress in terminal
    print('Getting features from supernova light curves ... {0}/{1}'.format((i+1), len(train_lcs)), end='\r')

    # Load photometric data as pandas dataframe
    lc_df = pd.read_csv(lc_path + sn)
    sn_id = sn.strip('.csv')

    # Get SN type
    sn_data = sn_cat.loc[sn_cat['SNID'] == sn_id]
    sn_type = sn_data.iloc[0]['TYPE']

    # Get list of filters
    flts = lc_df['FLT'].unique()

    # For each filter, do:
    # - Gaussian Regression fit, save fit to a .csv file
    # - Wavelet decomposition, store wavelet coefficients
    #   in an array outside of per filter loop
    wt_coeffs = []

    # If no_obs = True in any filter, then exit the loop
    no_obs = False
    for f in flts:

        # Select all observations in a single filter
        lc = lc_df.loc[lc_df['FLT'] == f]

        # Exit loop if there is no observations (Catch exception)
        try:
            # Do Gaussian Regression fit
            gpr_fit = gaussian_regfit(lc, f)

            # Save GPR fit to .csv file
            tabulate_gpr(gpr_fit[0], f, sn_id, sn_type, trainset_gprpath)

            # Get information needed for wavelet decomposition:
            # - phase
            # - predicted flux
            phase = gpr_fit[0][0]
            flux_pred = gpr_fit[0][1]

            # Do wavelet decomposition
            #wvlets = get_wavedec_2d(phase, flux_pred, wavelet) - OLD
            stationary_coeffs = get_stwavedec_1d(flux_pred, wavelet)

            sn_coeff_vector = np.array(stationary_coeffs).flatten()
            wt_coeffs.append(sn_coeff_vector)

            #wvlets_vec = flatten_coeffs(wvlets) - OLD
            #wt_coeffs.append(wvlets_vec) - OLD

        except ValueError:
            no_obs = True
            n_rej += 1
            break
    
    # Do nothing if any of the filters contain no observations
    if no_obs == True:
        print('\nNo observations in {0} band for {1}'.format(f, sn_id), end = '\n')
    else:
        # Combine wavelet coefficients for different filters into one list
        wt_coeffs = np.array(wt_coeffs).flatten()
        # Convert NaN to zero
        wt_coeffs = np.nan_to_num(wt_coeffs)

        features_dict = mk_coeff_dict(wt_coeffs, sn_id, sn_type)
        sn_features.append(features_dict)
    
print()
print('No. of rejected SNe: {0}'.format(n_rej))

# Create dataframe containing the wavelet coefficients of all SNe
#print(len(sn_wvlets), len(sn_typescol))
print('Shape of wavelet coefficients table: ', np.shape(sn_features))
sn_features_df = pd.DataFrame(sn_features)
#print(sn_features)

# Save all wavelet coefficients to a .csv file
sn_features_df.to_csv('sn_wvlet_coeffs_trainset.csv')