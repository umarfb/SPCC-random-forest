import pandas as pd
import random
import os

# Method to parse through an SPCC .DAT file and extract the following:
#   SNID: (Supernova ID number)
#   FILTERS: (Filters used)
#   RA: (Right ascension)
#   DECL: (Declination)
#   MAGREF: (Magnitude system used)
#   MWEBV: (Milky Way E(B - V))
#   HOST_GALAXY_PHOTO-Z: (Photometric redshift of host galaxy)
#   SIM_COMMENT: (Comments on SN type and model used)
#
# Return a list containing the above information for all
# supernovae
def extract_info(filename):

    # Open .DAT file, get the main text, close .DAT file
    wkfile = open(filename, mode = 'r')
    main_txt = wkfile.readlines()
    wkfile.close()

    # list to store SN information, in the order
    # SNID, FILTERS, RA, DECL, MWEBV, HOST_GALAXY_PHOTO-Z,
    # HOST_GALAXY_PHOTO-Z_ERROR, SN_TYPE, SPEC_Z, SPEC_Z_ERR
    sn_info = []

    # Parse through .DAT file
    # Go through each line and extract relevant information
    # Extract SN information to dictionary sn_dict
    for line in main_txt:
        line = line.strip('\n').split(':')

        if line[0] == 'SNID':
            snid = line[1].strip()

            # Convert to a six-digit long ID number
            id_len = 6
            n_zeros = id_len - len(snid)
            form_snid = 'SN' + n_zeros*'0' + snid

            sn_info.append(form_snid)
        
        if line[0] == 'SNTYPE':
            sn_type = line[1]
            sn_info.append(sn_type.strip())
    
        if line[0] == 'FILTERS':
            filters = line[1]
            sn_info.append(filters.strip())
        
        # RA and DEC given in degrees
        if line[0] == 'RA':
            ra = line[1]
            sn_info.append(ra.strip().strip('deg').strip())
        
        if line[0] == 'DECL':
            dec = line[1]
            sn_info.append(dec.strip().strip('deg').strip())
        
        '''
        # MAGREF not needed for original SPCC challenge set
        if line[0] == 'MAGREF':
            magref = line[1]
            sn_info.append(magref.strip())
        '''
        
        if line[0] == 'MWEBV':
            mw_ebv = line[1]
            sn_info.append(mw_ebv.split()[0].strip())
        
        if line[0] == 'REDSHIFT_SPEC':
            spec_z = line[1].split('+-')
            spec_z_val = spec_z[0]
            spec_z_err = spec_z[1]
            sn_info.append(spec_z_val.strip())
            sn_info.append(spec_z_err.strip())
        
        if line[0] == 'HOST_GALAXY_PHOTO-Z':
            host_z = line[1].split('+-')
            host_z_val = host_z[0]
            host_z_err = host_z[1]
            sn_info.append(host_z_val.strip())
            sn_info.append(host_z_err.strip())

        # Used to get SN type answer keys (known types for all)
        if line[0] == 'SIM_COMMENT':
            sn_info.append(line[1].split(',')[0].strip('SN Type = ').strip())

    return sn_info

# Method to parse through SPCC .DAT file to extract photometry data:
#   MJD: Time of observation
#   FLT: Filter used (griz)
#   FIELD
#   FLUXCAL: Calibrated flux
#   FLUXCALERR: Error in calibrated flux
#   SNR: Signal-to-noise ratio
#   MAG: Apparent magnitude
#   MAGERR: Error in apparent magnitude
#   SIM_MAG: Simulated magnitude value
#
# Return a list containing the above values, where one item in the list
# corresponds to a row in the table
def extract_phot(filename):

    # Open .DAT file, get the main text, close .DAT file
    wkfile = open(filename, mode = 'r')
    main_txt = wkfile.readlines()
    wkfile.close()

    # list to store photometric data, in the order
    # MJD, FLT, FIELD, FLUXCAL, FLUXCALERR, SNR, MAG, MAGERR, SIM_MAG
    phot_list = []

    # Parse through .DAT file
    # Go through each line and extract relevant information
    # Extract photometry to phot_list
    for line in main_txt:
        line = line.strip('\n').split(':')

        if line[0] == 'VARLIST':
            phot_header = line[1].strip()
            phot_list.append(phot_header)
        
        if line[0] == 'OBS':
            phot_row = line[1].strip()
            phot_list.append(phot_row)
    
    return phot_list

# Method to convert list of photometry data into a pandas dataframe
# Returns a pandas dataframe
def phot_to_dataframe(data_list):

    # Split each row in data_list, so that each row is a list of observables
    data_list = list(row.split() for row in data_list)

    # Separate out column headers and data rows
    df_header = data_list[0]
    df_main = data_list[1:]

    #print(df_header)
    #print(df_main)

    # Create pandas dataframe for lightcurve data
    lc_df = pd.DataFrame(df_main, columns=df_header)
    # Convert all numerical data to numerical format
    lc_df_num = lc_df[['MJD', 'FLUXCAL', 'FLUXCALERR', 'SNR', 'MAG', 'MAGERR', 'SIM_MAG']]
    lc_df[['MJD', 'FLUXCAL', 'FLUXCALERR', 'SNR', 'MAG', 'MAGERR', 'SIM_MAG']] = lc_df_num.apply(pd.to_numeric)

    # Get row containing value of peak r-band flux
    rband_rows = lc_df.loc[lc_df['FLT'] == 'r']
    peak_row = rband_rows.loc[rband_rows['FLUXCAL'].idxmax()]
    # Get time of peak r-band flux
    t_peak = peak_row['MJD']

    # Set t=0 at peak r-band flux (shift by epoch at peak r-band flux)
    # Create new column of normalised epochs in lightcurve dataframe
    lc_df['NORM_T'] = lc_df['MJD'] - t_peak

    return lc_df

# Method to convert a 2-D list of SN information into a pandas dataframe
# Return a pandas dataframe
def info_to_dataframe(data_list):

    # Define column headers
    df_header = ['SNID', 'TYPE', 'FILTERS', 'RA', 'DEC', 'MW_EBV', 'SPEC_Z', 'SPEC_Z_ERR', 'HOST_Z', 'HOST_ZERR', 'SIM_TYPE']

    #print(df_header)
    #print(data_list)

    # Create pandas dataframe for lightcurve data
    sn_df = pd.DataFrame(data_list, columns=df_header)

    # Convert all numerical data to numerical format
    sn_df_num = sn_df[['RA', 'DEC', 'MW_EBV', 'HOST_Z', 'HOST_ZERR', 'SPEC_Z', 'SPEC_Z_ERR']]
    sn_df[['RA', 'DEC', 'MW_EBV', 'HOST_Z', 'HOST_ZERR', 'SPEC_Z', 'SPEC_Z_ERR']] = sn_df_num.apply(pd.to_numeric)

    return sn_df

# ---------------------------------------------------------------------------------------------------------------
# TESTING WITH 32 SUPERNOVAE LIGHT-CURVES

'''
# Test data directory (32)
testdata_path = '/local/php18ufb/backed_up_on_astro3/SPCC/test_data_32/'

# Test lightcurves directory (32)
testlc_path = '/local/php18ufb/backed_up_on_astro3/SPCC/test_lightcurves_32/'

# Get list of .DAT files
files_list = os.listdir(testdata_path)

sn_table_rows = []

for sn_file in files_list:
    sn_data = extract_info(testdata_path + sn_file)
    sn_phot = extract_phot(testdata_path + sn_file)
    sn_phot_df = phot_to_dataframe(sn_phot)

    # Get unique SN id number
    snid = sn_data[0]

    # Add table row to list
    sn_table_rows.append(sn_data)

    # Save photometry to .csv file
    sn_phot_df.to_csv(testlc_path + snid + '.csv')

sn_table_df = info_to_dataframe(sn_table_rows)
# Save table of SN information to .csv file
sn_table_df.to_csv('sn_cat32.csv')
'''

# ---------------------------------------------------------------------------------------------------------------
# TESTING WITH 1000 SUPERNOVAE LIGHT-CURVES

'''
# Path to all SPCC data files
spcc_path = '/local/php18ufb/backed_up_on_astro3/SPCC/data/SIMGEN_PUBLIC_DES/'

# Test lightcurves directory (2500)
testlc_path = '/local/php18ufb/backed_up_on_astro3/SPCC/test_lightcurves_2500/'

# get list all .DAT files in the data files directory
spcc_files = [filename for filename in os.listdir(spcc_path) if filename.endswith('.DAT')]

# Select 2500 random supernovae from full list
sn_sample = random.sample(spcc_files, 2500)

sn_table_rows = []

for i, sn_file in enumerate(sn_sample):
    print('Extracting data ... {0}%'.format(round((i+1)*100/len(sn_sample), 1)), end='\r') # print progress in terminal

    sn_data = extract_info(spcc_path + sn_file)
    sn_phot = extract_phot(spcc_path + sn_file)
    sn_phot_df = phot_to_dataframe(sn_phot)

    # Get unique SN id number
    snid = sn_data[0]

    # Add table row to list
    sn_table_rows.append(sn_data)

    # Save photometry to .csv file
    sn_phot_df.to_csv(testlc_path + snid + '.csv')
print()

sn_table_df = info_to_dataframe(sn_table_rows)

# Save table of SN information to .csv file
sn_table_df.to_csv('sn_cat2500.csv')
'''

# ---------------------------------------------------------------------------------------------------------------
# FULL RUN WITH ALL SUPERNOVAE

# Path to all SPCC data files
spcc_path = '/local/php18ufb/backed_up_on_astro3/SPCC/data/SIMGEN_PUBLIC_DES/'
#spcc_path = '/local/php18ufb/backed_up_on_astro3/SPCC/data/DES_BLIND+HOSTZ/'

# Full lightcurves directory for saving
lc_path = '/local/php18ufb/backed_up_on_astro3/SPCC/lightcurves/'

# get list all .DAT files in the data files directory
spcc_files = [filename for filename in os.listdir(spcc_path) if filename.endswith('.DAT')]

sn_table_rows = []

for i, sn_file in enumerate(spcc_files):
    print('Extracting data ... {0}%'.format(round((i+1)*100/len(spcc_files), 1)), end='\r') # print progress in terminal

    sn_data = extract_info(spcc_path + sn_file)
    #sn_phot = extract_phot(spcc_path + sn_file)
    #sn_phot_df = phot_to_dataframe(sn_phot)

    # Get unique SN id number
    snid = sn_data[0]

    # Add table row to list
    sn_table_rows.append(sn_data)

    # Save photometry to .csv file
    #sn_phot_df.to_csv(lc_path + snid + '.csv')
print()
#print(sn_table_rows[-1])
sn_table_df = info_to_dataframe(sn_table_rows)

# Split into spectroscopically confirmed and non-spec. confirmed dataframes
sn_spec_types = sn_table_df.loc[sn_table_df['TYPE']!= '-9']
sn_no_spec_types = sn_table_df.loc[sn_table_df['TYPE'] == '-9']

# Save table of SN information to .csv file
#sn_table_df.to_csv('spcc_sn_cat.csv')

# Print size of each sample
print('Spectroscopically confirmed: {0}'.format(len(sn_spec_types)))
print('Not spectroscopically confirmed: {0}'.format(len(sn_no_spec_types)))
print('Total no. of SPCC supernovae: {0}'.format(len(sn_table_df)))

sn_spec_types.to_csv('SPCC_spectypes.csv')
sn_no_spec_types.to_csv('SPCC_no_spectypes.csv')