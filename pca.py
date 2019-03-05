from sklearn.decomposition import PCA
import pandas as pd
import numpy as np

# Load wavelet coefficients for dimensionality reduction
sn_features = pd.read_csv('sn_wvlet_coeffs_trainset.csv')

# Attempt principal component analysis
sn_coeffs_hdr = list(sn_features)

# Split into features and label
wv_coeffs = sn_coeffs_hdr[:-2]
sn_type = sn_coeffs_hdr[-1]
sn_id = sn_coeffs_hdr[-2]

wv_coeffs_df = sn_features.loc[:,wv_coeffs]
sn_type_df = sn_features.loc[:,sn_type]
sn_id_df = sn_features.loc[:,sn_id]

'''
# Standardize the data, column by column
for col in wv_coeffs_df:
    col_data = wv_coeffs_df[col]
    
    # Get mean and std. deviation
    col_mean = np.mean(col_data)
    col_stdv = np.std(col_data)
    
    # Standardise array
    wv_coeffs_df[col] = col_data - col_mean
    wv_coeffs_df[col] = col_data / col_stdv
'''

# Specify number of principal components
n_comps = int(input('Number of principal components: '))

pca = PCA(n_components = n_comps)
principal_comps = pca.fit_transform(wv_coeffs_df)
principalDf = pd.DataFrame(data = principal_comps, index=None)
principalDf['sn_id'] = sn_id_df
principalDf['sn_type'] = sn_type_df

print(pca.explained_variance_ratio_[:3])
print('percentage of original information retained in {0} components is {1}'.format(n_comps, sum(pca.explained_variance_ratio_[:])*100))

# Save reduced SN features to .csv file
principalDf.to_csv('reduced_SN_features_trainset.csv')