import numpy as np
import pandas as pd

from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Path to data
#dat_path = '/local/php18ufb/backed_up_on_astro3/PTF_classification/OSNC_cat/2019_02_20_r005/'

# Load dataframe containing SN features and target classes
sn_data = pd.read_csv('reduced_SN_features_trainset.csv')
#print(sn_data)

# Remove index column
sn_datahdr = list(sn_data)
#print(sn_datahdr)

# Relabel type II subtypes as 'II' to group all type II SNe into one class, 
# and all Ibc in to a 1bc group
sn_type_arr = sn_data['sn_type'].values

for i, sntype_id in enumerate(sn_data['sn_type']):
    if sntype_id =='II':
        pass
    if sntype_id == 'Ibc':
        pass
    if 'II' in sntype_id:
        #print('Changed type {0} to type {1}'.format(sntype_id, 'II'))
        sn_type_arr[i] = 'II'
    if 'Ib' in sntype_id:
        sn_type_arr[i] = 'Ibc'
    if 'Ic' in sntype_id:
        sn_type_arr[i] = 'Ibc'

sn_data['sn_type'] = sn_type_arr

# Separate data into features and targets (class label)
# Types we want to predict
targets = np.array(sn_data['sn_type'])

# Remove target and name from features
sn_data = sn_data.drop('sn_type', axis=1)
sn_data = sn_data.drop('sn_id', axis=1)
sn_data = sn_data.drop('Unnamed: 0', axis=1)

# Convert features to numpy array
sn_features = np.array(sn_data)

# Split data into training and testing sets
train_feats, test_feats, train_labels, test_labels = train_test_split(sn_data, targets, test_size=0.10)

print('Training Features Shape:', train_feats.shape)
print('Training Labels Shape:', train_labels.shape)
print('Testing Features Shape:', test_feats.shape)
print('Testing Labels Shape:', test_labels.shape)

# Instantiate RF moel with N decision trees
N = int(input('Number of decision trees: '))
rf = RandomForestClassifier(n_estimators=N)

# Train model on training set
rf.fit(train_feats, train_labels)

# Use the forest's predict method on the test data
predictions = rf.predict(test_feats)

i=0
# Classification accuracy
accuracy = metrics.accuracy_score(test_labels, predictions)

#for i in range(len(predictions)):
    #print('Predicted label: {0}, True label: {1}'.format(predictions[i], test_labels[i]))
print('Accuracy: {0}%'.format(round(accuracy*100,2)))
