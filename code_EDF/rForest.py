from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np


# EDF data
tr_input = pd.read_csv("../data_EDF/training_inputs.csv", sep = ";")
output = pd.read_csv("../data_EDF/challenge_output_data_training_file_predict"
                     "_which_clients_reduced_their_consumption.csv", sep = ";")
data_edf = pd.merge(left=tr_input, right=output, how='left', on="ID")

# ENEDIS data
data_enedis = pd.read_csv("../data_EDF/consommation-electrique-par-secteurs-dactivite.csv", sep = ";")


data_edf['is_train'] = np.random.uniform(0, 1, len(data_edf)) <= .75
data_edf['Target'] = data_edf['Target'].astype('Category')

train, test = data_edf[data_edf['is_train']==True], data_edf[data_edf['is_train']==False]

features = data_edf.columns[:1]
clf = RandomForestClassifier(n_jobs=2)
y, _ = pd.factorize(train['Target'])
clf.fit(train[features], y)

#preds = [clf.predict(test[features])]
#pd.crosstab(test['species'], preds, rownames=['actual'], colnames=['preds'])