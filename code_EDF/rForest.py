from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np


# EDF data
tr_input = pd.read_csv("../data_EDF/training_inputs.csv", sep = ";")
output = pd.read_csv("../data_EDF/challenge_output_data_training_file_predict"
                     "_which_clients_reduced_their_consumption.csv", sep = ";")
data_edf = pd.merge(left=tr_input, right=output, how='left', on="ID")

to_drop = ['S1', 'S6', 'S7', 'Q6', 'Q7', 'Q15', 'Q17', 'Q18', 'Q37', 'Q38', 'Q39', 'Q40', 'Q52', 'Q73', 'Q74', 'Q75']
data_edf.drop(to_drop,inplace=True,axis=1).dropna()

# ENEDIS data
#data_enedis = pd.read_csv("../data_EDF/consommation-electrique-par-secteurs-dactivite.csv", sep = ";")


data_edf['is_train'] = np.random.uniform(0, 1, len(data_edf)) <= .75
data_edf['TARGET'] = data_edf['TARGET'].astype('category')

train, test = data_edf[data_edf['is_train']==True], data_edf[data_edf['is_train']==False]

features 
clf = RandomForestClassifier(n_jobs=2)
y, _ = pd.factorize(train['TARGET'])
clf.fit(train[features], y)

preds = [clf.predict(test[features])]
print(pd.crosstab(test['TARGET'], preds, rownames=['actual'], colnames=['preds']))