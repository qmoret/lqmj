from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np

lb = LabelEncoder()

# EDF data
tr_input = pd.read_csv("../data_EDF/training_inputs.csv", sep = ";")
output = pd.read_csv("../data_EDF/challenge_output_data_training_file_predict"
                     "_which_clients_reduced_their_consumption.csv", sep = ";")
data_edf = pd.merge(left=tr_input, right=output, how='left', on="ID")

#S3-S7 : dates
to_drop = ['ID', 'COD_INSEE', 'COD_IRIS', 'S1', 'S3', 'S4', 'S5', 'S6', 'S7', 'Q6', 'Q7', 'Q15', 'Q17', 'Q18', 'Q37', 'Q38', 'Q39', 'Q40', 'Q52', 'Q73', 'Q74', 'Q75']
mask = ~data_edf.columns.isin(to_drop)
X = data_edf.loc[:,mask]

#data_edf.drop(to_drop,inplace=True,axis=1)
X.dropna()

categorical = ['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9', 'C12', 'C13', 'C14', 'C15', 'S2', 'Q1', 'Q2', 'Q3', 'Q8', 'Q10', 'Q11', 'Q12', 'Q16', 'Q21', 'Q23', 'Q24', 'Q25', 'Q26', 'Q27', 'Q28', 'Q29', 'Q32', 'Q34', 'Q36', 'Q53', 'Q54', 'Q55', 'Q56', 'Q57', 'Q58', 'Q59', 'Q60', 'Q61', 'Q62', 'Q63', 'Q64', 'Q65', 'Q66', 'Q67', 'Q68', 'Q69', 'Q70', 'Q71', 'Q72']

#categorical = ['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9', 'C12', 'C13', 'C14', 'C15', 'S2', 'Q1', 'Q2', 'Q3', 'Q7', 'Q8', 'Q10', 'Q11', 'Q12', 'Q16', 'Q21', 'Q23', 'Q24', 'Q25', 'Q26', 'Q27', 'Q28', 'Q29', 'Q32', 'Q34', 'Q36', 'Q39', 'Q53', 'Q54', 'Q55', 'Q56', 'Q57', 'Q58', 'Q59', 'Q60', 'Q61', 'Q62', 'Q63', 'Q64', 'Q65', 'Q66', 'Q67', 'Q68', 'Q69', 'Q70', 'Q71', 'Q72', 'Q73', 'Q74', 'Q75']

for a in categorical:
	X[a] = X[a].astype('category')
	X[a] = lb.fit_transform(X[a])
# ENEDIS data
#data_enedis = pd.read_csv("../data_EDF/consommation-electrique-par-secteurs-dactivite.csv", sep = ";")


X['is_train'] = np.random.uniform(0, 1, len(X)) <= .75

train, test = X[X['is_train']==True], X[X['is_train']==False]

# features = ['C1', 'C2', 'C3']
features = list(X)[:-1]

clf = RandomForestClassifier(n_jobs=2)
y, _ = pd.factorize(train['TARGET'])
clf.fit(train[features], y)

preds = [clf.predict(test[features])]
print(pd.crosstab(test['TARGET'], preds, rownames=['actual'], colnames=['preds']))