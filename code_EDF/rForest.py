from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np

lb = LabelEncoder()

# EDF data
tr_input = pd.read_csv("../data_EDF/training_inputs.csv", sep = ";")
test_input = pd.read_csv("../data_EDF/testing_inputs.csv", sep = ";")
output = pd.read_csv("../data_EDF/challenge_output_data_training_file_predict"
                     "_which_clients_reduced_their_consumption.csv", sep = ";")
data_edf = pd.merge(left=tr_input, right=output, how='left', on="ID")

X = data_edf

# ------------------------------------------------------------------------------
# Column Hardcore Cleaning


# NAs
if False:
    for c in X.columns:
        if (sum(X[c].isnull()*100/X.shape[0])>5):
            print('Colonnes %s : %f' %(c, sum(X[c].isnull()*100/X.shape[0])))

shit = ['C1']
dates = ['S3', 'S4', 'S5']
codes = ['COD_INSEE', 'COD_IRIS']
to_many_nas = ['S1', 'S6', 'S7', 'Q6', 'Q7', 'Q15', 'Q17', 'Q18', 'Q26', 'Q35',
                    'Q37', 'Q38', 'Q39', 'Q40', 'Q52', 'Q54', 'Q55','Q56','Q57',
                    'Q73', 'Q74', 'Q75']

data_edf.set_index('ID',inplace = True)

to_drop = shit+dates+codes+to_many_nas

mask_train = ~data_edf.columns.isin(to_drop)
X = data_edf.loc[:,mask_train]

test_input.set_index("ID",inplace=True)

mask_test = ~test_input.columns.isin(to_drop)
test_data = test_input.loc[:,mask_test]

clean_test_data = test_data.dropna()
filth = test_data[~test_data.index.isin(clean_test_data.index)]


# ------------------------------------------------------------------------------
# dropna

X = X.dropna()


# ------------------------------------------------------------------------------
# Categorical data encoding

# Explore
if False:
    for c in X.columns:
        print('Colonnes %s : %s' %(c, X[c].dtype))

categ_all = ['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9', 'C12',
                'C13', 'C14', 'C15', 'S2', 'Q1', 'Q2', 'Q3', 'Q7', 'Q8', 'Q10',
                'Q11', 'Q12', 'Q16', 'Q21', 'Q23', 'Q24', 'Q25', 'Q26', 'Q27',
                'Q28', 'Q29', 'Q32', 'Q34', 'Q36', 'Q39', 'Q53', 'Q54', 'Q55',
                'Q56', 'Q57', 'Q58', 'Q59', 'Q60', 'Q61', 'Q62', 'Q63', 'Q64',
                'Q65', 'Q66', 'Q67', 'Q68', 'Q69', 'Q70', 'Q71', 'Q72', 'Q73',
                'Q74', 'Q75']

categ = list(set(categ_all) - set(to_drop))

# Encode
for a in categ:
    X[a] = X[a].astype('category')
    X[a] = lb.fit_transform(X[a])
    clean_test_data[a] = clean_test_data[a].astype('category')
    clean_test_data[a] = lb.fit_transform(clean_test_data[a])

train, test = X, clean_test_data

features = list(X)[:-1]

clf = RandomForestClassifier(n_jobs=3)
y, _ = pd.factorize(train['TARGET'])
clf.fit(train[features], y)

print(pd.crosstab(test['TARGET'], preds, rownames=['actual'], colnames=['preds']))

test['prediction0'] = clf.predict_proba(test[features])[:,0]
filth['prediction0'] = np.random.uniform()

result = pd.concat([test, filth])
result['prediction0'].to_csv("result1.csv")


print(pd.crosstab(test['TARGET'], preds, rownames=['actual'], colnames=['preds']))
