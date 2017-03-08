import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

training = True

if training:
    X = pd.read_csv("../data_EDF/training_inputs.csv", sep = ";")
else:
    X = pd.read_csv("../data_EDF/testing_inputs.csv", sep = ";")

# ------------------------------------------------------------------------------
# Column Hardcore Cleaning
# NAs
shit = ['C1']
dates = ['S3', 'S4', 'S5']
codes = ['COD_INSEE', 'COD_IRIS']
to_many_nas = ['S1', 'S6', 'S7', 'Q6', 'Q7', 'Q15', 'Q17', 'Q18', 'Q26', 'Q35',
                    'Q37', 'Q38', 'Q39', 'Q40', 'Q52', 'Q54', 'Q55','Q56','Q57',
                    'Q73', 'Q74', 'Q75']

to_drop = shit+dates+codes+to_many_nas

X.set_index('ID',inplace = True)

mask = ~X.columns.isin(to_drop)
X = X.loc[:,mask]


# ------------------------------------------------------------------------------
# dropna
if False:
    X = X.dropna()
    print(X.shape)

# ------------------------------------------------------------------------------
# Categorical data encoding

categ_all = ['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9', 'C12',
                'C13', 'C14', 'C15', 'S2', 'Q1', 'Q2', 'Q3', 'Q7', 'Q8', 'Q10',
                'Q11', 'Q12', 'Q16', 'Q21', 'Q23', 'Q24', 'Q25', 'Q26', 'Q27',
                'Q28', 'Q29', 'Q32', 'Q34', 'Q36', 'Q39', 'Q53', 'Q54', 'Q55',
                'Q56', 'Q57', 'Q58', 'Q59', 'Q60', 'Q61', 'Q62', 'Q63', 'Q64',
                'Q65', 'Q66', 'Q67', 'Q68', 'Q69', 'Q70', 'Q71', 'Q72', 'Q73',
                'Q74', 'Q75']

categ = list(set(categ_all) - set(to_drop))

# Encode
if False:
    lb = LabelEncoder()
    for a in categ:
    	X[a] = X[a].astype('category')


le = LabelEncoder()
mapping = dict()
for col, dtype in zip(X.columns, X.dtypes):
    if dtype == 'object':
        X[col] = X[col].apply(lambda s: str(s))
        # Replace 0 and NaNs with unique label : 'None'
        #data[col] = data[col].where(~data[col].isin(['0', 'nan']), 'None')
        X[col] = le.fit_transform(X[col])
        mapping[col] = dict(zip(le.inverse_transform(
            X[col].unique()), X[col].unique()))


if training:
    X.to_csv("pp_training.csv")
else:
    X.to_csv("pp_testing.csv")
