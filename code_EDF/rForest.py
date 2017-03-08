from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.cross_validation import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn import neighbors
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pylab


lb = LabelEncoder()

# EDF data
tr_input = pd.read_csv("../data_EDF/training_inputs.csv", sep = ";")
test_input = pd.read_csv("../data_EDF/testing_inputs.csv", sep = ";")
output = pd.read_csv("../data_EDF/challenge_output_data_training_file_predict"
                     "_which_clients_reduced_their_consumption.csv", sep = ";")
data_edf = pd.merge(left=tr_input, right=output, how='left', on="ID")

# ------------------------------------------------------------------------------
# Imputation des valeurs manquantes par plus proche voisins

def fill_na(data):
 
    n_neighbors = 10
    cols = list(data)
    for col in cols:
        if data[col].isnull().sum() > 0:
            print("%s : %f%% de valeurs manquantes" %
                  (col, 100 * data[col].isnull().sum() / data.shape[0]))
            knn = neighbors.KNeighborsRegressor(n_neighbors)
            data_temp = data.loc[~data[col].isnull(), :]
            mask = ~data.columns.isin(cols)
            X = data_temp.loc[:, mask]
            null_index = data[col].isnull()
            y_ = knn.fit(X, data_temp[col]).predict(data.loc[null_index, mask])
            data.loc[null_index, col] = y_
            data[col] = data[col].astype(float)
 
    return data

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
X = fill_na(X)

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

# ------------------------------------------------------------------------------
# Encode

for a in categ:
    X[a] = X[a].astype('category')
    X[a] = lb.fit_transform(X[a])
    clean_test_data[a] = clean_test_data[a].astype('category')
    clean_test_data[a] = lb.fit_transform(clean_test_data[a])

# ------------------------------------------------------------------------------
# Training

train = X
features = list(X)[:-1]

y, _ = pd.factorize(train['TARGET'])

# Stratified K folds
n_splits = 5
skf = StratifiedKFold(n_splits, shuffle=True)
accuracies, index = np.zeros(n_splits), 0
conf_mat = []

# fit model no training data
ypred = np.zeros(len(y))
yprob = np.zeros(len(y))

for train_index, test_index in skf.split(X, y):
    Xtrain, Xtest = X.ix[train_index], X.ix[test_index]
    ytrain, ytest = y[train_index], y[test_index]

    clf = RandomForestClassifier(n_jobs=3)
    clf.fit(train[features], ytrain)
    ypred[test_index] = clf.predict(Xtest)
    probas = clf.predict_proba(Xtest)

    index_of_class_1 = 1 - ytrain.values[0] # 0 if the first sample is positive, 1 otherwise
    yprob[test_index] = probas[:, index_of_class_1]

    accuracies[index] = np.mean(ypred == ypred[test_index])
    index += 1

# ROC curve
fpr, tpr, tres = metrics.roc_curve(y, yprob, pos_label = 1)
auc = metrics.auc(fpr, tpr)

plt.plot(fpr, tpr, '-', color='red', label = 'XGB AUC = %.3f' %auc)
plt.xlabel('False Positive Rate', fontsize=16)
plt.ylabel('True Positive Rate', fontsize=16)
plt.title('ROC curves', fontsize=16)
plt.legend(loc="lower right")

# ------------------------------------------------------------------------------
# Prediction

# test = clean_test_data

# test['prediction0'] = clf.predict_proba(test[features])[:,0]
# filth['prediction0'] = np.random.uniform()

# result = pd.concat([test, filth])
# result['prediction0'].to_csv("result1.csv")
