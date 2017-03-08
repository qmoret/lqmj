import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import train_test_split
from sklearn.model_selection import StratifiedKFold
import xgboost as xgb


train = pd.read_csv("../data_EDF/training_inputs.csv", sep = ";")
train_out = pd.read_csv("../data_EDF/challenge_output_data_training_file_predict"
                     "_which_clients_reduced_their_consumption.csv", sep = ";")
X = train
y = train_out['TARGET']

# XGB parameters
param = {'max_depth': 6, 'eta': 0.1, 'silent': 1,
         'objective': 'binary:logistic', 'num_class': 2}
num_round = 20


# Stratified K folds
n_splits = 5
skf = StratifiedKFold(n_splits, shuffle=True)
accuracies, index = np.zeros(n_splits), 0
conf_mat = []

for train_index, test_index in skf.split(X, y):

    Xtrain, Xtest = X.ix[train_index], X.ix[test_index]
    ytrain, ytest = y[train_index], y[test_index]

    label_train = ytrain
    label_test = ytest
    dtrain = xgb.DMatrix(Xtrain, label=label_train)
    dtest = xgb.DMatrix(Xtest, label=label_test)
    bst = xgb.train(param, dtrain, num_round)
    ypred = bst.predict(dtest)

    accuracies[index] = np.mean(ypred == ytest)

    ytest_lab = ytest
    ytest_lab.index = range(1, len(ytest_lab) + 1)
    ytest_lab.name = 'Actual'
    ypred_lab = pd.Series(ypred, name='Predicted')
    try:
        conf_mat = conf_mat + pd.crosstab(ytest_lab, ypred_lab)
    except:
        conf_mat = pd.crosstab(ytest_lab, ypred_lab)
    index += 1


for feat, imp in feat_imp.items():
    feat_imp[feat] = imp / n_splits
feat_imp = sorted(feat_imp.items(), key=lambda x: x[1], reverse=True)[:10]
print("\nCross validated accuracy: %0.2f (+/- %0.2f)\n" %
      (accuracies.mean(), accuracies.std()))

print('\nCross validated Confusion matrix :')
print(conf_mat)
