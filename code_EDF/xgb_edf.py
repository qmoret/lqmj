import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import train_test_split
from sklearn.model_selection import StratifiedKFold
import xgboost as xgb
import matplotlib.pyplot as plt
import pylab
from sklearn import neighbors
pylab.ion()

X = pd.read_csv("preprocessed_edf.csv", index_col=0)
y = = pd.read_csv("../data_EDF/challenge_output_data_training_file_predict"
                     "_which_clients_reduced_their_consumption.csv", sep = ";")

# XGB parameters
param = {'max_depth': 6, 'eta': 0.1, 'silent': 1,
         'objective': 'binary:logistic', 'num_class': 2}
num_round = 20

#-------------------------------------------------------------------------------
# Model evaluation
n_splits = 7
skf = StratifiedKFold(n_splits, shuffle=True)
accuracies, index = np.zeros(n_splits), 0
conf_mat = []
len(y)
# fit model no training data
ypred = np.zeros(len(y))
yprob = np.zeros(len(y))

for train_index, test_index in skf.split(X, y):
    print("new fold...")
    Xtrain, Xtest = X.ix[train_index], X.ix[test_index]
    ytrain, ytest = y[train_index], y[test_index]

    model = xgb.XGBClassifier(max_depth=10, n_estimators=200, objective='binary:logistic')
    model.fit(Xtrain, ytrain)
    ypred[test_index] = model.predict(Xtest)
    probas = model.predict_proba(Xtest)

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

print("\nCross validated accuracy: %0.2f (+/- %0.2f)\n" %
      (accuracies.mean(), accuracies.std()))

#-------------------------------------------------------------------------------
# New predictions
Xnew = pd.read_csv("pp_testing.csv", index_col=0)

# Fit on all data
model = xgb.XGBClassifier(max_depth=10, n_estimators=200, objective='binary:logistic')
model.fit(X, y)

# Predict
ynew_pred = model.predict(Xnew)
ynew_proba = model.predict_proba(Xnew)[:,1]

sub = pd.DataFrame(ynew_proba, index=Xnew.index)

sub.to_csv("xgb_submission.csv")
