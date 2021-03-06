import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import scale
import matplotlib.pyplot as plt
import pylab
from sklearn import neighbors
pylab.ion()

X = pd.read_csv("pp_training_e_i.csv", index_col=0)
y = pd.read_csv("../data_EDF/challenge_output_data_training_file_predict_which_"
                     "clients_reduced_their_consumption.csv", sep = ";",
                     usecols = [1])
y = y['TARGET']

# RFC parameters
param_grid = {'n_estimators':[100, 200, 300, 400], 'max_depth':[3,5,7,10]}

#-------------------------------------------------------------------------------
# Model evaluation
n_splits = 7
skf = StratifiedKFold(n_splits, shuffle=True)
accuracies, index = np.zeros(n_splits), 0

# fit model no training data
ypred = np.zeros(len(y))
yprob = np.zeros(len(y))

for train_index, test_index in skf.split(X, y):
    print("new fold...")
    Xtrain, Xtest = X.ix[train_index], X.ix[test_index]
    ytrain, ytest = y[train_index], y[test_index]

    rfc = GridSearchCV(RandomForestClassifier(n_jobs=2), param_grid, scoring= 'roc_auc')
    rfc.fit(Xtrain, ytrain)
    ypred[test_index] = rfc.predict(Xtest)
    probas = rfc.predict_proba(Xtest)

    index_of_class_1 = 1 - ytrain.values[0] # 0 if the first sample is positive, 1 otherwise
    yprob[test_index] = probas[:, index_of_class_1]

    print(rfc.best_estimator_)

# ROC curve
fpr, tpr, tres = metrics.roc_curve(y, yprob, pos_label = 1)
auc = metrics.auc(fpr, tpr)
auc

# Feature importance
imp = model.feature_importances_ * 100
feat = X.columns
feat_imp = pd.DataFrame(imp, feat, columns=['imp'])

feat_imp.sort('imp', ascending = False)

selected = feat_imp[feat_imp['imp']>2].index

plt.plot(fpr, tpr, '-', color='red', label = 'XGB AUC = %.3f' %auc)
plt.xlabel('False Positive Rate', fontsize=16)
plt.ylabel('True Positive Rate', fontsize=16)
plt.title('ROC curves', fontsize=16)
plt.legend(loc="lower right")

print("\nCross validated accuracy: %0.2f (+/- %0.2f)\n" %
      (accuracies.mean(), accuracies.std()))

# Save for ensemble
rf = pd.DataFrame(yprob, columns=['rf_proba'])
rf.to_csv("rf.csv")

#-------------------------------------------------------------------------------
# New predictions
Xnew = pd.read_csv("pp_testing_e_i.csv", index_col=0)

# Fit on all data
rfc = RandomForestClassifier(n_jobs=2, n_estimators=400, max_depth=10)
rfc.fit(X, y)

# Predict
ynew_pred = rfc.predict(Xnew)
ynew_proba = rfc.predict_proba(Xnew)[:,1]

getin = pd.read_csv("../data_EDF/testing_inputs.csv", sep = ";", usecols=[1])
sub = pd.DataFrame(ynew_proba, index=getin.index)

sub.to_csv("rfc_submission.csv")
