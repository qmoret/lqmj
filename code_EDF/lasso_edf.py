import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn.linear_model import Lasso
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

# Lasso parameters
param_grid = {'alpha':[0.0001, 0.001, 0.01]}

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

    las = GridSearchCV(Lasso(), param_grid, scoring= 'roc_auc')
    las.fit(Xtrain, ytrain)
    yprob[test_index] = las.predict(Xtest)

    print(las.best_estimator_)

# ROC curve
fpr, tpr, tres = metrics.roc_curve(y, yprob, pos_label = 1)
auc = metrics.auc(fpr, tpr)
auc

plt.plot(fpr, tpr, '-', color='red', label = 'XGB AUC = %.3f' %auc)
plt.xlabel('False Positive Rate', fontsize=16)
plt.ylabel('True Positive Rate', fontsize=16)
plt.title('ROC curves', fontsize=16)
plt.legend(loc="lower right")

print("\nCross validated accuracy: %0.2f (+/- %0.2f)\n" %
      (accuracies.mean(), accuracies.std()))

# Save for ensemble
las = pd.DataFrame(yprob, columns=['las'])
las.to_csv("las.csv")

#-------------------------------------------------------------------------------
# New predictions
Xnew = pd.read_csv("pp_testing_i.csv", index_col=0)

# Fit on all data
las = Lasso(alpha=0.0001)
las.fit(X, y)
yprob[test_index] = las.predict(Xnew)

# Predict
ynew_proba = rfc.predict(Xnew)

getin = pd.read_csv("../data_EDF/testing_inputs.csv", sep = ";", usecols=[1])
sub = pd.DataFrame(ynew_proba, index=getin.index)

sub.to_csv("las_submission.csv")
