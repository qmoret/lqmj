import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import train_test_split
from sklearn.model_selection import StratifiedKFold
import xgboost as xgb
import matplotlib.pyplot as plt
import pylab
pylab.ion()

#-------------------------------------------------------------------------------
# Compare'em
if False:
    EDF=set(data_enedis_kept['COD_IRIS'])
    ENEDIS=set(data_edf['COD_IRIS'])
    len(EDF-ENEDIS)
    len(ENEDIS-EDF)

# Merge'em
#data = pd.merge(left=data_edf, right=data_enedis_clusters, how='left', on="COD_IRIS")
#data.shape
#data.head()
#-------------------------------------------------------------------------------


#-------------------------------------------------------------------------------
# Import data
#train = pd.read_csv("../data_EDF/training_inputs.csv", sep = ";")
#train_out = pd.read_csv("../data_EDF/challenge_output_data_training_file_predict"
#                     "_which_clients_reduced_their_consumption.csv", sep = ";")
#X = train
#y = train_out['TARGET']
output = pd.read_csv("../data_EDF/challenge_output_data_training_file_predict"
                     "_which_clients_reduced_their_consumption.csv", sep = ";")
data_edf = pd.merge(left=tr_input, right=output, how='left', on="ID")

X = pd.read_csv("preprocessed_edf.csv", index_col=0)
y = X['TARGET']
X = X.drop(['TARGET'], axis=1)

# XGB parameters
param = {'max_depth': 6, 'eta': 0.1, 'silent': 1,
         'objective': 'binary:logistic', 'num_class': 2}
num_round = 20

X['is_train'] = np.random.uniform(0, 1, len(X)) <= .75
Xtrain, Xtest = X[X['is_train']==True], X[X['is_train']==False]
ytrain, ytest = y[X['is_train']==True], y[X['is_train']==False]

model = xgb.XGBClassifier()
model.fit(Xtrain, ytrain)
ypred = model.predict(Xtest)
yproba = model.predict_proba(Xtest)[1]



np.mean(ypred == ytest)



# Stratified K folds
n_splits = 5
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

    model = xgb.XGBClassifier()
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
