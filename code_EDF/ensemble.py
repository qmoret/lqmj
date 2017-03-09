import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn.cross_validation import train_test_split
from sklearn.model_selection import StratifiedKFold
import xgboost as xgb
import matplotlib.pyplot as plt
import pylab
from sklearn import neighbors
pylab.ion()

Xx = pd.read_csv("xgboo.csv", index_col=0)
Xr = pd.read_csv("rf.csv", index_col=0)
Xl = pd.read_csv("las.csv", index_col=0)
y = pd.read_csv("../data_EDF/challenge_output_data_training_file_predict_which_"
                     "clients_reduced_their_consumption.csv", sep = ";",
                     usecols = [1])
y = y['TARGET']


yprob = 0.4*Xx['xgb_proba']+0.5*Xr["rf_proba"]+0.1*Xl["las"]

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
