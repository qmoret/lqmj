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

Xx = pd.read_csv("xgb_submission.csv", index_col=0)
Xr = pd.read_csv("rfc_submission.csv", index_col=0)
Xl = pd.read_csv("Las_submission.csv", index_col=0)
y = pd.read_csv("../data_EDF/challenge_output_data_training_file_predict_which_"
                     "clients_reduced_their_consumption.csv", sep = ";",
                     usecols = [1])
y = y['TARGET']


ens_prob = 0.4*Xx['xgb_proba']+0.5*Xr["rf_proba"]+0.1*Xl["las"]

fpr, tpr, tres = metrics.roc_curve(y, ens_prob, pos_label = 1)
auc = metrics.auc(fpr, tpr)
auc

getin = pd.read_csv("../data_EDF/testing_inputs.csv", sep = ";", usecols=[1])
ens_sub = pd.DataFrame(ens_prob, index=getin.index)

ens_sub.to_csv("ens_submission.csv")
