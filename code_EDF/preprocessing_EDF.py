import pandas as pd
import numpy as np
%pylab inline

# EDF data
tr_input = pd.read_csv("../data_EDF/training_inputs.csv", sep = ";")
output = pd.read_csv("../data_EDF/challenge_output_data_training_file_predict"
                     "_which_clients_reduced_their_consumption.csv", sep = ";")
data_edf = pd.merge(left=tr_input, right=output, how='left', on="ID")

# ENEDIS data
data_enedis = pd.read_csv("../data_EDF/consommation-electrique-par-secteurs-dactivite.csv", sep = ";")

# Clean it
to_keep = ['Code IRIS',
           'Nb sites secteur résidentiel Enedis',
           'Conso totale secteur résidentiel Enedis (MWh)',
           'Conso moyenne secteur résidentiel Enedis (MWh)']
data_enedis_kept = data_enedis[to_keep]
data_enedis_kept = data_enedis_kept.dropna()

# Merge'em
data = pd.merge(left=data_edf, right=data_enedis_kept, how='left', left_on="COD_INSEE", right_on="Code IRIS")
