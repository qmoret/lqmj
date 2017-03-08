import pandas as pd
import numpy as np
%pylab inline

# EDF data
tr_input = pd.read_csv("../data_EDF/training_inputs.csv", sep = ";")
output = pd.read_csv("../data_EDF/challenge_output_data_training_file_predict"
                     "_which_clients_reduced_their_consumption.csv", sep = ";")
data_edf = pd.merge(left=tr_input, right=output, how='left', on="ID")
data_edf['COD_IRIS'] = data_edf['COD_IRIS'].fillna(0).apply(int)
data_edf.shape

# ENEDIS data
data_enedis = pd.read_csv("../data_EDF/consommation-electrique-par-secteurs-dactivite.csv", sep = ";")
data_enedis.shape

# Clean it
to_keep = ['Code IRIS',
           'Nb sites secteur résidentiel Enedis',
           'Conso totale secteur résidentiel Enedis (MWh)',
           'Conso moyenne secteur résidentiel Enedis (MWh)']
data_enedis_kept = data_enedis[to_keep]
data_enedis_kept = data_enedis_kept.dropna()
data_enedis_kept.shape
data_enedis_kept = data_enedis_kept[data_enedis_kept['Code IRIS'].str.find('x')==-1]
data_enedis_kept['Code IRIS'] = data_enedis_kept['Code IRIS'].apply(int)
data_enedis_kept.columns = ['COD_IRIS', 'nb_sites', 'conso_tot', 'conso_moy']

# Merge'em
data = pd.merge(left=data_edf, right=data_enedis_kept, how='left', on="COD_IRIS")
data.shape
