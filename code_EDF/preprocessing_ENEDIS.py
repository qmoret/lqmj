import pandas as pd
import numpy as np
%pylab inline
pylab.ion()

# ENEDIS data
data_enedis = pd.read_csv("../data_EDF/consommation-electrique-par-secteurs-dactivite.csv", sep = ";")
data_enedis.shape

# Clean it
to_keep = ['Code IRIS',
           'Année',
           'Nb sites secteur résidentiel Enedis',
           'Conso totale secteur résidentiel Enedis (MWh)',
           'Conso moyenne secteur résidentiel Enedis (MWh)']
data_enedis_kept = data_enedis[to_keep]
data_enedis_kept = data_enedis_kept.dropna()
data_enedis_kept.shape
data_enedis_kept = data_enedis_kept[data_enedis_kept['Code IRIS'].str.find('x')==-1]
data_enedis_kept['Code IRIS'] = data_enedis_kept['Code IRIS'].apply(int)
data_enedis_kept.columns = ['COD_IRIS', 'annee', 'nb_sites', 'conso_tot', 'conso_moy']

# Select only 2015...
#data_enedis_kept = data_enedis_kept[data_enedis_kept['annee'] == 2015]

# ... or group by max year
idx = data_enedis_kept.groupby(['COD_IRIS'])['annee'].transform(max) == data_enedis_kept['annee']


data_enedis_kept = data_enedis_kept[idx]
data_enedis_kept.shape


# Cluster it
from sklearn.cluster import KMeans
from sklearn.preprocessing import scale
from scipy.cluster.hierarchy import dendrogram, linkage

data = data_enedis_kept
mask = ~data.columns.isin(['COD_IRIS', 'annee'])

X = data.loc[:, mask]

# Evaluate number of clusters with dendrogram
# Kmeans pour réduire à 1000 groupes
if False:
    kmeans_ = KMeans(n_clusters=1000, random_state=0).fit(X)
    X_ = kmeans_.cluster_centers_
    # Générer la matrice des liens
    Z = linkage(X_, method='ward', metric='euclidean')
    # Affichage du dendrogramme
    plt.title("CLUS")
    dendrogram(Z, orientation='left', color_threshold=0)
    plt.show()

    X = X.T.values #Transpose values
    Y = pdist(X)
    Z = linkage(Y)
    dendrogram(Z, labels = df.columns)

# On retient 5 cluster
kmeans_5 = KMeans(n_clusters=5, random_state=0).fit(X)
kmeans_5.labels_.shape

data_enedis_kept['cluster'] = kmeans_5.labels_

# Check
data_enedis_kept.groupby(['cluster'])['cluster'].count()

data_enedis_clusters = data_enedis_kept[['COD_IRIS','cluster']]
