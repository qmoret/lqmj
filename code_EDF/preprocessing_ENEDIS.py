import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import scale
from scipy.cluster.hierarchy import dendrogram, linkage
%pylab inline
pylab.ion()

# ENEDIS data
data_enedis = pd.read_csv("../data_EDF/consommation-electrique-par-secteurs-dactivite.csv", sep = ";")
data_enedis.shape

#-------------------------------------------------------------------------------
# Clean it
# Filter columns
to_keep = ['Code IRIS',
           'Nb sites secteur résidentiel Enedis',
           'Conso totale secteur résidentiel Enedis (MWh)',
           'Conso moyenne secteur résidentiel Enedis (MWh)']
data_enedis_kept = data_enedis[to_keep]
data_enedis_kept.columns = ['COD_IRIS', 'nb_sites', 'conso_tot', 'conso_moy']
data_enedis_kept.shape
sum(data_enedis_kept.groupby('COD_IRIS').COD_IRIS.nunique())

# Clean XXX
data_enedis_kept['COD_IRIS'] = data_enedis_kept['COD_IRIS'].str.replace('x', '0')
data_enedis_kept.shape

# group by max year
data_enedis_kept = data_enedis_kept.groupby(['COD_IRIS'], as_index=False).mean()
data_enedis_kept.shape
sum(data_enedis_kept.groupby('COD_IRIS').COD_IRIS.nunique())


# Drop NA
data_enedis_kept = data_enedis_kept.dropna()
data_enedis_kept.shape

# bails
data_enedis_kept['COD_IRIS'] = data_enedis_kept['COD_IRIS'].apply(int)



sum(data_enedis_kept.groupby('COD_IRIS').COD_IRIS.nunique())


# Cluster it
data = data_enedis_kept
mask = ~data.columns.isin(['COD_IRIS'])

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
kmeans_4 = KMeans(n_clusters=4, random_state=0).fit(X)
kmeans_4.labels_.shape

data_enedis_kept['cluster'] = kmeans_4.labels_

# Check
data_enedis_kept.groupby(['cluster'])['cluster'].count()

data_enedis_clusters = data_enedis_kept[['COD_IRIS','cluster']]
data_enedis_clusters.to_csv('clusters.csv')
