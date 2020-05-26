# -*- coding: utf-8 -*-
"""
Created on Sun May 24 13:17:07 2020

@author: ricci
"""

from sklearn.cluster import KMeans,AgglomerativeClustering
from sklearn.preprocessing import StandardScaler,MinMaxScaler
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import dexplot as dxp
import seaborn as sns
import gower
import scipy.cluster.hierarchy as sch
from sklearn.preprocessing import normalize
from sklearn.metrics import silhouette_samples, silhouette_score
organics_dummy=pd.read_csv('C:/Users/ricci/Desktop/Coding & Techniques/GitHub/Classification & Clustering/organics_dummy.csv')
organics_dt=pd.read_csv('C:/Users/ricci/Desktop/Coding & Techniques/GitHub/Classification & Clustering/organics_dt.csv')

#%%
#Hierarchical Clustering

#plot dendrogram
plt.figure(figsize=(10, 7))  
plt.title("Dendrograms")
dend = sch.dendrogram(sch.linkage(normalize(organics_dt), method='ward'))

#AgglomerativeClustering
def agg_cluser(data):
    #normalize data
    scaled_data=pd.DataFrame(normalize(organics_dt))
    scaled_data.columns=data.columns
    
    #iterate number of clusters
    n_cluster_list=[2,3,4,5,6,7,8]
    for i in n_cluster_list:
        cluster = AgglomerativeClustering(n_clusters=i, affinity='euclidean', linkage='ward')  
        y_cluster=cluster.fit_predict(scaled_data)

    #SILHOUETTE ANALYSIS
# =============================================================================
#    Silhouette coefficients (as these values are referred to as) near +1 
#    indicate that the sample is far away from the neighboring clusters. 
#    A value of 0 indicates that the sample is on or very close to the decision boundary 
#    between two neighboring clusters and negative values indicate that those samples 
#    might have been assigned to the wrong cluster.
# =============================================================================
        silhouette_avg = silhouette_score(scaled_data, y_cluster)

        print("For n_clusters =", i,
              "The average silhouette_score is :", silhouette_avg)

agg_cluser(organics_dt)

#normalize(this method normalize data by row, so each observation has all features under the same scale)
scaled_data=pd.DataFrame(normalize(organics_dt))
scaled_data.columns=organics_dt.columns
cluster = AgglomerativeClustering(n_clusters=4, affinity='euclidean', linkage='ward')  
y_cluster=cluster.fit_predict(scaled_data)
unique, counts = np.unique(y_cluster, return_counts=True)
dict(zip(unique, counts))       
scaled_data.insert((scaled_data.shape[1]),'group',y_cluster)

##########using min_max_scaler
#min_max_scaler = MinMaxScaler()
#x_scaled = pd.DataFrame(min_max_scaler.fit_transform(organics_dt))
#x_scaled.columns=organics_dt.columns
#cluster = AgglomerativeClustering(n_clusters=4, affinity='euclidean', linkage='ward')  
#y_cluster=cluster.fit_predict(x_scaled)
#unique, counts = np.unique(y_cluster, return_counts=True)
#dict(zip(unique, counts)) 
#x_scaled.insert((x_scaled.shape[1]),'group',y_cluster)
#%%
#K-Means Clustering
kmeans = KMeans(
    n_clusters=3, init='random',
    n_init=10, max_iter=200, 
    tol=1e-04, random_state=0
)
kmeans.fit(scaled_data)
y_cluster = kmeans.predict(scaled_data)
unique, counts = np.unique(y_cluster, return_counts=True)
dict(zip(unique, counts))

#%%
#visualization
plt.figure(figsize=(10, 7))  
sns.scatterplot(x='DemAffl', y='PromSpend',hue='group',palette=['blue','orange','green','purple'],data=scaled_data) 
plt.xlabel('DemAge',fontsize = '12')
plt.ylabel('PromSpend',fontsize = '12')

plt.figure(figsize=(10, 7))  
sns.scatterplot(x='DemAge', y='PromTime',hue='group',palette=['blue','orange','green','purple'],data=scaled_data) 
plt.xlabel('DemAge',fontsize = '12')
plt.ylabel('PromTime',fontsize = '12')
plt.legend(loc="upper left")

#summarize result
cluster_result=organics_dt.copy(deep=False)
cluster_result.insert((cluster_result.shape[1]),'group',y_cluster)

cluster_view=cluster_result.groupby(['group']).agg({
        'TargetBuy': 'sum',
        'DemAffl': 'mean',
        'DemGender': 'mean',
        'DemClusterGroup': 'mean',
        'DemReg': 'mean',
        'PromClass': 'mean',
        'PromSpend': 'mean',
        'PromTime': 'mean',
        'DemAge': 'mean',
        }).round(1)

cluster_result['group']=cluster_result['group'].astype(str)
cluster_result['DemGender']=cluster_result['DemGender'].astype(str)
dxp.aggplot(agg='DemGender', data=cluster_result, hue='group', normalize='DemGender',figsize=(8, 4),stacked=True)
sns.barplot(x="kmeans", y="DemAge", data=cluster_result)

g = sns.FacetGrid(cluster_result, row="group", margin_titles=True)
bins = np.linspace(0, 60, 13)
g.map(plt.hist, "DemAge", color="steelblue", bins=bins)
