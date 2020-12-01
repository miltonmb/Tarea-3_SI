import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.cluster import AgglomerativeClustering

#Parte 1
def means():
  for i in range(1,4):
    df = pd.read_csv('Parte 1/datos_'+str(i)+'.csv')
    for j in range(1,6):
      kmeans = KMeans(n_clusters=j).fit(df)
      centroids = kmeans.cluster_centers_
      #print(centroids)
      plt.scatter(df['x'], df['y'], c= kmeans.labels_.astype(float), s=50, alpha=0.5)
      plt.scatter(centroids[:, 0], centroids[:, 1], c='red', s=50)
      plt.savefig("Parte 1/K-means/dataset-"+str(i)+"-"+str(j)+".png")
      plt.cla()
      plt.clf()



def agg_clustering():
  for i in range(1,4):
    df = pd.read_csv('Parte 1/datos_'+str(i)+'.csv')
    for j in range(1,6):
      agg_clust = AgglomerativeClustering(n_clusters=j, linkage="single", distance_threshold=None).fit(df)
      labels = agg_clust.labels_
      plt.scatter(df['x'], df['y'], c = labels)
      plt.savefig("Parte 1/Clustering aglomerativo/Cluster/dataset-"+str(i)+"-"+str(j)+".png")
      plt.cla()
      plt.clf()
  for i in range(1,4):
    k = 0
    for j in range(1,6):
      k = k + 0.25
      if (k == 1.25):
        k = k + 0.25
      print(k)
      agg_clust = AgglomerativeClustering(n_clusters=None, linkage="single", distance_threshold=k).fit(df)
      labels = agg_clust.labels_
      plt.scatter(df['x'], df['y'], c = labels)
      plt.savefig("Parte 1/Clustering aglomerativo/Umbral/dataset-"+str(i)+"-"+str(j)+".png")
      plt.cla()
      plt.clf()

def DB_Scan():
  #EPS 0.25
  for i in range(1,4):
    m  = 0
    df = pd.read_csv('Parte 1/datos_'+str(i)+'.csv')
    for j in range(1,4):
      m = m + 5
      kdb25 = DBSCAN(eps=0.25,min_samples = m)
      kdb25.fit(df)
      y_kmeans = kdb25.fit_predict(df)
      plt.scatter(df["x"], df["y"], c=y_kmeans, s=50, cmap='viridis')
      plt.savefig("Parte 1/DBSCan/dataset-"+ str(i) + "-25-"+str(m)+".png")
      plt.cla()
      plt.clf()

  #EPS 0.35    
  for i in range(1,4):
    m  = 0
    df = pd.read_csv('Parte 1/datos_'+str(i)+'.csv')
    for j in range(1,4):
      m = m + 5
      kdb35 = DBSCAN(eps=0.35,min_samples = m)
      kdb35.fit(df)
      y_kmeans = kdb35.fit_predict(df)
      plt.scatter(df["x"], df["y"], c=y_kmeans, s=50, cmap='viridis')
      plt.savefig("Parte 2/DBSCan/dataset-"+ str(i) + "-35-"+str(m)+".png")
      plt.cla()
      plt.clf()
  
  #EPS 0.50
  for i in range(1,4):
    m  = 0
    df = pd.read_csv('Parte 2/datos_'+str(i)+'.csv')
    for j in range(1,4):
      m = m + 5
      kdb50 = DBSCAN(eps=0.50,min_samples = m)
      kdb50.fit(df)
      y_kmeans = kdb50.fit_predict(df)
      plt.scatter(df["x"], df["y"], c=y_kmeans, s=50, cmap='viridis')
      plt.savefig("Parte 2/DBSCan/dataset-"+ str(i) + "-50-"+str(m)+".png")
      plt.cla()
      plt.clf()
    
  



def main():
  means()
  agg_clustering()
  DB_Scan()
  

  