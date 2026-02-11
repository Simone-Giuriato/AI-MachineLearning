import os 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.preprocessing import StandardScaler,LabelEncoder
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score,silhouette_score

file_path="champions_league.csv"
if not os.path.exists(file_path):
    raise FileNotFoundError(f"File csv non trovato al percoso{file_path}")

#Carciamneto dataset
df=pd.read_csv(file_path,low_memory=False,na_values=['','-','NA','ND','n/a'])

print(f"Grandezza dataset {df.shape}")
print(f"Prime righe dataset {df.info()}")
print("Info generali")
df.info()
target='MatchResult'


#DATA CLEANING
df.dropna(axis=1,how="all",inplace=True)
df.dropna(axis=0,how="any",subset=target,inplace=True)

print("Valori che assumone features non numeriche:")
print("AwayTeam")
print(df['AwayTeam'].unique())
print("HomeTeam")
print(df['HomeTeam'].unique())
print("MatchResult")
print(df['MatchResult'].unique())


#noto che ci sono variabili categoriche--> converto in numero con labelEncoder
le_home=LabelEncoder()
le_away=LabelEncoder()
le_result=LabelEncoder()

df['HomeTeam']=le_home.fit_transform(df['HomeTeam'])
df['AwayTeam']=le_away.fit_transform(df['AwayTeam'])
#df['MatchResult']=le_result.fit_transform(df['MatchResult'])


print("\nValori che assumone features non numeriche(post encoding):")
print("AwayTeam")
print(df['AwayTeam'].unique())
print("HomeTeam")
print(df['HomeTeam'].unique())
print("MatchResult")
print(df['MatchResult'].unique())


"""
#MATRICE DI CORRELAZIONE LINEARE
numeric_cols=df.select_dtypes(include=[np.number]).columns.tolist()
corr=df[numeric_cols].corr()
sns.heatmap(corr,annot=True,cmap="Blues")
plt.title("Matrice correlazione lienare")
plt.show()

#CLUSTER può riconsocere relazioni/pattern non lineari nascoste quindi tengo le features con poca correlazione col target, reputo matrice correlazione sostanzialmente inutile
"""


numeric_cols=df.select_dtypes(include=[np.number]).columns.tolist()
#numeric_cols.remove(target)

print(f"\nFeatures numeriche: {numeric_cols}")
num_clu=len(df[target].unique())    #tanti cluster=k=numero classi asunte da target
print(f"Numero classi assunte da target: {num_clu}")
print(f"Valori assunti dal target: {df[target].value_counts()}")

#PRE-PROCESSING
X=df[numeric_cols]

#cluster, ninete split

#Scaling
scaler=StandardScaler()
x_scaled=scaler.fit_transform(X)

#MODELLO cluster-->kmeans

kmeans=KMeans(n_clusters=num_clu,random_state=10)
labels=kmeans.fit_predict(x_scaled)


#crosstab
crosstab=pd.crosstab(df[target],labels,colnames=["clusters"])
print("\nCrosstab")
print(crosstab)

#mapping corsstab

cluster_to_target=crosstab.idxmax(axis=0).to_dict() #assegno al cluster, la classe dominante
print(f"Mapping cluster-->target {cluster_to_target}")

#Valutazione modello
target_pred=pd.Series(labels,index=df.index).map(cluster_to_target)

accuracy=(df[target]==target_pred).mean()
print(f"Accuracy globale: {accuracy *100:.2f}%")

print(f"Silhouette score {silhouette_score(x_scaled,labels)*100:.2f}%")  #numero di cluster non appropraito perchè silhouette score neanche 20%, i clster esistono ma non sono ben separati

#Visualizzazione
pca=PCA(n_components=2)
features_compressed=pca.fit_transform(x_scaled)

plt.scatter(
    features_compressed[:,0],
    features_compressed[:,1],
    c=labels,
    s=10,
)
plt.title(f"PCA 2d k={num_clu}")
plt.grid()
plt.colorbar(label="clusters")
plt.show()
