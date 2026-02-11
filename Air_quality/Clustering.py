import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.preprocessing import StandardScaler,LabelEncoder
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA



file_path="air_quality.csv"
if not os.path.exists(file_path):
    raise FileNotFoundError("File csv non trovato al percorso: ",file_path)

df=pd.read_csv(file_path,low_memory=False,na_values=['','-','NA','ND','n/a'],nrows=1000000)

print(f"Grandezza dataset: {df.shape}")
print(f"Prime righe: {df.head()}")
print("Info su dataset:")
df.info()

#Data-Cleaning
target='status'
df.dropna(axis=1,how="all",inplace=True)        #elimino colonne vuote
df.dropna(axis=0,how="any",subset=target,inplace=True)      #elimino righe Nan sul target


#porto tutti in numero per fare la matrice di correlazione
#Verrà usato il Label Encoder per semplicità e velocità ma bisognerebbe usare il Lbael Encoder per la features ordinali, mentre one hot encoding per le feature nominali
#Un uso di one hot encoding su tante colonne va ad aumante drasticamnete il numero di colonne del dataset... andando a provocare più danni che benefici

lencoder=LabelEncoder()
df['date']=lencoder.fit_transform(df['date'])
df['sitename']=lencoder.fit_transform(df['sitename'])
df['county']=lencoder.fit_transform(df['county'])
df['pollutant']=lencoder.fit_transform(df['pollutant'])
df['target']=lencoder.fit_transform(df[target]) #mi porto il target in numerico per la correlazione poi la droppo
#target mi serve in stringa per un confortno succesivo sul lavoro fatto dai clusters

#Matrice di correlazione
numeric_cols_data=df.select_dtypes(include=[np.number])
cm=numeric_cols_data.corr()
sns.heatmap(cm,annot=True, cmap="coolwarm")
plt.title("Matrice di correlazione")
plt.show()

col_drop=['date','sitename','county','no','windspeed','winddirec','longitude','latitude','siteid'] #|r|<0.20, rumore [K-means vuole solo relazioni lineari, quelle poco lineari le va a forzare il che provoca rumore]

df.drop(columns=col_drop, inplace=True) #pochissima correlazione con target--> rumore
df.drop(columns='pm2.5_avg', inplace=True)  #media calcolata di una feature presente--> data leakege
df.drop(columns='pm10_avg', inplace=True)  #media calcolata di una feature presente--> data leakege
df.drop(columns='so2_avg', inplace=True)  #media calcolata di una feature presente--> data leakege
df.drop(columns='target', inplace=True)  #colonna aggiunta per la matrice di correlazione riguardane target in numerico
#df.drop(columns='aqi', inplace=True)  #tolgo perchè status il target dipende da questa variabile qua


#seleziono nomi colonne numeriche
name_num_col=df.select_dtypes(include=[np.number]).columns.tolist()
df.dropna(axis=0,how="any",subset=name_num_col,inplace=True)    #rimuovo righe Nan solo nelle colonne numeriche con cui lavorerò

print("Dataset post pulizia:")
df.info()

#PRE-PROCESSING
X=df[name_num_col]



#niente split nel cluster

#Scale
scaler=StandardScaler()
x_scaled=scaler.fit_transform(X)

#numero cluster=classi assunte da target
num_cluster=len(df[target].unique())
print(f"\nValori assunti dal target: {df[target].unique()} per un totale di {num_cluster} classi")
print(f"Conteggio valori assunti da target: {df[target].value_counts()}")   #sbilancito per Very Unhealty, si potrebbe raggruppare

#MODELLO --> clustering kmean (tanti cluster quanti le classi che puo assumere il target)

kmeans=KMeans(n_clusters=num_cluster,random_state=42)
labels=kmeans.fit_predict(x_scaled) #modellomlavora con dati X scalati

#crosstab
crosstab=pd.crosstab(df[target],labels,colnames=["cluster"])    #faccio un cross tab
print("\nCrosstab:")
print(crosstab)

#mapping del crosstab
cluster_to_target=crosstab.idxmax(axis=0).to_dict() #per fare un mapping associo il cluster alal classe dominante
print("Mapping cluster--target ",cluster_to_target)

#VALUTAZIONE

target_pred=pd.Series(labels,index=df.index).map(cluster_to_target)

accuracy=(target_pred==df[target]).mean()   #clacolo una percentuale in decimale di quando la classe reale target appartiene al cluster
print(f"\nAccuracy globale:{accuracy*100:.2f}%")

#VISUALIZZAZIONE
pca=PCA(n_components=2) #2D
feature_compressed=pca.fit_transform(x_scaled)

plt.scatter(
    feature_compressed[:,0],
    feature_compressed[:,1],
    c=labels,
    s=8
)
plt.title("PCA 2D (K=5 clusters)")
plt.colorbar(label="clusters")
plt.show()

