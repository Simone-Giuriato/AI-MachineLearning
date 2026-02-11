import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA


#INDIVIDUAZIONE DEL PERCORSO
file_path="travel.csv"
if not os.path.exists(file_path):
    raise FileNotFoundError(f"File csv not found al percorso: {file_path}")

#CARICAMENTO DATASET
df=pd.read_csv(file_path,low_memory=False,na_values=['-','','NA','ND','n/a'])   #poche righe le carico tutte, sost. i valori -'',NA... con Nan di pandas

print("\nInfo generali sul dataFrame:")
print(f"Grandezza dataframe: {df.shape}")
print(f"Prime righe: {df.head()}")
df.info()   #noto che qualche features assume tipo stringa

target='cancelled'

#DATA CLEANING
df.dropna(axis=1,how="all",inplace=True)    #elimino colonne tutte vuote
df.dropna(axis=0,how="any",inplace=True) #elimino righe vuote perchè kmeans non le sa gestire (sia quelle su target ma anche sulle altre)


print("Valori assunti da feautres stringa: ")
print("\ncountry")
print(df['country'].unique())
print("\nctravel_type")
print(df['travel_type'].unique())
print("\ntransport")
print(df['transport'].unique())
print("\ntravel_insurance")
print(df['travel_insurance'].unique())
print("\nseason")
print(df['season'].unique())
print("\ncancelled (TARGET)")
print(df['cancelled'].unique())

#Label Encoder
#Kmeans lavora con valori numeri, motivo per cui devo trasformare le features tipo obect/stringhe in numeri
#Le varibili categoriche (stringhe) si dividono in nominali e ordinali:
    #Oridnal Encoding--> variabili ordinali che hanno un ordine
    #One Hot Encoding --> variabili nominali 
#Per semplicità userò Label Encoder poichè One hot Encoding (scelta più giusta teoricamnete) mi va ad aggiungere una colonna per ogni classe che la features assume, facendomi "scoppiare" il dataset


le_country=LabelEncoder()
df['country']=le_country.fit_transform(df['country'])

le_traveltype=LabelEncoder()
df['travel_type']=le_traveltype.fit_transform(df['travel_type'])

le_transport=LabelEncoder()
df['transport']=le_transport.fit_transform(df['transport'])

le_season=LabelEncoder()
df['season']=le_season.fit_transform(df['season'])

le_travelInsurance=LabelEncoder()
df['travel_insurance']=le_travelInsurance.fit_transform(df['travel_insurance'])

le_target=LabelEncoder()
df['target']=le_target.fit_transform(df['cancelled'])   #mi servirà dopo post-prima esecuzione con matrice correlazione vediamo se migliora


print("Valori assunti da feautres stringa: ")
print("\ncountry")
print(df['country'].unique())
print("\nctravel_type")
print(df['travel_type'].unique())
print("\ntransport")
print(df['transport'].unique())
print("\ntravel_insurance")
print(df['travel_insurance'].unique())
print("\nseason")
print(df['season'].unique())
print("\ncancelled (TARGET)")   
print(df['target'].unique())

#Dopo la prima esecuzione mi accorgo che posso eliminare del rumore/features inutili per la divisione in cluster
#matrice di correlazione:
numeric_cols=df.select_dtypes(include=[np.number])
corr=numeric_cols.corr()
sns.heatmap(corr,cmap=("Blues"),annot=True)
plt.title("Matrice di Correlazione")
plt.show()

#Dalla matrice di correlazione vedo che posso andare a togliere quelle feature veramente poco collegate o che non sono utili per la divisione
df.drop(columns='booking_id',inplace=True)
df.drop(columns='transport',inplace=True)
df.drop(columns='season',inplace=True)
df.drop(columns='target',inplace=True)
#OCCHIO RIMUOVERE LE FEATURE POSSO CORRELATE LINEARI, fai a meno perchè potrebebro esserci pattern nascosti
print("\nInfo post pulizia:")
df.info()
#PRE-PROCESSING

#Quanti cluster? Tanti cluster Quante le classi che può assumereil target
num_clu=len(df[target].unique())
print(f"Avrò {num_clu} clusters")

#seleziono solo colonne numeriche--> kmeans lavora con quelle
numeric_cols=df.select_dtypes(include=[np.number]).columns.tolist()
#numeric_cols.remove('target')   #elimino colonna target che mi era utile per la matrice di correlazione (droppata prima)
print(f"\Features numeriche: {numeric_cols}")

X=df[numeric_cols]

#Scaling
scaler=StandardScaler()
X_scaled=scaler.fit_transform(X)

#MODELLO--> cluster con kmeans (k= numero cluster=classi assunte da "target")
kmeans=KMeans(n_clusters=num_clu,random_state=10)
labels=kmeans.fit_predict(X_scaled) #labels è un array dove ogni riga viene associata ad un cluster

#crosstab
crosstab=pd.crosstab(df[target],labels,colnames=["clusters"])
print("\nCrosstab:")
print(crosstab)

#mapping sul crosstab (cluster-classe target) [Per ogni cluster assegna come classe predetta il cancelled (target) più frequente al suo interno (majority vote)]
cluster_to_target=crosstab.idxmax(axis=0).to_dict()
print(f"Mapping cluster--target {cluster_to_target}")

#VALUTAZOIONE DEL MODELLO

target_pred=pd.Series(labels,index=df.index).map(cluster_to_target)

accuracy=(df[target]==target_pred).mean()
print(f"Accuracy globale : {accuracy*100:.2f}%")    #98% il modello ha classificato corretamente , valore ben alto a causa dello sbilanciamneto del datsset

#VISUALIZZAZIONE DEL MODELLO

pca=PCA(n_components=2) #2d
features_compressd=pca.fit_transform(X_scaled)

plt.scatter(
    features_compressd [:,0],
    features_compressd [:,1],
    c=labels,
    s=10
)
plt.colorbar(label="clusters")
plt.title(f"PCA 2D {num_clu}")
plt.show()

#SEPARAZIONE ABBASTANZA NETTA, IL KMEANS lavora abbastanza bene

print(f"Conteggio valori assunti dal target: {df[target].value_counts()}")  #è mooolto sbilanciato, il modello spara a caso e becca no ecco l accuracy ottima






