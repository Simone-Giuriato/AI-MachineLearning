import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA


#Caricamento Dataset
file_path="5b_winequality-white_cat.csv"
if not os.path.exists(file_path):
    raise FileNotFoundError("File csv non trovato al percorso",file_path)

df=pd.read_csv(file_path,low_memory=False,na_values=['','-','NA','ND','n/a'],nrows=1000000)

print(f"\nGrandezza del dataset: {df.shape}")
print(f"\nPrime righe: {df.head()}")
print("\nInfo sul dataset importato:") #noto tutti numeri, tranne quality_cat che è un po il target, non ho colonne vuote
df.info()

#DATA-CLEANING

target="quality_cat"    #lo tengo dentro al dataset per un confronto successivo per il clustering
df.dropna(axis=1,how="all",inplace=True)
df.dropna(axis=0,how="any",inplace=True)

print("\nInfo sul dataset importato, post pulizia:") #noto tutti numeri, tranne quality_cat che è un po il target, non ho colonne vuote
df.info()
num_col_names=df.select_dtypes(include=[np.number]).columns.tolist()    #seleziono solo colonne numeriche
print(f"Colonne numeriche: {num_col_names}")

#PRE-PROCESSING
#niente split nei clusterin

#Scaling dati numerici
scaler=StandardScaler()
df[num_col_names]=scaler.fit_transform(df[num_col_names])

#quali e quante classi assume il target? [Avrò tanti clusters quante sono le classi che assume]

print(f"Valori assunti dal target: {df[target].value_counts()}") #sbilanciato, soprattuto nelle categorie degli estremi--> potrei unire in Alto,medio,basso?
num_clusters=len(df[target].unique())
print(f"\nIl target può assumere {df[target].unique()} quindi:  {num_clusters} classi") #avrò 7 clusters



#MODELLO---> clustering Kmeans (k= numero clusters)

kmeans=KMeans(n_clusters=num_clusters,random_state=10)

labels=kmeans.fit_predict(df[num_col_names]) #fit calcola i centroidi, predict assegna le righe ai cluster
#labels è un array grande quanto il numero di righe, e per ogni riga ha associato il cluster di appartenenza, ovviamneto kmeans lavora con feature numeriche

#Crosstab
crosstab=pd.crosstab(df[target],labels,colnames=["clusters"])
print("\nCrosstab:")
print(crosstab)

#Mapping
cluster_to_target=crosstab.idxmax(axis=0).to_dict()#quale classe è dominanate nel cluster
print("\n Mapping corrispondenza cluster--target:",cluster_to_target) #separa bene E e D, ma le altre fa molta fatica

#VALUTAZIONE

#Converto gli identificativi numerici cluster nel nome della classe del target  in una serie pandas
pred_target=pd.Series(labels,index=df.index).map(cluster_to_target) #map associa ogni numero del cluster alla classe dominanate, e sostituisce il numero cluster con la classe

#confronta la colonna reale e calcola l accuracy
accuracy= (pred_target==df[target]).mean()  #(pred_status == df['status']) crea una serie di booleani True/False, .mean() trasforma True→1 e False→0, quindi ottieni direttamente la percentuale (deicmale) di correttezza dividendo il numero totale dir righe
print(f"\nAccuracy globale: {accuracy*100:.2f}%")   #fa un po schifo accuracy del 47%

#VISUALIZZAZIONE
pca=PCA(n_components=2) #per graficare in 2D e non in tante dimensioni quante le features
compressed_features=pca.fit_transform(df[num_col_names])

plt.scatter(
    compressed_features[:,0],
    compressed_features[:,1],
    c=labels,#fodnamentale per dare ongi cluster un colore
    s=8
)
plt.colorbar(label="cluster")
plt.title("PCA 2D (7Clusters)")
plt.show()

#SI EVINCE COME LA CLASSE D SIA MOLTO NUMEROSA, ED INFATTI DAL MAPPING VENGA SUDDIVISA IN PIU CLUSTER

#PROVO UN CLUSTER con dataset Meno sbilanciato, andando a ragruppare le categorie in ALTO,MEDIO,BASSO

print("PROVO A RAGGRUPPARE LE CATEGORIE")
#è SBILANCIATO?
print(f"Valori assunti dal target: {df[target].value_counts()}") #sbilanciato, soprattuto nelle categorie degli estremi--> potrei unire in Alto,medio,basso?

#provo a ragruppare:
df[target]=df[target].replace({
    "A":"Alta_qualità",
    "B":"Alta_qualità",
    "C":"Alta_qualità",
    "D":"Media_qualità",
    "E":"Bassa_qualità",
    "F":"Bassa_qualità",
    "G":"Bassa_qualità"
})

print(f" Nuovi Valori assunti dal target: {df[target].value_counts()}") #già piu bilanciato
numero_clu=len(df[target].unique())  #dovrei avere 3 cluster perchè 3 classi
print("\nNumero di cluster/classi assunte dal target:",numero_clu)

# ================== MODELLO --> Clustering KMeans ==================
kmeans = KMeans(n_clusters=numero_clu, random_state=10)  
# Creo un modello KMeans con "numero_clu" cluster (qui corrispondenti alle classi raggruppate Alto, Medio, Basso)
# random_state serve per avere risultati riproducibili

labels = kmeans.fit_predict(df[num_col_names])  
# fit_predict fa due cose in una:
# 1. fit: calcola i centroidi dei cluster usando le feature numeriche
# 2. predict: assegna a ciascuna riga del dataset il cluster di appartenenza
# labels è un array della stessa lunghezza del dataset, ogni elemento indica il cluster della riga

# ================== Crosstab ==================
crosstab = pd.crosstab(df[target], labels, colnames=["cluster"])  
# Creo una tabella che mette:
# - sulle righe: le classi reali del target
# - sulle colonne: i cluster trovati da KMeans
# Ogni cella contiene quante righe di quella classe reale sono state assegnate a quel cluster

print("\nCrosstab")
print(crosstab)

# ================== Mapping cluster --> classe ==================
cluster_to_target = crosstab.idxmax(axis=0).to_dict()  
# idxmax(axis=0) trova per ogni colonna del crosstab (cluster) la riga con valore massimo, cioè la classe dominante in quel cluster
# to_dict converte il risultato in un dizionario {cluster: classe dominante}

print("Mapping cluster-->classi:", cluster_to_target)

# ================== Valutazione ==================
pred_target = pd.Series(labels, index=df.index).map(cluster_to_target)  
# Creo una serie Pandas dove ogni riga del dataset prende il nome della classe dominante del cluster a cui appartiene
# In pratica trasformo l'identificativo numerico del cluster nella classe corrispondente

accuracy = (df[target] == pred_target).mean()  
# Calcolo l'accuracy globale: confronto la classe reale con quella "predetta" dal clustering
# True -> 1, False -> 0, la media fornisce la percentuale di righe assegnate correttamente

print(f"\nAccuracy globale post raggruppamento :{accuracy*100:.2f}%")

# ================== Visualizzazione ==================
pca = PCA(n_components=2)  
# Riduzione dimensionale a 2D per poter visualizzare i dati

compressed_features = pca.fit_transform(df[num_col_names])  
# Applico la PCA sulle features numeriche, ottengo due colonne per ogni riga

plt.scatter(
    compressed_features[:,0],  # asse x PCA
    compressed_features[:,1],  # asse y PCA
    c=labels,                  # il colore di ciascun punto dipende dal cluster di appartenenza
    s=10                        # dimensione dei punti
)
plt.colorbar(label="clusters")  # aggiunge una legenda dei colori associati ai cluster
plt.title("PCA-2D (3 cluster)")
plt.show()
# Mostra il grafico dei dati ridotti in 2D con colori che rappresentano i cluster

"""
nterpretazione ACCURACY  45%

Significa che solo il 45% dei campioni si trova in un cluster la cui classe dominante corrisponde alla loro classe reale.

Gli altri 55% sono stati messi in cluster dove la classe dominante è diversa dalla loro.

⚠️ Questo non indica che KMeans abbia fallito, ma che i cluster non coincidono perfettamente con le classi che hai definito.
"""