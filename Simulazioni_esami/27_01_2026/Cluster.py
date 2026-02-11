import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.preprocessing import LabelEncoder,StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score


file_path="exam_score.csv"
if not os.path.exists(file_path):
    raise FileNotFoundError(f"File csv non trovato al percorso: {file_path}")

#CARICAMENTO DATASET
df=pd.read_csv(file_path,low_memory=False,na_values=['','-','NA','ND','n/a'])

#Info generali sul dataset
print(f"\nDataset di grandezza: {df.shape}")
print(f"Prime righe: {df.head()}")
print("Info dataset")
df.info()

target='sleep_quality'  #definisco un target sebbene il cluster sia un algoritmo non supervisionato noi avremo tanti cluster quanti sono le classi che assume il target
num_clu=len(df[target].unique()) #il numero di cluster= numero di classi che target assume
print(f"\nNumero di cluster: {num_clu}")
#DATA CLEANING
df.dropna(axis=1,how="all",inplace=True)    #rimuovo colonne full Nan
df.dropna(axis=0,how="any",subset=target,inplace=True)  #rimuovo righe con valori Nan nel target che creano difficolatà ai cluster


print("\nValori che assumono le feature non numeriche:")
print("\nGender")
print(df['gender'].unique())
print("\ncourse")
print(df['course'].unique())
print("\ninternet_access")
print(df['internet_access'].unique())
print("\nsleep_quality (target)")
print(df['sleep_quality'].unique())
print("\nstudy_method")
print(df['study_method'].unique())
print("\nfacility_rating")
print(df['facility_rating'].unique())
print("\nexam_difficulty")
print(df['exam_difficulty'].unique())

#Clsuster lavora con riferimenti numerici (distanze numeriche), occorre convertire le fetures categoriche in numeri
#Uso il Label econder per semplifciare le analisi ma per modelli non ad Albero bisognerebbe:
    #Ordinal Encoding: variabili ordinali
    #One hot encoding: variabili nominative
#Non uso One hot encoding (scelta teoricamnet corretta) perchè va a creare una colonna per ogni classe col rischio diaumentare notevolmente la dimensione di dataset complicando l'elaborazione
#Label Encoder:

le_gender=LabelEncoder()
df['gender']=le_gender.fit_transform(df['gender'])

le_course=LabelEncoder()
df['course']=le_course.fit_transform(df['course'])

le_internet=LabelEncoder()
df['internet_access']=le_internet.fit_transform(df['internet_access'])

#le_target=LabelEncoder()
#df['sleep_quality']=le_target.fit_transform(df['sleep_quality']) #target

le_study=LabelEncoder()
df['study_method']=le_study.fit_transform(df['study_method'])

#le_facility=LabelEncoder()
#df['facility_rating']=le_facility.fit_transform(df['facility_rating'])
df['facility_rating']=df['facility_rating'].map({
    "low":0,
    "medium":1,
    "high":2
})

#le_difficulty=LabelEncoder()
#df['exam_difficulty']=le_difficulty.fit_transform(df['exam_difficulty'])

df['exam_difficulty']=df['exam_difficulty'].map({
    "easy":0,
    "moderate":1,
    "hard":2
})



print("\nValori che assumono le feature non numeriche (post encoding):")
print("\nGender")
print(df['gender'].unique())
print("\ncourse")
print(df['course'].unique())
print("\ninternet_access")
print(df['internet_access'].unique())
print("\nsleep_quality")
print(df['sleep_quality'].unique())
print("\nstudy_method")
print(df['study_method'].unique())
print("\nfacility_rating")
print(df['facility_rating'].unique())
print("\nexam_difficulty")
print(df['exam_difficulty'].unique())

"""
#Matrice di correlazione lineare
num_col=df.select_dtypes(include=[np.number])
corr=num_col.corr()
sns.heatmap(corr,annot=True,cmap="Blues")
plt.title("Matrice di correlazione")
plt.show()

#NOTO che potrei eliminare qualche features per il poco collegamento col target (migliorando probabilmete le prestazioni)
#MA così facendo mi baserei solo sulle connessioni lineari, e precluderei al modello cluster kmeans di scoprire pattern nascosti oltre la linearità (motivo per cui lascio anche features con correlazione bassa)
"""
df.info()
df.drop(columns="student_id",inplace=True)#info inutile per l'analisi
numeric_cols=df.select_dtypes(include=[np.number]).columns.tolist()     #lista di features numeriche
print(f"\nFeature numerirche: {numeric_cols}")
df.dropna(axis=0,how="any",subset=numeric_cols,inplace=True)    #pulisco da righe con valori Nan nelle features utili


#PRE-PROCESSING

X=df[numeric_cols]

#Cluster--> niente split

#Scaling,standardizzo i dati
scaler=StandardScaler()
X_scaled=scaler.fit_transform(X)

#Modello--> cluster con Kmeans--> K=numero di cluster=numero classi assunte da target
kmeans=KMeans(n_clusters=num_clu,random_state=12)
labels=kmeans.fit_predict(X_scaled)

#crosstab
crosstab=pd.crosstab(df[target],labels,colnames=["clusters"])
print("\nCrosstab:")
print(crosstab)

#mapping di crosstab
cluster_to_target=crosstab.idxmax(axis=0).to_dict()
print(f"Mapping cluster-->target: {cluster_to_target}")

#VALUTAZIONE DEL MODELLO

Y_pred=pd.Series(labels,index=df.index).map(cluster_to_target)  

# confrontando il cluster assegnato dal modello con il target reale, poi calcolo percentuale di corrispondenza, quanto i cluster trovati da KMeans coinciderebbero con le classi reali se fossero classi
accuracy=(df[target]==Y_pred).mean()    #calcolo accuracy golbale, quante ne azzecca
print(f"\nAccuracy globale: {accuracy *100:.2f}%")

silhouette=silhouette_score(X_scaled,Y_pred)
print(f"\ Silhouette score: {silhouette *100:.2f}%")    #valore basso:Hai imposto K cluster, ma i dati non hanno una struttura naturale ben separata.(scelta sbagliata)”

"""
Silhouette	Significato
> 0.5	Cluster ben separati
0.2–0.5	Struttura debole ma presente
< 0.2	Cluster poco significativi
≈ 0	Nessuna vera separazione
"""


#VISUALIZZAZIONE del Modello


pca=PCA(n_components=2)
features_compressed=pca.fit_transform(X_scaled)

plt.scatter(
    features_compressed[:,0],
    features_compressed[:,1],
    c=labels,
    s=8
)
plt.colorbar(label="clusters")
plt.title(f"Grafico PCA 2D (k={num_clu})")
plt.show()

print(f"Conteggio Valori target: {df[target].value_counts()}") #bilanciato
