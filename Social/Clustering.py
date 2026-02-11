import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder,StandardScaler
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA


file_path="Social.csv"
if not os.path.exists(file_path):
    raise FileNotFoundError("File csv non trovato al path:",file_path)

df=pd.read_csv(file_path,low_memory=False,na_values=['-','','NA','ND','n/a'],nrows=1000000)

print(f"\nGrandezza del dataset:{df.shape}")
print(f"\nPrime righe dataset{df.head()}")
print(f"\nInfo dataset:")
df.info()


#DATA-CLEANING
target='Most_Used_Platform'
df.dropna(axis=1,how="all",inplace=True) #elimino colonne nulle (qui non ce ne sono)
df.dropna(axis=0,how="any",subset=target,inplace=True)  #elimino righe non nulle nel target (mi servrà poi per fare unc onfronto tra predizione cluster e target reali)

df.drop(columns="Student_ID",inplace=True)  #tolgo l'id visto che è ininfluente per l'elaborazione

print("\nValori assunti da features Stringhe:")
print("\nGender")
print(df['Gender'].unique())
print("\Academic_Level")
print(df['Academic_Level'].unique())
print("\Country")
print(df['Country'].unique())
print("\Affects_Academic_Performance")
print(df['Affects_Academic_Performance'].unique())
print("\Relationship_Status")
print(df['Relationship_Status'].unique())

#Sklearn.Kemans non lavora con stringhe bensì con features numeriche
#Label_Encoder  (viene eseguito il label encoder per analisi semplificativa visto che occrerrebbe fare una distinzione tra valori stringhe ordinali [usare Lbale Encoder] e valori catgorici [usare One hot Encoding])
#Per distinguere se categorico o nominale bisognerebbe avere una profonda conoscienza del dataset, e il fare one hot encoding andrebbe ad aumentare dratiscamnte il numero di colonne del dataset, portando a più mali che beneficì
lencoder=LabelEncoder()
#df['Academic_Level']=lencoder.fit_transform(df['Academic_Level']) #c'è un ordine posso provare con ordinal encoding e vedere se migliora (Label converte in numero in base all'ordine alfabetico)
df['Gender']=lencoder.fit_transform(df['Gender'])
df['Country']=lencoder.fit_transform(df['Country'])
df['Affects_Academic_Performance']=lencoder.fit_transform(df['Affects_Academic_Performance'])
df['Relationship_Status']=lencoder.fit_transform(df['Relationship_Status'])
df['target']=lencoder.fit_transform(df[target])
#il target Most_Used_Platform lo lascio stirnga per il confronto con i cluster
#aggiungo colonna target in int per la correlazione (poi droppo)

df["Academic_Level"]=df["Academic_Level"].map({
    "Undergraduate":"0",
    "Graduate":"1",
    "High School":"2"

})

print("\nValori assunti da features Stringhe (post encoding):")
print("\Gender")
print(df['Gender'].unique())
print("\nAcademic_Level")
print(df['Academic_Level'].unique())
print("\nCountry")
print(df['Country'].unique())
print("\nAffects_Academic_Performance")
print(df['Affects_Academic_Performance'].unique())
print("\nRelationship_Status")
print(df['Relationship_Status'].unique())

#MATRICE DI CORRELAZIONE
numeric_cols_data=df.select_dtypes(include=[np.number])   #seleziono dati di sole colonne numeriche
cm=numeric_cols_data.corr()
sns.heatmap(cm,cmap="coolwarm",annot=True)
plt.title("Matrice di correlazione")
plt.show()


#Dalla matrice di correlazione noto la poco correlazione (causerebbe rumore) con il target di:Relationship_Status,Sleep_Hours_Per_Night,Country,Academic_Level,Age
#Quelle con le correlazioni basse tipo 0.21 o 0.13 le tengo perchè son le piu alte
df.drop(columns="target",inplace=True)  #colonne fatta per la correlazione target in numero
df.drop(columns="Age",inplace=True)
df.drop(columns="Relationship_Status",inplace=True)
df.drop(columns="Sleep_Hours_Per_Night",inplace=True)
df.drop(columns="Country",inplace=True)
df.drop(columns="Academic_Level",inplace=True)
#df.drop(columns="Avg_Daily_Usage_Hours",inplace=True)  #Non tolgo la media poichè ha una buona correlazione col target. Non possiedo la feature Daily_Usage_Hours, Altrimenti la dovreii togliere perchè rappresenterebbe data Leakege


#Slezione feature numeriche

numeric_cols=df.select_dtypes(include=[np.number]).columns.tolist()
df.dropna(axis=0,how="all",subset=numeric_cols,inplace=True)
print("Colonne numeriche:",numeric_cols)

print("\nDataSet post modifiche:")
df.info()

#PRE-PROCESSING

#numero di cluster= numero di classi che assume il target
num_clu=len(df[target].unique())

print(f"Il target può assumere {df[target].unique()} per un totale di {num_clu} classi")

#cluster--> niente split

#scaling
X=df[numeric_cols]
scaler=StandardScaler()
x_scaled=scaler.fit_transform(X)

#MODELLO--> clustering non supervisonato kmeans
kmeans=KMeans(n_clusters=num_clu,random_state=10)
labels=kmeans.fit_predict(x_scaled) #labels è array dove ogni riga associata al cluster

#crosstab
crosstab=pd.crosstab(df[target],labels,colnames=["clusters"])
print("\nCrosstab:")
print(crosstab)

#Mapping sul crosstab
cluster_to_target=crosstab.idxmax(axis=0).to_dict() #per ognni cluster prendo la classe più dominanate
print(f"Mapping cluster to target:{cluster_to_target}")

#VALUTAZIONE

target_pred=pd.Series(labels, index=df.index).map(cluster_to_target)    # Converto i cluster numerici in etichette di classe

accuracy=(df[target]==target_pred).mean()
print("\nAccuracy globale",accuracy)


#VISUALIZZAZIONE

pca=PCA(n_components=2)
feature_compresse=pca.fit_transform(x_scaled)

plt.scatter(
    feature_compresse[:,0],
    feature_compresse[:,1],
    c=labels,
    s=10
)
plt.title("PCA 2D (12 cluster)")
plt.colorbar(label="clsuters")
plt.show()




