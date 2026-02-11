import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#Lavoro su etichetta status
# =================== Caricamento Dataset ===================
file_path="air_quality.csv"
if not os.path.exists(file_path):
    raise FileNotFoundError("File csv non trovato al percorso {file_path}")

df=pd.read_csv(file_path,low_memory=False, na_values=['','-','NA','ND','n/a'],nrows=1000000)

print(f"Grandezza del dataset:{df.shape}")
df.info()
print("\nPrime 5 righe:")
print(df.head(5))


# ================== Data cleaning =====================
df.dropna(axis=1,how="all",inplace=True) #elimino colonna con tutti NaN (vedi colonna unit)
df.dropna(axis=0, how="any", subset="status",inplace=True) #elimino righe con un NaN in status
print(f"Grandezza del dataset, dopo pulizia:{df.shape}")
df.info()
print("\nPrime 5 righe, dopo la pulizia:")
print(df.head(5))



#PRE PROCESSING
# ================ Feature Transformation =================
df["date"]=pd.to_datetime(df["date"],errors="coerce")   #metto la data in formata datetime  #coerce trasforma data non interpretabile in NAT non lanciando errori
df['year']=df['date'].dt.year   #aggiungo nuova feature con l anno estranedolo dalla data

print("\nValori che assume status:")
print(df["status"].unique())
print("\nPrime 5 righe, dopo trasfromazione:")
df.info()
print(df.head(5))

"""
 Il dataset contiene variabili categoriali/stringhe nelle X (es. sitename, country, pollutant),
e l’implementazione degli alberi decisionali di scikit-learn(sklearn) non supporta direttamente feature categoriali,
si procede con una codifica numerica tramite Label Encoding.
Nota:
- Non viene applicata normalizzazione (mettere sulla stessa scala ) delle feature, poiché gli alberi decisionali non ne traggono beneficio
  (sono in larga parte invarianti rispetto a trasformazioni di scala).
"""
# Label Encoder
# Poiché il modello utilizzato è un albero decisionale, le variabili categoriali
# vengono codificate tramite LabelEncoder per convertire i valori stringa in interi.
# Nei modelli ad albero il LabelEncoder è adatto anche per variabili nominali,
# in quanto l’algoritmo non assume relazioni di ordine o distanza tra i valori.
#
# In generale, per modelli non ad albero, sarebbe preferibile distinguere tra:
# - variabili ordinali → ordinal encoding
# - variabili nominali → one-hot encoding
#
# Il one-hot encoding può aumentare significativamente la dimensionalità del dataset,
# motivo per cui in questo contesto si preferisce l’uso del LabelEncoder.

from sklearn.preprocessing import LabelEncoder

# Stampa delle etichette originali presenti nelle colonne categoriali
print("Etichette originali per sitename, county, pollutant: ")
print("\nSitename ")
print(df["sitename"].unique())
print("\ncounty ")
print(df["county"].unique())
print("\npollutant ")
print(df["pollutant"].unique())


#Label Encoder
lencoder=LabelEncoder()
df["sitename"]=lencoder.fit_transform(df["sitename"])
df["county"]=lencoder.fit_transform(df["county"])
df["pollutant"]=lencoder.fit_transform(df["pollutant"])

lencoder_target=LabelEncoder()  #faccio un ecnoder "diverso" che richiamerò per avere i nomi classi nella matrice id confusione
df["status"]=lencoder_target.fit_transform(df["status"])   #label encoder anche sul target (ma si potrebbe fare a meno, camba nulla)


# Verifica delle etichette numeriche ottenute dopo la trasformazione
print("Etichette dopo LabelEncoder per sitename, county, pollutant: ")
print("\nSitename ")
print(df["sitename"].unique())
print("\ncounty")
print(df["county"].unique())
print("\npollutant")
print(df["pollutant"].unique())
print(df["status"].unique())


# ======================= Visualizzazione dati: MATRICE DI CORRELAZIONE =========================

#(MATRICE DI CORRELAZIONE)

num_col=df.select_dtypes(include=["Float64","int64"]) #seleziono solo colonne che hanno numeri (escluso la data e status) [county,sitename sono in numeri grazie label encoder]
cm=num_col.corr() #calcola correlazione tra le colonne numeriche
plt.figure(figsize=(20,10))#visualizzare graficamente la matrice di correlazione
sns.heatmap(cm,annot=True,cmap="Blues",fmt=".2f",linewidths=0.5)
plt.title("Matrice di correlazione")
plt.show()



#Raggruppo classi correlate
features_names=[
    'pm2.5', 'pm10',
    'co', 'co_8hr',
    'no2', 'nox', 'no',
    'so2',
    'o3',
    'windspeed', 'winddirec',
    'year', 'longitude', 'latitude'
]# Abbiamo rimosso pm2.5_avg, pm10_avg --> sono troppo correlate con la colonna target 'status' (target leakage), guardo matrice correlazione e tengo solo una di quelle correlate
#così non sovraccarico il mio modello di feature

df.dropna(axis=0, how='any', subset=features_names, inplace=True)   #elimino righe con valori Nan nelle fetures
# ======================= ADDESTRAMENTO DEL MODELLO (split and scale) =========================

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# Creazione di matrice delle feature (X) e vettore target (y)
X = df[features_names] 
print(X)    
Y=df['status']
print(Y)

#Split 
# Suddivisione in training set e test set
X_train,X_test,Y_train,Y_test=train_test_split(
    X,Y,
    test_size=0.3,  #30% dati per test
    train_size=0.7, #70% per allenamento
    random_state=10 #mettere sempre lo stesso random state, per ottenere stesso albero
)

#Scale (Scaling si potrebbe fare a meno poichè alberi decisionali non traggono beneficio)
scaler=StandardScaler()
X_train_scaled=scaler.fit_transform(X_train) #na volta che lo applichi (con .fit_transform()), il suo lavoro è trasformare i tuoi dati affinché abbiano tutti una "forma standard".
X_test_scaled= scaler.transform(X_test) #Sul TRAINING Set -> Usa fit_transform(impara e applica), Sul TEST Set (o nuovi dati) -> Usa SOLO transform(applica)


# ======================= MODELLO =========================
from sklearn.tree import DecisionTreeClassifier,plot_tree

#Inizializzaione modello d'addestramento--> Albero Decisionale
model=DecisionTreeClassifier(random_state=10)
model.fit(X_train_scaled,Y_train)   #addestro albero---> fit=studia (x le domande, y le risposte corrette)


#Visualizzazione dell'albero 
plt.figure(figsize=(20,10)) #Dimensioni
plot_tree(model, feature_names=features_names, max_depth=2)
plt.title("Decision Tree")
plt.show()

# ======================= VALUTAZIONE DEL MODELLO =========================
from sklearn.metrics import accuracy_score,confusion_matrix,ConfusionMatrixDisplay

#Accuracy: percentuale di predizioni corrette.
#Matrice di confusione 

#Accuracy
Y_pred=model.predict(X_test_scaled)
print(f"\nAccuray {accuracy_score(Y_test,Y_pred)}")

#Matrice di confusione
cm=confusion_matrix(Y_test,Y_pred,normalize="true")  #calcola
disp=ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=lencoder_target.classes_) #serve per mostrare cosi come è, lencoder_target.classes_ vado a metetre il nome delle classi prima di convertire in numero richiamando lencoder usato solo per il target 
disp.plot(cmap="Blues") 
plt.title("Matrice di confusione")
plt.show()














