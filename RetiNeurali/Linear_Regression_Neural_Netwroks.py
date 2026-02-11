
# Variabile target (Lineare):  BestSquatKg → prevedere il miglior squat

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import classification_report
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import balanced_accuracy_score,accuracy_score,root_mean_squared_error
import tensorflow as tf
import keras
from keras import layers
from keras.callbacks import EarlyStopping
from keras import optimizers
from keras import callbacks
from keras.utils import to_categorical



file_path="openpowerlifting.csv"
if not os.path.exists(file_path):
    raise FileNotFoundError(f"File csv non trovato al percorso  {file_path}")

#CARICHIAMO IL DATASET
# -> nrows indica il numero di righe massimo da caricare 
df= pd.read_csv(file_path, low_memory=False, na_values=['-', 'NA', 'n/a', 'ND',''], nrows=1000000)

print(f"\nGrandezza del dataset:{df.shape}")
print(f"\nPrime 5 righe: {df.head()}")
df.info()
#vedo che non ho righe nulle, ma alcune con veramente pochi valori non nulli potrei toglierle (Squat4Kg,Bench4Kg,Deadlift4Kg) per eliminare il "rumore"
#ho delle stringhe da convertire, e posso togliere MeetID  e Name inutili

target = "BestSquatKg"
class_names = df[target].unique()   #quali valori può assumere
num_classes = len(class_names)  #numero di valori che può assumere, ci servirà poi per il one hot encoding
print(f"\nValori che può assumere il target: {class_names}")
print("\nConteggio valori che assume il target:")
print(df[target].value_counts().sort_index()) #è bilanciato

#DATA CLEANING
df.dropna(axis=0, how='any', subset=target, inplace=True) # Eliminiamo tutte le righe che hanno un valore None nel campo target poichè non sarebbero classificabili
df.dropna(axis=1, how='all', inplace=True) # Eliminiamo tutte le colonne che hanno tutti valori None (SOLO VALORI NULLI)



print("\nValori assunti dalle features stringhe:")
print("Name")
print(df['Name'].unique())
print("Sex")
print(df['Sex'].unique())
print("Equipment")
print(df['Equipment'].unique())
print("Division")
print(df['Division'].unique())
print("WeightClassKg")
print(df['WeightClassKg'].unique())
print("Place")
print(df['Place'].unique())

# Label Encoder

# In generale, per modelli non ad albero [come la regressione lineare], sarebbe preferibile distinguere tra:
# - variabili ordinali → ordinal encoding
# - variabili nominali → one-hot encoding
#
# Il one-hot encoding può aumentare significativamente la dimensionalità del dataset,
# motivo per cui in questo contesto si preferisce l’uso del LabelEncoder in questo caso

#Faccio il mapping delle feature che devono avere un ordine di importanza perchè ho notato che l'encoding, dato che ragiona in ordine alfabetico, non da il giusto peso alle feature
#Così facendo miglioro di molto le predizioni del modello(map-->ordinal encoding) [qui non lo faccio]

#ES (gender lascia così con lencoder)
"""
df['satisfaction'] = df['satisfaction'].map({
    'Low': 0,
    'Medium': 1,
    'High': 2
})
"""


lencoder=LabelEncoder()
#NO encoder su target che è gia nuemro
df['Name']=lencoder.fit_transform(df['Name'])
df['Sex']=lencoder.fit_transform(df['Sex'])
df['Division']=lencoder.fit_transform(df['Division'])
df['WeightClassKg']=lencoder.fit_transform(df['WeightClassKg'])
df['Place']=lencoder.fit_transform(df['Place'])
df['Equipment']=lencoder.fit_transform(df['Equipment'])

print("\nValori assunti dalle features stringhe (post-encoding):")
print("Name")
print(df['Name'].unique())
print("Sex")
print(df['Sex'].unique())
print("Equipment")
print(df['Equipment'].unique())
print("Division")
print(df['Division'].unique())
print("WeightClassKg")
print(df['WeightClassKg'].unique())
print("Place")
print(df['Place'].unique())


#Matrice di correlazione (che funziona solo con tipi numerici) per vedere le correlazioni se posso scartare qualche featuresì inutile
num_cols = df.select_dtypes(include=[np.number]).columns.tolist()   #seleziono le colonne numeriche (tutte)
cm = df[num_cols].corr()
plt.figure(figsize=(12,12))
sns.heatmap(cm, cmap="coolwarm", annot=True, fmt=".2f", linewidths=1.5)
plt.title("Correlation Matrix")
plt.show()
#NOTO: essendo una regressione lineare, le correlazioni lineari le posso togliere poichè causano rumore
#per la classificazione si potrebbe lasciarle poichè possono essere presenti altre correlazioni nascoste che vanno oltre la linearità

num_cols.remove('MeetID')#rimuovo che è una features sensibile, ed inutile ai fini dell'addestramento
num_cols.remove('Squat4Kg') #rumore, pochi non null
num_cols.remove('Bench4Kg') #rumore, pochi non null
num_cols.remove('Deadlift4Kg') #rumore, pochi non null
num_cols.remove('Name') #non c'entra nulla con la classificazione
#rimuovo colonne ininfluenti per la bassissima correlazione, nella Regressione Lineare è inutile (|r|<0.20)
num_cols.remove('Age') #rimuovo perchè davvero troppo poco correlata 0.03-->rumore
num_cols.remove('Division') #rimuovo perchè davvero troppo poco correlata 0.04-->rumore
num_cols.remove('Place') #rimuovo perchè davvero troppo poco correlata 0.03-->rumore
num_cols.remove('Equipment') #rimuovo perchè davvero troppo poco correlata 0.12-->rumore
num_cols.remove('TotalKg') #rimuovo perchè con la formula di queste 3 danno al modello il BestSquatkg (sono cose che si sanno post gara), ma il modello deve predire prima della gara [hanno una correlazione alta e va bene, non rimuovo per quello ma per la troppa influenza sul target]
num_cols.remove('BestBenchKg') # rimuovo perchè con la formula di queste 3 danno al modello il BestSquatkg(sono cose che si sanno post gara) ma il modello deve predire prima della gara
num_cols.remove('BestDeadliftKg') #rimuovo perchè con la formula di queste 3 danno al modello il BestSquatkg(sono cose che si sanno post gara) ma il modello deve predire prima della gara


if target in num_cols:  #dopo aver visto la correlazione, se c'è target lo rimuovo perchè num_cols sarà per la matrice X per l addestramento
    num_cols.remove(target)

df.dropna(axis=0, how='any', subset=num_cols, inplace=True) #elimino righe non nulle nelle colonne numeriche
print("DataSet post modifiche a righe e colonne")   #qui avrò ancora le fetures che ho eliminato
df.info() #post-modifiche

print(f"Colonne numeriche:{num_cols}")  #noto che mancano le fetures eliminate che per un motivo e per l'altro non centrano con la classificazione
#userò num_cols come matrice X d'addestramento

#PRE-PROCESSING

# Scegliamo come X (le feature che il modello userà per fare la predizione) tutte le feature numeriche 
X= df[num_cols]
# Scegliamo come Y (la feature target che il modello dovrà predire)
Y= df[target]

# SPLIT 1:
# Splittiamo:
# -> le righe presenti in X in: righe per l'addestramento (80%: X_train) + righe per il test (20%: X_test)
# -> le righe presenti in Y in: righe per l'addestramento (80%: Y_train) + righe per il test (20%: Y_test)
X_train, X_test, Y_train, Y_test= train_test_split(
    X, Y,
    test_size=0.2,
    random_state=42
)

# SPLIT 2
# Splittiamo:
# -> le righe presenti in X_train in: righe per l'addestramento (90%: X_train) + righe per la validazione (10%: X_val)
# -> le righe presenti in Y_train in: righe per l'addestramento (90%: Y_train) + righe per la validazione (10%: Y_val)
X_train, X_val, Y_train, Y_val= train_test_split(   #Il secondo split è fatto tra i dati di training
    X_train, Y_train,
    test_size=0.1,
    random_state=42
)

# Mostriamo il numero di righe e colonne presenti in X_train, X_val ed X_test
print("\nNumero di righe e colonne di X_train:", X_train.shape)
print("Numero di righe e colonne di X_validation: ", X_val.shape)   #validazione per vedere se modello sta imparando bene o memorizzando i dati dell esempio(overfitting)
print("Numero di righe e colonne di X_test: ", X_test.shape)

# Vediamo che classi di 'status' sono presenti dentro ad Y_train e ad Y_test
print(f"Classi di target presenti in Y_train: {Y_train.unique()}, {len(Y_train.unique())} classi")
print(f"Classi di target presenti in Y_test:{Y_test.unique()}, {len(Y_test.unique())} classi ")   #Diverso numero di classi tra test e train, in regressione non è un problema

#Non serve fare label encoder e one hot encodign delle Y, son gia numeri continui [PER LA REGRESSIONE LINEARE]

# Facciamo lo scaling dei valori presenti in X_train, X_val ed X_test
scaler= StandardScaler()
X_train_scaled= scaler.fit_transform(X_train) # Fittiamo lo scaler sui valori di X_train e scaliamo i valori
X_val_scaled= scaler.transform(X_val) # Facciamo la stessa cosa sui valori di X_val ma senza fittare di nuovo la scaler 
X_test_scaled= scaler.transform(X_test) # Facciamolo anche sui valori di X_test


#MODELLO

# Creiamo il modello
model= keras.Sequential([
    # Qua mettiamo il primo layer (ingresso) della rete neurale
    layers.Input(shape=(X_train_scaled.shape[1],)),
    # Qua mettiamo i layer intermedi con quanti neuroni vogliamo. Posso usare anche più layer input con un numero diverso di neuroni
    layers.Dense(16, activation="relu"),
    layers.Dense(8, activation="relu"),
    # Qua mettiamo il layer finale con tanti neuroni quante sono le possibili classi di 'status' predicibili ("num_classes"). Usiamo 'softmax' come funzione di
    # attivazione perché stiamo facendo una classificazione multiclasse, altrimenti se fosse una classificazione binaria [M o F] dovremmo usare 'sigmoid' (e 1 neurone)
    layers.Dense(1, activation="linear")   #linear per la regressione e 1 neurone, softmax per classifier neuroni quante le colonne
])

# Configuriamo il modello
model.compile(
    optimizer=optimizers.Adam(learning_rate=1e-3),
    loss="mse",   
    metrics=["mse"]    #per la regressione guardo mse
)

# Addestriamo il modello sui dati di train, ovvero:
# -> X_train_scaled
# -> Y_val 
history= model.fit(
    X_train_scaled, Y_train,
    validation_data=(X_val_scaled, Y_val),
    epochs=20, # "epochs" indica il numero di volte che vengono aggiornati i pesi della rete neurale sugli stessi dati di test, quante volte l'esempio viene visto dal modello durante l addestramento
    batch_size=128, 
    verbose=1 
)

#VALUTAZIONE DEL MODELLO

Y_pred = model.predict(X_test_scaled).flatten() #prendi direttamente i valori predetti
#.argmax() serve solo quando la rete è una classificazione multi-classe con one-hot encoding (softmax).

# Vediamo quanto è diventato bravo il modello usando la funzione "model.evaluate()" per calcolare la loss e l'accuracy (% di previsioni corrette) sui dati di test
loss, mse= model.evaluate(X_test_scaled, Y_test, verbose=0) 
rmse = root_mean_squared_error(Y_test, Y_pred)
print(f"Loss: {loss}")
print(f"MSE: {mse}")
print(f"Range:{(Y_test.max()-Y_test.min())}")
print(f"RMSE on range (%): {(rmse/(Y_test.max()-Y_test.min()))*100:.2f}")


#GRAFICO REGRESSIONE LINEARE
plt.figure(figsize=(12, 12))
plt.scatter(Y_test, Y_pred, color="blue", alpha=0.6, label="Predict")
plt.plot([Y_test.min(), Y_test.max()], [Y_test.min(), Y_test.max()], color="red", label="Linear Regression")
plt.title("Valutazione predict")
plt.xlabel("Valori predetti")
plt.ylabel("Valori reali")
plt.legend()
plt.show()

#Mse di validazione
plt.figure(figsize=(12, 12))
plt.plot(history.history.get("mse"), color="red", label="MSE")
plt.plot(history.history.get("val_mse"), color="blue", label="VAL MSE")
plt.title("Model MSE")
plt.xlabel("Epoche")
plt.ylabel("MSE")
plt.legend()
plt.show()

#Loss di validazione
plt.figure(figsize=(12, 12))
plt.plot(history.history.get("loss"), color="red", label="MSE")
plt.plot(history.history.get("val_loss"), color="blue", label="VAL loss")
plt.title("Model loss")
plt.xlabel("Epoche")
plt.ylabel("Loss")
plt.legend()
plt.show()


#è NORMALE CHE OGNI VOLTA CHE ESEGUO OTTENGO RISULATI UN PO DIVERSI NELLE RETI NEURALI, ogni volta i neuroni partono nda punti diversi... a casos

#Ho una forete regressione lineare (il modello cattura la struttura principale del problema):
    #chiaramente allungata lungo la diagonale
    #con trend lineare netto

#La dispersione aumenta con valori alti (ho pochi punti e dispersi in alto)--> simbolo che fa fatica con valori alti forse dovuti alla poca presenza nel dataset generando un errore ben maggiore rispetto la media