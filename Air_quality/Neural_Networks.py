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
from sklearn.metrics import balanced_accuracy_score
import tensorflow as tf
import keras
from keras import layers
from keras.callbacks import EarlyStopping
from keras import optimizers
from keras import callbacks
from keras.utils import to_categorical

file_path="air_quality.csv"
if not os.path.exists(file_path):
    raise FileNotFoundError(f"File csv non trovato al percorso  {file_path}")

#CARICHIAMO IL DATASET
# -> nrows indica il numero di righe massimo da caricare (qua ne carichiamo 300.000 per cercare di avere nel dataframe almeno 5 possibili classi di 'status')
# -> na_values contiene le stringhe del file che indicano valori assenti nelle celle (Pandas li sostituisce con un valore None)
df= pd.read_csv(file_path, low_memory=False, na_values=['-', 'NA', 'n/a', 'ND',''], nrows=300000)

print(f"\nGrandezza del dataset:{df.shape}")
print(f"\nPrime 5 righe: {df.head()}")
df.info()
#vedo che: date,sitename,county,pollutant,(status) sono stringhe--> reti neurali ragionano coi numeri

# LA COLONNA (o feature) 'status' È IL TARGET DA PREDIRE (ovvero la nostra Y)

#DATA_CLEANING
df.dropna(axis=0, how='any', subset='status', inplace=True) # Eliminiamo tutte le righe che hanno un valore None nel campo 'status' (perché non sono classificabili)
df.dropna(axis=1, how='all', inplace=True) # Eliminiamo tutte le colonne che hanno tutti valori None (SOLO VALORI NULLI)

# Sostituiamo i valori +inf e -inf con NaN
df.replace([np.inf, -np.inf], np.nan, inplace=True) #qui non ne avevamo, si può evitare

# Selezioniamo i nomi delle colonne con valori puramente numerici (ci serviranno per dopo e soprattutto li usiamo anche qua nella sezione di data cleaning)
numeric_cols_names= df.select_dtypes(include=np.number).columns.tolist()
# Useremo "numeric_cols_names" come X, ma prima rimuoviamo 'aqi' perché è troppo correlata a 'status' (renderebbe la vita troppo facile al modello)[lo vedo tramite matrice di correlazione]
numeric_cols_names.remove('aqi')
numeric_cols_names.remove('siteid') # Rimuoviamo anche 'siteid' perché non è una feature necessaria alla predizione della classe di 'status'

# Dato che poi useremo le colonne numeriche come feature X (per ottenere la predizione), potremmo dover togliere la colonna target da "numeric_cols_names"
# Nel nostro caso non serve perché 'status' è categorica (stringa), non numerica
"""
if 'status' in numeric_cols_names:
    numeric_cols_names.remove('status')
"""

# Eliminiamo tutte le righe che hanno almeno un campo numerico (almeno una colonna tra quelle di "numeric_cols_names") con valore NaN
# QUESTO È IMPORTANTISSIMO PER IL CORRETTO ADDESTRAMENTO DELLA RETE NEURALE: NON FUNZIONA CON DEI NaN TRA I DATI DI TRAIN!
df.dropna(axis=0, how='any', subset=numeric_cols_names, inplace=True)# prima l ho fatto sul target(stringa) ora anche sulle features X

# Vediamo che classi di 'status' (target) abbiamo e quante sono
class_names= df['status'].unique()
num_classes= len(df['status'].unique()) #FONDAMENTALE OER ONE HOT ENCODING
print("\nPossibili classi di 'status':", class_names)
print("\nNumero totale di classi:", num_classes)

print("DataSet post modifiche a righe e colonne")
df.info() #post-modifiche

"""
#(MATRICE DI CORRELAZIONE), prima fare label encoder dei campi stringa per portarli in numeri... poi vedere correlazione e capire. Qua il prof li elimina le stringhe a caso

num_col=df.select_dtypes(include=["Float64","int64"]) #seleziono solo colonne che hanno numeri (escluso la data e status) [county,sitename sono in numeri grazie label encoder]
cm=num_col.corr() #calcola correlazione tra le colonne numeriche
plt.figure(figsize=(20,10))#visualizzare graficamente la matrice di correlazione
sns.heatmap(cm,annot=True,cmap="Blues",fmt=".2f",linewidths=0.5)
plt.title("Matrice di correlazione")
plt.show()

print(f"\nValori Numerici {num_col.columns}")
"""

#PRE-PROCESSING
# FACOLTATIVO (secondo il procedimento che useremo qua per estrarre X ed Y non è obbligatorio):
# Eliminiamo dal dataset le colonne non numeriche e quelle non utili per la predizione di 'status', quindi non vado a convertire le stringhe in numero, le tolgo [quelle molto correlate tra loro tengo quella piu corrwelata col target]
# Rimuoviamo anche 'aqi' perché è troppo correlata a 'status' (renderebbe la vita troppo facile al modello)[prima avevo riomosso cose da numeric_cols, ora tolgo proprio dal dataste]
columns_to_drop= ["sitename", "county", "aqi", "unit", "siteid", "pollutant", "date"]
df.drop(columns=columns_to_drop, errors='ignore', inplace=True)
print(df.info()) # Vediamo ora cosa rimane nel dataset

#[Negli alberi decisionali le stringhe le convertivamo in numeri con label encoder], faremo così... qua non ho capito perchè le toglie
# Scegliamo come X (le feature che il modello userà per fare la predizione) tutte le feature numeriche (che alla fine sarn le stesse di quelle nel df meno status, poichè le str le ho tolte)
X= df[numeric_cols_names]
# Scegliamo come Y (la feature target che il modello dovrà predire) la feature 'status'
Y= df['status']

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

# FACOLTATIVO:
# Vediamo che classi di 'status' sono presenti dentro ad Y_train e ad Y_test
print("Classi di 'status' presenti in Y_train: ", Y_train.unique())
print("Classi di 'status' presenti in Y_test: ", Y_test.unique())   #(le stesse, ottimo così)

# Codifichiamo con dei numeri interi i valori delle classi di 'status' (es. 'Good': 1; 'Moderate': 2; eccetera){etichette target}, andrebbe fatto anche encoder delle feature stringhe
encoder= LabelEncoder()
Y_train_encoded= encoder.fit_transform(Y_train)
Y_val_encoded= encoder.transform(Y_val)
Y_test_encoded= encoder.transform(Y_test)

# Facciamo il one-hot encoding dei valori numerici che abbiamo ottenuto prima codificando le classi di 'status' (es. 1: [0, 0, 1]; 2: [0, 1, 0]; eccetera)
# "num_classes" indica il numero totale di classi di 'status' da codificare (calcolato prima: num_classes= len(df['status'].unique()))
Y_train_onehot= to_categorical(Y_train_encoded, num_classes=num_classes)
Y_val_onehot= to_categorical(Y_val_encoded, num_classes=num_classes)
Y_test_onehot= to_categorical(Y_test_encoded, num_classes=num_classes)

# Facciamo lo scaling dei valori presenti in X_train, X_val ed X_test
scaler= StandardScaler()
#Per le features
X_train_scaled= scaler.fit_transform(X_train) # Fittiamo lo scaler sui valori di X_train e scaliamo i valori
X_val_scaled= scaler.transform(X_val) # Facciamo la stessa cosa sui valori di X_val ma senza fittare di nuovo la scaler (non serve)
X_test_scaled= scaler.transform(X_test) # Facciamolo anche sui valori di X_test
#sul target non si fa scaler

#MODELLO

# Creiamo il modello
model= keras.Sequential([
    # Qua mettiamo il primo layer (ingresso) della rete neurale
    # "shape" indica il numero di ingressi della rete neurale (pari al numero di feature necessarie per fare la predizione, quindi pari al numero di colonne di X_train)
    # Possiamo estrarre il numero di colonne di X_train con "X_train.shape" che restituisce una tupla del tipo (num_righe, num_colonne), e poi selezionando solo il
    # numero di colonne come se fosse un array con "X_train.shape[1]"
    # Il parametro "shape" però accetta solo tuple (perché può funzionare anche con ingressi a più dimensioni: usando una tupla capisce facilmente la dimensione), ma
    # noi abbiamo solo il valore "X_train.shape[1]": dobbiamo creare una tupla a singola dimensione dove mettiamo "X_train.shape[1]" facendo "(X_train.shape[1],)"
    layers.Input(shape=(X_train_scaled.shape[1],)),
    # Qua mettiamo i layer intermedi con quanti neuroni vogliamo. Usiamo 'relu' come funzione di attivazione, posso usare anche più layer input con un numero diverso di neuroni
    layers.Dense(16, activation="relu"),
    layers.Dense(8, activation="relu"),
    # Qua mettiamo il layer finale con tanti neuroni quante sono le possibili classi di 'status' predicibili ("num_classes"). Usiamo 'softmax' come funzione di
    # attivazione perché stiamo facendo una classificazione multiclasse, altrimenti se fosse una classificazione binaria [M o F] dovremmo usare 'sigmoid' (e 1 neurone)
    layers.Dense(num_classes, activation="softmax")
])

# Configuriamo il modello, usare sempre Adam
model.compile(
    optimizer=optimizers.Adam(learning_rate=1e-3),
    loss="categorical_crossentropy",    #loss, se ho regressione: "mean_squared_error"
    metrics=["accuracy"]    #se regressione: "mean_squared_error"
)

# Addestriamo il modello sui dati di train, ovvero:
# -> X_train_scaled
# -> Y_train_onehot (ovvero le classi di 'status' corrispondenti alle righe di X_train: sono le soluzioni della predizione che il modello userà per capire qual'è
#    lo 'status' di una riga)
history= model.fit(
    X_train_scaled, Y_train_onehot,
    validation_data=(X_val_scaled, Y_val_onehot),
    epochs=50, # "epochs" indica il numero di volte che vengono aggiornati i pesi della rete neurale sugli stessi dati di test, quante volte l'esmpio viene visto dal modello durante l addestramento
    batch_size=128, #32,64,128
    verbose=1 # Mostra i progessi dell'addestramento FACOLTATIVO
)

# Vediamo quanto è diventato bravo il modello usando la funzione "model.evaluate()" per calcolare la loss e l'accuracy (% di previsioni corrette) sui dati di test
loss, acc= model.evaluate(X_test_scaled, Y_test_onehot, verbose=0) # "verbose=0" significa che non vogliamo vedere i progressi del calcolo
print(f"Accuracy del modello sulle previsioni di X_test: {acc:.4f}")

# Facciamo predire la classe 'status' dei valori di X_test e salviamo il risultato delle predizioni in "Y_pred"
# Le reti neurali danno in output la probabilità che un dato in ingresso appartenga ad ogni possibile classe predicibile: selezioniamo per ogni input la classe
# in output con maggiore probabilità usando la funzione "argmax(axis=1)" che trasfroma etichette predette in one hot encoding--> se no avrei da una parte one hot encodign, dall altra probabilità
Y_pred= model.predict(X_test_scaled, verbose=0).argmax(axis=1)


# Confrontiamo le classi reali dei dati di test con le classi predette dal modello per creare la matrice di confusione
# "normalize=True" fa in modo che i campi della matrice di confusione non siano corrispondenti al numero complessivo di casi ma che siano invece dei valori tra 0 e 1
cm= confusion_matrix(Y_test_encoded, Y_pred, normalize="true")
# Dobbiamo dire però alla funzione che visualizza la matrice di confusione come si chiamano le classi: lo facciamo nel parametro "display_labels" e gliele diciamo
# in ordine con la funzione sorted()

disp= ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=sorted(Y.unique()))
disp.plot(cmap='Blues') # Formattazione disegno
plt.title("Matrice di confusione") # Titolo disegno
plt.show()

# Usiamo Matplotlib
plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], label='Training Loss') # Disegnamo il grafico dell'andamento del loss calcolato sui dati di train
plt.plot(history.history['val_loss'], label='Validation Loss') # Disegnamo il grafico dell'andamento del loss calcolato sui dati di validation
plt.title('Andamento del loss durante le epoche')
plt.xlabel('Numero epoche trascorse')
plt.ylabel('Loss')
plt.legend() # Mostra la legenda
plt.grid(True) # Formattazione disegno

# Visualization of the validation accuracy trend
plt.figure(figsize=(10, 6))
plt.plot(history.history['accuracy'], label='Train Accuracy') # Disegnamo il grafico dell'andamento dell'accuracy calcolato sui dati di train
plt.plot(history.history['val_accuracy'], label='Validation Accuracy') # Disegnamo il grafico dell'andamento dell'accuracy calcolato sui dati di validation
plt.title('Andamento dell\'accuracy durante le epoche')
plt.xlabel('Numero epoche trascorse')
plt.ylabel('Accuracy')
plt.legend() # Mostra la legenda
plt.grid(True) # Formattazione disegno
plt.show()

#se oltre una certa epoca la validation Loss si Apre, vuol dire che va in overfitting il modello (facendo fatica a generalizzare),ù
# ecco che occorre riaddestarre modificato le epoche mettendo il numero fino a quando vallitazion loss rimane vicina

"""
“Il loss misura quanto il modello sbaglia nelle predizioni ed è utilizzato per aggiornare i pesi durante l’addestramento, 
mentre l’accuracy indica la percentuale di predizioni corrette ed è una metrica di valutazione della performance del modello,
 ma non influisce sull’ottimizzazione.”
"""

#SE REGRESSIONE CAMBIARE LOSS E E TOGLIERE ACCURACY E MATRICE DI CONFUSIONE, MAGARI METTERE LA RETTA
