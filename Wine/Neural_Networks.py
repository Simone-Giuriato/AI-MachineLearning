#rete neurale classifier vs albero, su target:quality_cat
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
from sklearn.metrics import balanced_accuracy_score,accuracy_score
import tensorflow as tf
import keras
from keras import layers
from keras.callbacks import EarlyStopping
from keras import optimizers
from keras import callbacks
from keras.utils import to_categorical
from sklearn.tree import DecisionTreeClassifier,plot_tree

file_path="5b_winequality-white_cat.csv"
if not os.path.exists(file_path):
    raise FileNotFoundError(f"File csv non trovato al percorso  {file_path}")

#CARICHIAMO IL DATASET
# -> nrows indica il numero di righe massimo da caricare (qua ne carichiamo 300.000 per velocizzare l'addestramento)
df= pd.read_csv(file_path, low_memory=False, na_values=['-', 'NA', 'n/a', 'ND',''], nrows=300000)

print(f"\nGrandezza del dataset:{df.shape}")
print(f"\nPrime 5 righe: {df.head()}")
df.info()
#vedo che non ho righe nulle
#tutte righe tranne il target quality_cat sono float/int

target = "quality_cat"
class_names = df[target].unique()   #quali valori può assumere
num_classes = len(class_names)  #numero di valori che può assumere, ci servirà poi per il one hot encoding
print(f"Valori che può assumere il target: {class_names}")
print(df[target].value_counts().sort_index()) #è sblianciato, visto la costruzione del dataset e le categorie potrei fare un raggruppamento qualità Alta,media,Bassa

#DATA CLEANING
df.dropna(axis=0, how='any', subset=target, inplace=True) # Eliminiamo tutte le righe che hanno un valore None nel campo target poichè non sarebbero classificabili
df.dropna(axis=1, how='all', inplace=True) # Eliminiamo tutte le colonne che hanno tutti valori None (SOLO VALORI NULLI)

#faccio encoder per portare in numero le stringhe (qui solo target è stringa)

lencoder=LabelEncoder()
df[target]=lencoder.fit_transform(df[target])

#Matrice di correlazione (che funziona solo con tipi numerici) per vedere le correlazioni se posso scartare qualche featuresì inutile
num_cols = df.select_dtypes(include=[np.number]).columns.tolist()   #seleziono le colonne numeriche (tutte)
cm = df[num_cols].corr()
plt.figure(figsize=(12,12))
sns.heatmap(cm, cmap="coolwarm", annot=True, fmt=".2f", linewidths=1.5)
plt.title("Correlation Matrix")
plt.show()
#Noto che non ci sono feature altamente correlate tra loro (tutte inferiori a 0.90), le tengo tutte, nelle reti neurali per classifciazione togliere le feature poco correlate possono servire a scoprire relazioni complesse/nascoste
"""
***Regressione lineare / modelli lineari → togliere le feature poco correlate conviene molto, perché influiscono direttamente sulla predizione lineare.

***Alberi decisionali / reti neurali → conviene meno togliere le feature poco correlate al target, ecco perché:

    Reti neurali e alberi possono catturare relazioni complesse

    Una feature poco correlata singolarmente con il target può diventare utile se combinata con altre feature

    Quindi eliminare troppo aggressivamente potrebbe peggiorare la performance

***Rimozione di feature altamente correlate tra loro

***Conviene evitare ridondanza → troppe feature simili possono causare overfitting o rallentare l’addestramento

"""

if target in num_cols:  #dopo aver visto la correlazione, se c'è target lo rimuovo perchè num_cols sarà per la matrice X per l addestramento
    num_cols.remove(target)

df.dropna(axis=0, how='any', subset=num_cols, inplace=True) #elimino righe non nulle nelle colonne numeriche, le reti neurali non vogliono Nan
print("DataSet post modifiche a righe e colonne")
df.info() #post-modifiche
print(f"\nColonne numeriche:{num_cols}")  #noto che manca il target, ottimo

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
print("Classi di 'status' presenti in Y_train: ", Y_train.unique())
print("Classi di 'status' presenti in Y_test: ", Y_test.unique())   #AIA IN TEST HO UNA CLASSE IN MENO

# Codifichiamo con dei numeri interi i valori delle classi di del target (es. 'A': 1; 'B': 2; eccetera){etichette target}
encoder= LabelEncoder()
Y_train_encoded= encoder.fit_transform(Y_train)
Y_val_encoded= encoder.transform(Y_val)
Y_test_encoded= encoder.transform(Y_test)

# Facciamo il one-hot encoding (li mettiamo a vettori) dei valori numerici che abbiamo ottenuto prima codificando le classi di target (es. A: [0, 0, 1]; B: [0, 1, 0]; eccetera)
# "num_classes" indica il numero totale di classi di 'status' da codificare (calcolato prima)
Y_train_onehot= to_categorical(Y_train_encoded, num_classes=num_classes)
Y_val_onehot= to_categorical(Y_val_encoded, num_classes=num_classes)
Y_test_onehot= to_categorical(Y_test_encoded, num_classes=num_classes)

# Facciamo lo scaling dei valori presenti in X_train, X_val ed X_test
scaler= StandardScaler()
X_train_scaled= scaler.fit_transform(X_train) # Fittiamo lo scaler sui valori di X_train e scaliamo i valori
X_val_scaled= scaler.transform(X_val) # Facciamo la stessa cosa sui valori di X_val ma senza fittare di nuovo la scaler 
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

# Configuriamo il modello
model.compile(
    optimizer=optimizers.Adam(learning_rate=1e-3),
    loss="categorical_crossentropy",    #loss, se ho regressione: "mean_squared_error"
    metrics=["accuracy"]    #se regressione: "mean_squared_error"
)

# Addestriamo il modello sui dati di train, ovvero:
# -> X_train_scaled
# -> Y_train_onehot (ovvero le classi di target corrispondenti alle righe di X_train: sono le soluzioni della predizione che il modello userà per capire qual'è  lo 'status' di una riga)
history= model.fit(
    X_train_scaled, Y_train_onehot,
    validation_data=(X_val_scaled, Y_val_onehot),
    epochs=50, # "epochs" indica il numero di volte che vengono aggiornati i pesi della rete neurale sugli stessi dati di test, quante volte l'esempio viene visto dal modello durante l addestramento
    batch_size=128, 
    verbose=1 # Mostra i progessi dell'addestramento 
)

#VALUTAZIONE DEL MODELLO

# Vediamo quanto è diventato bravo il modello usando la funzione "model.evaluate()" per calcolare la loss e l'accuracy (% di previsioni corrette) sui dati di test
loss, accuracy= model.evaluate(X_test_scaled, Y_test_onehot, verbose=0) # "verbose=0" significa che non vogliamo vedere i progressi del calcolo
print(f"Loss: {loss}")
print(f"Accuracy: {accuracy}")

Y_pred = model.predict(X_test_scaled).argmax(axis=1)

#MATRICE DI CONFUSIONE NORMALIZZATA
cm= confusion_matrix(Y_test_encoded, Y_pred, normalize="true")
sns.heatmap(cm, annot=True, fmt=".2f", cmap="Blues",xticklabels=class_names, yticklabels=class_names) #uso questo perchè ho un numero di classi diverse tra test e train e confusion matrux mi da problemi
#disp= ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=sorted(Y.unique()))   
#disp= ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=np.arange(num_classes))   
#disp.plot(cmap='Blues') # Formattazione disegno
plt.title("Matrice di confusione") # Titolo disegno
plt.show()



#LOSS DI VALIDAZIONE TREND
plt.figure(figsize=(12, 12))
plt.plot(history.history['loss'], label='Training Loss') # Disegnamo il grafico dell'andamento del loss calcolato sui dati di train
plt.plot(history.history['val_loss'], label='Validation Loss') # Disegnamo il grafico dell'andamento del loss calcolato sui dati di validation
plt.title('Andamento del loss durante le epoche')
plt.xlabel('Numero epoche trascorse')
plt.ylabel('Loss')
plt.legend() 
plt.grid(True) 


#ACCURACY DI VALIDAZIONE TREND
plt.figure(figsize=(10, 6))
plt.plot(history.history['accuracy'], label='Train Accuracy') # Disegnamo il grafico dell'andamento dell'accuracy calcolato sui dati di train
plt.plot(history.history['val_accuracy'], label='Validation Accuracy') # Disegnamo il grafico dell'andamento dell'accuracy calcolato sui dati di validation
plt.title('Andamento Accuracy durante le epoche')
plt.xlabel('Numero epoche trascorse')
plt.ylabel('Accuracy')
plt.legend() 
plt.grid(True) 
plt.show()

"""
Training loss → indica quanto il modello sta imparando sui dati di addestramento

Validation loss → indica quanto il modello sta performando su dati mai visti durante l’addestramento

Training accuracy → percentuale di predizioni corrette sul training set

Validation accuracy → percentuale di predizioni corrette sul validation set
"""

#se oltre una certa epoca la validation Loss si Apre, vuol dire che va in overfitting il modello (facendo fatica a generalizzare),ù
# ecco che occorre riaddestarre modificato le epoche mettendo il numero fino a quando vallitazion loss rimane vicina

#è NORMALE CHE OGNI VOLTA CHE ESEGUO OTTENGO RISULATI UN PO DIVERSI NELLE RETI NEURALI, ogni volta i neuroni partono nda punti diversi... a caso

#CONFRONTO CON ALBERO

# SPLIT 1:
# Splittiamo:
# -> le righe presenti in X in: righe per l'addestramento (80%: X_train) + righe per il test (20%: X_test)
# -> le righe presenti in Y in: righe per l'addestramento (80%: Y_train) + righe per il test (20%: Y_test)
X_train_albero, X_test_albero, Y_train_albero, Y_test_albero= train_test_split(
    X, Y,
    test_size=0.2,
    random_state=42
)

# Codifichiamo con dei numeri interi i valori delle classi di del target (es. 'A': 1; 'B': 2; eccetera){etichette target}
encoder= LabelEncoder()
Y_train_encoded_albero= encoder.fit_transform(Y_train)
Y_test_encoded_albero= encoder.transform(Y_test)

#Niente one hot encodign alberi

# Facciamo lo scaling dei valori presenti in X_train, X_val ed X_test
scaler= StandardScaler()
X_train_scaled_albero= scaler.fit_transform(X_train) # Fittiamo lo scaler sui valori di X_train e scaliamo i valori
X_test_scaled_albero= scaler.transform(X_test) # Facciamolo anche sui valori di X_test


model = DecisionTreeClassifier(random_state=42)
model.fit(X_train_scaled_albero, Y_train_encoded_albero)
Y_pred_albero = model.predict(X_test_scaled_albero)

accuracy = accuracy_score(Y_test_albero, Y_pred_albero)
print(f"Accuratezza modello: {accuracy:.4f}")

cm = confusion_matrix(Y_test_albero, Y_pred_albero,normalize='true' )
disp = ConfusionMatrixDisplay(cm, display_labels=class_names)
disp.plot(cmap='Blues')
plt.show()


#risulta meglio l albero, forse per via del dataset piccolo
