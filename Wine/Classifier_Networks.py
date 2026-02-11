#------------------CLASSIFICAZIONE su quality_cat--------------------
##NON ADATTO A RETI NEURALI


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



file_path="5b_winequality-white_cat.csv"
if not os.path.exists(file_path):
    raise FileNotFoundError(f"File csv non trovato al percorso  {file_path}")

#CARICHIAMO IL DATASET
# -> nrows indica il numero di righe massimo da caricare 
df= pd.read_csv(file_path, low_memory=False, na_values=['-', 'NA', 'n/a', 'ND',''], nrows=1000000)

print(f"\nGrandezza del dataset:{df.shape}")
print(f"\nPrime 5 righe: {df.head()}")
df.info()
#vedo che non ho righe nulle, ma alcune con veramente pochi valori non nulli potrei toglierle (Squat4Kg,Bench4Kg,Deadlift4Kg) per eliminare il "rumore"
#ho delle stringhe, e posso togliere MeetID inutile

target = "quality_cat"
print("\nConteggio valori che assume il target:")
print(df[target].value_counts().sort_index()) #è sblianciato,

class_names = df[target].unique()   #quali valori può assumere
num_classes = len(class_names)  #numero di valori che può assumere, ci servirà poi per il one hot encoding
print(f"\nValori che può assumere il target: {class_names}")   
print(df[target].value_counts().sort_index()) #è bilanciato

#DATA CLEANING

df.dropna(axis=1, how='all', inplace=True) # Eliminiamo tutte le colonne che hanno tutti valori None (SOLO VALORI NULLI)
df.dropna(axis=0, how='any', subset=target, inplace=True) # Eliminiamo tutte le righe che hanno un valore None nel campo target poichè non sarebbero classificabili

# Label Encoder
# Poiché il modello utilizzato è un albero decisionale, le variabili categoriali
# vengono codificate tramite LabelEncoder per convertire i valori stringa in interi.
# Nei modelli ad albero il LabelEncoder è adatto anche per variabili nominali,
# in quanto l’algoritmo non assume relazioni di ordine o distanza tra i valori.
#
# In generale, per modelli non ad albero, sarebbe preferibile distinguere tra:
# - variabili ordinali --> ordinal encoding
# - variabili nominali --> one-hot encoding
#
# Il one-hot encoding può aumentare significativamente la dimensionalità del dataset,
# motivo per cui in questo contesto si preferisce l’uso del LabelEncoder.

lencoder=LabelEncoder()




lencoder_target=LabelEncoder()
df['target']=lencoder_target.fit_transform(df[target]) #target in numerico , poi posso chiamarlo nella matrice confusione per ottenre il nome delle classi e non i numeri

#Matrice di correlazione (che funziona solo con tipi numerici) per vedere le correlazioni se posso scartare qualche featuresì inutile
num_cols = df.select_dtypes(include=[np.number]).columns.tolist()   #seleziono le colonne numeriche (tutte)
cm = df[num_cols].corr()
plt.figure(figsize=(12,12))
sns.heatmap(cm, cmap="coolwarm", annot=True, fmt=".2f", linewidths=1.5)
plt.title("Correlation Matrix")
plt.show()
#Noto che non ci sono feature altamente correlate tra loro (tutte inferiori a 0.90), le tengo tutte, nelle reti neurali per classifciazione NON togliere le feature poco correlate possono servire a scoprire relazioni complesse/nascoste

df.dropna(axis=0, how='any', subset=num_cols, inplace=True) #elimino righe non nulle nelle colonne numeriche
print("DataSet post modifiche a righe e colonne")   #qui avrò ancora le fetures che ho eliminato
df.info() #post-modifiche
num_cols.remove('target')
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
    random_state=8
)

# SPLIT 2
# Splittiamo:
# -> le righe presenti in X_train in: righe per l'addestramento (90%: X_train) + righe per la validazione (10%: X_val)
# -> le righe presenti in Y_train in: righe per l'addestramento (90%: Y_train) + righe per la validazione (10%: Y_val)
X_train, X_val, Y_train, Y_val= train_test_split(   #Il secondo split è fatto tra i dati di training
    X_train, Y_train,
    test_size=0.1,
    random_state=8
)

# Mostriamo il numero di righe e colonne presenti in X_train, X_val ed X_test
print("\nNumero di righe e colonne di X_train:", X_train.shape)
print("Numero di righe e colonne di X_validation: ", X_val.shape)   #validazione per vedere se modello sta imparando bene o memorizzando i dati dell esempio(overfitting)
print("Numero di righe e colonne di X_test: ", X_test.shape)

# Vediamo che classi di 'status' sono presenti dentro ad Y_train e ad Y_test
print(f"Classi di target presenti in Y_train: {Y_train.unique()}, {len(Y_train.unique())} classi")
print(f"Classi di target presenti in Y_test:{Y_test.unique()}, {len(Y_test.unique())} classi ")   #stesso numero di classi tra test e train,bene (se non è lo stesso o è sbilanciatissmo [tolgo/raggruppo, o è casualità e cambio random state])

# Codifichiamo con dei numeri interi i valori delle classi di del target 
encoder= LabelEncoder()
Y_train_encoded= encoder.fit_transform(Y_train)
Y_val_encoded= encoder.transform(Y_val)
Y_test_encoded= encoder.transform(Y_test)

# Facciamo il one-hot encoding (li mettiamo a vettori) dei valori numerici che abbiamo ottenuto prima codificando le classi di target 
# "num_classes" indica il numero totale di classi di target da codificare (calcolato prima)
Y_train_onehot= to_categorical(Y_train_encoded, num_classes=num_classes)
Y_val_onehot= to_categorical(Y_val_encoded, num_classes=num_classes)
Y_test_onehot= to_categorical(Y_test_encoded, num_classes=num_classes)



# Facciamo lo scaling dei valori presenti in X_train, X_val ed X_test
scaler= StandardScaler()
X_train_scaled= scaler.fit_transform(X_train) # Fittiamo lo scaler sui valori di X_train e scaliamo i valori
X_val_scaled= scaler.transform(X_val) # Facciamo la stessa cosa sui valori di X_val ma senza fittare di nuovo la scaler 
X_test_scaled= scaler.transform(X_test) 


#MODELLO-->rete neurale classificatrice

# Creiamo il modello
model= keras.Sequential([
    # Qua mettiamo il primo layer (ingresso) della rete neurale
    layers.Input(shape=(X_train_scaled.shape[1],)),
    # Qua mettiamo i layer intermedi con quanti neuroni vogliamo. Posso usare anche più layer input con un numero diverso di neuroni
    layers.Dense(16, activation="relu"),
    layers.Dense(8, activation="relu"),
    # Qua mettiamo il layer finale con tanti neuroni quante sono le possibili classi di 'status' predicibili ("num_classes"). Usiamo 'softmax' come funzione di
    # attivazione perché stiamo facendo una classificazione multiclasse, altrimenti se fosse una classificazione binaria [M o F] dovremmo usare 'sigmoid' (e 1 neurone)
    layers.Dense(num_classes, activation="softmax")
])

# Configuriamo il modello
model.compile(
    optimizer=optimizers.Adam(learning_rate=1e-3),
    loss="categorical_crossentropy",   
    metrics=["accuracy"]    
)

# Addestriamo il modello sui dati di train, ovvero:
# -> X_train_scaled
# -> Y_train_onehot 
history= model.fit(
    X_train_scaled, Y_train_onehot,
    validation_data=(X_val_scaled, Y_val_onehot),
    epochs=50, # "epochs" indica il numero di volte che vengono aggiornati i pesi della rete neurale sugli stessi dati di test, quante volte l'esempio viene visto dal modello durante l addestramento
    batch_size=128, 
    verbose=1 
)

#VALUTAZIONE DEL MODELLO

# Vediamo quanto è diventato bravo il modello usando la funzione "model.evaluate()" per calcolare la loss e l'accuracy (% di previsioni corrette) sui dati di test
loss, accuracy= model.evaluate(X_test_scaled, Y_test_onehot, verbose=0) # 
print(f"Loss: {loss}")
print(f"Accuracy: {accuracy}")

Y_pred = model.predict(X_test_scaled).argmax(axis=1)

#MATRICE DI CONFUSIONE NORMALIZZATA
cm= confusion_matrix(Y_test_encoded, Y_pred, normalize="true")  #mettere Y_test encoded e non one hot
#sns.heatmap(cm, annot=True, fmt=".2f", cmap="Blues", xticklabels=class_names, yticklabels=class_names) #uso questo perchè ho un numero di classi diverse tra test e train e confusion matrux mi da problemi [vedi soluzioni se è sbilanciato/cambia random_state nello split]-->non serve più questo ma nel caso..
disp= ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=encoder.classes_)   #display_labels=sorted(Y.unique()),quesro da la stessa cosa ma senza lettere nelle labels lasciando i numeri del encode 
disp.plot(cmap='Blues') # Formattazione disegno
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
