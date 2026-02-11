import os 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.preprocessing import LabelEncoder,StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import root_mean_squared_error,mean_squared_error,accuracy_score



#Individuazione del percrso del file csv
file_path="exam_score.csv"
if not os.path.exists(file_path):
    raise FileNotFoundError(f"File csv non trovato al percorso {file_path}")

#CARICAMENTO DATASET
df=pd.read_csv(file_path,low_memory=False,na_values=['n/a','NA','ND','','-'])   #carico tutte le righe (20mila) e sostitusico i valori mancanti con Nan di pandas

#Info generali sul dataset:
print(f"\nGrandezza Dataset: {df.shape}")
print(f"Prime 5 righe dataset {df.head()}")
print("Info generali sul dataset:")
df.info() #noto la presenza di stringhe
target='exam_score' #target numerico--> regressione lineare

#DATA-CLEANING
df.dropna(axis=1,how="all",inplace=True)    #elimino eventuali colonne vuote (Nan intere)
df.dropna(axis=0,how="any",subset=target,inplace=True)  #elimino righe con Nan sul target che danno noia al modello


print("\nValori assunti dal feature categoriche:")
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
#Regressione Lineare lavora bene con i numeri motivo per cui dovrei codificare le feature stringhe
#Le procedure teoriche per modelli come Regressione Lineare indicano:
    #Ordinal Encoding: variabili ordinali
    #One Hot Encoding: variabili nominative

#Per semplifciare le analisi uso il Label Encoder che assegna la codifica in ordine alfabetico... il One hot encoding (scelta più giusta) crea colonne per ogni classe e rischio di far "esplodere il dataset"

#Label Encoder
le_gender=LabelEncoder()
df['gender']=le_gender.fit_transform(df['gender'])

le_course=LabelEncoder()
df['course']=le_course.fit_transform(df['course'])

le_internetaccess=LabelEncoder()
df['internet_access']=le_internetaccess.fit_transform(df['internet_access'])

#le_sleepquality=LabelEncoder()
#df['sleep_quality']=le_sleepquality.fit_transform(df['sleep_quality'])
df['sleep_quality']=df['sleep_quality'].map({   #faccio un ordinal encoding
    "poor":0,
    "average":1,
    "good":2
})

lestudymethod=LabelEncoder()
df['study_method']=lestudymethod.fit_transform(df['study_method'])

#lefacility=LabelEncoder()
#df['facility_rating']=lefacility.fit_transform(df['facility_rating'])
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


print("\nValori assunti dal feature categoriche (post encoding):")
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


#Matrice di correlazione Lineare
numeric_cols=df.select_dtypes(include=["float64","int64"])
corr=numeric_cols.corr()
sns.heatmap(corr,annot=True,cmap="Blues")
plt.title("Matrice di correlazione")
plt.show()
#La matrice di correlazione lineare mi mostra delle features con correlazione lineare molto basse col target che potrei rimuovere per evitare il rumore
df.drop(columns='student_id',inplace=True)
df.drop(columns='course',inplace=True)
df.drop(columns='age',inplace=True)
df.drop(columns='gender',inplace=True)
df.drop(columns='internet_access',inplace=True)
df.drop(columns='exam_difficulty',inplace=True)

print("Dataset post pulizia:")
numeric_cols=df.select_dtypes(include=["float64","int64"]).columns.tolist() #lista delle feature numeriche che usrò nella matrice X
df.dropna(axis=0,how="any",subset=numeric_cols,inplace=True)    #rimuovo righe con Nan nelle features numeriche
numeric_cols.remove('exam_score')   #rimuovo il target dalle numeriche
df.info()
print(f"Feature numeriche importanti: {numeric_cols}")

#PRE-PROCESSING
X=df[numeric_cols]
Y=df[target]

#Split
X_train,X_test,Y_train,Y_test=train_test_split(
    X,Y,
    train_size=0.7, #70% del dataset per allenamento, 30% per il test
    random_state=10
)

print(f"Grandezza campione allenamento: {X_train.shape}")   #stessi numero di colonne bene
print(f"Grandezza campione test: {X_test.shape}")

#Scaling, standardizzo i numeri
scaler=StandardScaler()
X_train_scaled=scaler.fit_transform(X_train)
X_test_scaled=scaler.transform(X_test)

#MODELLO--> Regressione lineare poichè devo predire un numero

model=LinearRegression()
model.fit(X_train_scaled,Y_train)   #addestro modello con campione allenmento

Y_pred=model.predict(X_test_scaled)

#Visualizzazione Modello

plt.plot(   #retta -->plot
     [Y_test.min(),Y_test.max()],
    [Y_test.min(),Y_test.max()],
    color="Red" 
)

plt.scatter(    #grafico dispersione
   Y_test,
   Y_pred,
   color="Blue"
    
)
plt.grid()
plt.xlabel("Valori reali")
plt.ylabel("Valori predetti")
plt.title("Regressione Lineare")
plt.show()


#VALUTAZIONE Modello

mse=mean_squared_error(Y_test,Y_pred)
rmse=root_mean_squared_error(Y_test,Y_pred)
range=(Y_test.max()-Y_test.min())

print(f"MSE: {mse}")
print(f"RMSE: {rmse}")
print(f"range: {range}")
print(f"RMSE in funzione del range: {(rmse/range)*100:.2f}%")   #senza ordinal encoding ottengo un 13.46%


#Con ordinal Encoding ottengo un rmse inferiore arrivando ad un 12.47% sul range




#PROVO AD addestrae una rete neurale per vdere se cambia qualcosa
import tensorflow as tf
import keras
from keras import layers
from keras.callbacks import EarlyStopping
from keras import optimizers
from keras import callbacks
from keras.utils import to_categorical
print("\n\n\nRETE NEURALE")
# Scegliamo come X (le feature che il modello userà per fare la predizione) tutte le feature numeriche 
X= df[numeric_cols]
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
plt.plot([Y_test.min(), Y_test.max()], [Y_test.min(), Y_test.max()], color="red", label="Linear Regression ")
plt.title("Regressione Lineare(RETE NEURALE)")
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
# in questo caso vediamo che la rete ha performance simili a quanto visto negli altri sistemi ( se non peggio )