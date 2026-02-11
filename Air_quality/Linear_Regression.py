#---------------------------Lavoro su aqi (che si collega con status)-----------------------------
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,root_mean_squared_error

#CARICAMENTO DATASET
file_path="air_quality.csv"
if not os.path.exists(file_path):
    raise FileNotFoundError(f"File csv non trovato al percorso {file_path}") # raise genera un'eccezione: blocca il programma e stampa messaggio di errore

df=pd.read_csv(file_path,low_memory=False,na_values=['','-','NA','ND','n/a'],nrows=1000000)
# legge il file air_quality e lo salva in una tabella (DataFrame)
 # low_memoty=False--> evita warning sui tipi di dato

# controllo il dataset per capire cosa posso fare
print(f"\nGrandezza del dataset: {df.shape}")
print(f"\nPrime 5 righe: {df.head()}")
df.info()

#DATA CLEANING
df.dropna(axis=1,how="all",inplace=True) # elimino le colonne nulle
df.dropna(axis=0,how="any",subset='aqi',inplace=True)   
df.info()

# Label Encoder

# In generale, per modelli non ad albero [come la regressione lineare], sarebbe preferibile distinguere tra:
# - variabili ordinali → ordinal encoding
# - variabili nominali → one-hot encoding
#
# Il one-hot encoding può aumentare significativamente la dimensionalità del dataset,
# motivo per cui in questo contesto si preferisce l’uso del LabelEncoder in questo caso

#Faccio il mapping delle feature che devono avere un ordine di importanza perchè ho notato che l'encoding, dato che ragiona in ordine alfabetico, non da il giusto peso alle feature
#Così facendo miglioro di molto le predizioni del modello(map-->ordinal encoding) [qui non lo faccio]
lencoder=LabelEncoder()

df['county']=lencoder.fit_transform(df['county'])
df['pollutant']=lencoder.fit_transform(df['pollutant'])
df['sitename']=lencoder.fit_transform(df['sitename'])
#status non lo trasformo, così lo cavo dirttamente che è troppo collegato al target aqi (sono accoppiati)
# ======================= Visualizzazione dati: MATRICE DI CORRELAZIONE =========================

#(MATRICE DI CORRELAZIONE)

num_col_data=df.select_dtypes(include=["Float64","int64"]) #seleziono solo colonne che hanno numeri (escluso la data e status) [county,sitename sono in numeri grazie label encoder]
cm=num_col_data.corr() #calcola correlazione tra le colonne numeriche
plt.figure(figsize=(20,10))#visualizzare graficamente la matrice di correlazione
sns.heatmap(cm,annot=True,cmap="coolwarm",fmt=".2f",linewidths=0.5)
plt.title("Matrice di correlazione")
plt.show()
df.info()

#Tolgo le colonne con bassisima correlazione linaere [|r|<20] visto che è una relazione linaere e non considereo evntuali relazioni nascoste (non lineari) come avviene nel classificatore
df.drop(columns=['sitename'],inplace=True)
df.drop(columns=['county'],inplace=True)
df.drop(columns=['no'],inplace=True)
df.drop(columns=['windspeed'],inplace=True)
df.drop(columns=['winddirec'],inplace=True)
df.drop(columns=['longitude'],inplace=True)
df.drop(columns=['latitude'],inplace=True)
df.drop(columns=['siteid'],inplace=True)
#Rimuovo feature delle medie calcolate da feature nel dataset che rappresentano data leakege
df.drop(columns=['so2_avg'],inplace=True)
df.drop(columns=['pm2.5_avg'],inplace=True)
df.drop(columns=['pm10_avg'],inplace=True)


# Seleziona tutte le colonne numeriche (int, float) per l’addestramento del modello (tranne aqi che è il target)
numeric_cols=df.select_dtypes(include=[np.number]).columns.tolist() #qui prendo solo i nomi, in num_col_data anche i dati
numeric_cols.remove('aqi')  #rimuovo aqi che è il target (ed era numerico)
#elimino le righe con almeno un Nan nelle colonne numeriche
df.dropna(axis=0,how='any',subset=numeric_cols,inplace=True)
print(f"\nLa nuova dimensione del dataframe è:{df.shape}\n")
df.info()
print(numeric_cols)

#PRE-PROCESSING

X=df[numeric_cols] # Definiamo X (le feature che il modello userà per fare la predizione) (feature numeriche)
Y=df['aqi'] # Scegliamo come Y (la feature target che il modello dovrà predire) la feature 'aqi'


#split
X_train,X_test,Y_train,Y_test=train_test_split(
    X,Y,
    train_size=0.7,  #30% per il test e 70% per l'addestramento
    random_state=10
)

print(f"\nDimensioni X_train {X_train.shape}")
print(f"\nDimensioni X_test {X_test.shape}")

#scaler (# OSS non è neccessario standardizzare, ma facciamo lo stesso)
scaler=StandardScaler()
X_train_scaled=scaler.fit_transform(X_train)
X_test_scaled=scaler.transform(X_test)

#MODELLO
model=LinearRegression() # Creiamo il modello
model.fit(X_train_scaled,Y_train) # Addestriamo il modello sui dati di train

# Testiamo quanto è diventato bravo il modello facendogli predire l''aqi' delle righe di X_test
Y_pred=model.predict(X_test_scaled)# Mettiamo i risultati di ogni predizione (una per ogni riga di X_test) in Y_pred

mse=mean_squared_error(Y_test,Y_pred)  # MSE-->media degli errori al quadrato,È difficile da interpretare perché è in unità “al quadrato”
rmse=root_mean_squared_error(Y_test,Y_pred) #RMSE--> È l’errore medio nelle stesse unità del target
range=Y_test.max()-Y_test.min() #tutto dipende dalla scala del range, È una misura di quanto si estendono i dati.


print("\nValutazione performance del modello:")
print(f"\nMSE: {mse:.2f}")
print(f"\nRMSE:{rmse:.2f}")
print(f"\nIl range di aqi è: {range:.2f}")
print(f"\nRMSE in percentuale: {(rmse/range)*100:.2f}") #Il modello sbaglia in media il 4.60% dell’intero range possibile

# Stampiamo sull'asse X i valori reali di 'aqi' e sull'asse Y i valori predetti
# Ogni punto rappresenta una osservazione del test set.
plt.scatter( #per disegnare un grafico a dispersione (
   Y_test, # Valori dell'asse X
   Y_pred, # Valori corrispondenti sull'asse Y

   
)

# Stampiamo una linea retta che rappresenta la predizione perfetta
plt.plot(
    [Y_test.min(),Y_test.max()],
    [Y_test.min(),Y_test.max()],
    color='red'
    
)
plt.xlabel("AQI reale") # Titolo asse X
plt.ylabel("AQI predetto") # Titolo asse Y
plt.title("Regressione lineare: AQI reali vs AQI predetti") 
plt.grid() # Formattazione disegno
plt.show()


