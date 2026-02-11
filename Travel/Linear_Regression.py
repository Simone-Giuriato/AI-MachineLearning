#Regressione Lineare su total price

import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.preprocessing import LabelEncoder,StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,root_mean_squared_error

#individuazione del percorso file
file_path="travel.csv"
if not os.path.exists(file_path):
    raise FileNotFoundError(f"File csv non trovato al percorso{file_path}")

#caricamento del dataSet in un dataFame di pandas--> sostituisco NA,-,'-' con valori Nan, poche righe le inserisco tutte
df=pd.read_csv(file_path,low_memory=False,na_values=['-','','NA','ND','n/a'])

#Info generali sul dataframe
print(f"Grandezza del Dtaframe:{df.shape}")
print(f"Prime 5 righe: {df.head()}")
print("Info generali:")
df.info()   #noto delle stringhe

target='total_price'

#DATA CLEANING
df.dropna(axis=1,how='all',inplace=True)    #rimuovo colonne interamnete nulle (qui non ne ho)
df.dropna(axis=0,how='any',subset=target,inplace=True)  #rimuovo righe con Nan nella colonna target

num_class=len(df[target].unique())
print(f"\nValori assunti dal target: {df[target].unique()}, {num_class} classi")
print(f"\nConteggio valori che assume il target: {df[target].value_counts()}")  #è bilanciato

#Valori assunti dalle features
print("Valori assunti dalle features nominali:")
print("\ncountry:")
print(df['country'].unique())
print("\ntravel_type:")
print(df['travel_type'].unique())
print("\ntransport:")
print(df['transport'].unique())
print("\nseason:")
print(df['season'].unique())
print("\ntravel_insurance:")
print(df['travel_insurance'].unique())
print("\ncancelled:")
print(df['cancelled'].unique())

#Poichè Sklearn per la regressione lineare lavora con feature numeriche occorre convertire i valori nominali in numeri
#Uso il Label Encoder per semplificare le operazioni di codifica sebbene un giusto uso sarebbe:
    #Ordinal Encoding per le variabili ordinali 
    #One Hot Encodign per le variabili nominali
#Non uso il One hot Encoding perchè per ogni classe possibile si andrebbe ad aggiungere una colonna in più al dataFrame rischiando di farlo "esplodere"

le_country=LabelEncoder()
df['country']=le_country.fit_transform(df['country'])

le_traveltype=LabelEncoder()
df['travel_type']=le_traveltype.fit_transform(df['travel_type'])

le_transport=LabelEncoder()
df['transport']=le_transport.fit_transform(df['transport'])

le_season=LabelEncoder()
df['season']=le_season.fit_transform(df['season'])

#le_travelinsurance=LabelEncoder()
#df['travel_insurance']=le_travelinsurance.fit_transform(df['travel_insurance'])

#le_cancelled=LabelEncoder()
#df['cancelled']=le_cancelled.fit_transform(df['cancelled'])

df['travel_insurance']=df['travel_insurance'].map({ #ordinali binari 
    "yes":"1",
    "no": "0"
})

"""
Se travel_insurance = 1 (yes), l’impatto sul prezzo è +b.

Se travel_insurance = 0 (no), l’impatto è 0."""

"""
Attenzione però

Questo non vuol dire che il modello “capisce” yes > no in senso assoluto, ma assegna un effetto positivo o negativo al fatto che sia 1 rispetto a 0.

Se il coefficiente stimato è positivo → “yes” aumenta il prezzo rispetto a “no”.

Se negativo → “yes” riduce il prezzo rispetto a “no”."""

df['cancelled']=df['cancelled'].map({   #ordinali binari
    "yes":"1",
    "no": "0"
})

print("Valori assunti dalle features nominali, post encoder:")
print("\ncountry:")
print(df['country'].unique())
print("\ntravel_type:")
print(df['travel_type'].unique())
print("\ntransport:")
print(df['transport'].unique())
print("\nseason:")
print(df['season'].unique())
print("\ntravel_insurance:")
print(df['travel_insurance'].unique())
print("\ncancelled:")
print(df['cancelled'].unique())



#Ora che ho tutto in numerico faccio una MATRICE DI CORRELAZIONE LINEARE
features_numeriche=df.select_dtypes(include=[np.number])    #seleziono dati con colonne numeriche
corr=features_numeriche.corr()
sns.heatmap(corr,annot=True,cmap='Blues')
plt.title("Matrice di correlazione")
plt.show()

#Essendo una relazione lineare vado a togliere dal dataset quelle fetures con poca correlazione con il target (potendo essere del rumore che influenza il mio modello)
#la regola sarebbe eliminare quelli con r<0.2... qui è tanto 0.2 tolgo quello piccolo
#poi potrei eliminare le media su feature presenti poichè sarebbero data leakege
df=df.drop(columns='booking_id') 
df.drop(columns='age',inplace=True) 
df=df.drop(columns='season') 
df=df.drop(columns='transport') 
df.drop(columns='travel_type',inplace=True) 
df.drop(columns='distance_km',inplace=True) 

print("\nInfo dataset post modifiche:")
df.info()

features_numeriche=df.select_dtypes(include=[np.number]).columns.tolist() #lista delle features numeriche rimaste (sono già tutte numeriche)

features_numeriche.remove(target)   #rimuovo il target che è numerico dalle features (successivamente userò per la matrice X)
print(f"\nFeatures numeriche: {features_numeriche}")
df.dropna(axis=0,how="any",subset=features_numeriche,inplace=True)  #rimuovo righe con valori Nan nelle features numeriche sulle quali svolegrò l'addestramento(non ne ho)


#PRE-PROCESSING
X=df[features_numeriche]
Y=df[target]

#Split

X_train,X_test,Y_train,Y_test=train_test_split(
    X,Y,
    train_size=0.7,  #splitto in 70% del campione per l'allenamento e 30% per il test
    random_state=10
)

print(f"Grandezza X_train: {X_train.shape}")    #noto che entrambi i campioni hanno lo stesso numero di colonne, train e test devono avere lo stesso numero di feature per addestrare il modello bene,altrimenti non saprebbe come interpreatre aìfeatures che magari nel test mancano
print(f"Grandezza X_test: {X_test.shape}")  #se non sono stesso numero, dovrei raggruppare (soprattutto se sbilanciato) o provare con random state diversi


#effettuo lo scaling, standardizzo i dati: fondamnetale per la regresione lineare avere una scala comune
scaler=StandardScaler()
X_train_scaled=scaler.fit_transform(X_train)
X_test_scaled=scaler.transform(X_test)

#MODELLO-->il Target è numerico, uso regressione lineare

model=LinearRegression()
model.fit(X_train_scaled,Y_train)   #addestro il modello

Y_pred=model.predict(X_test_scaled)

#Visualizzaizone del modello:
plt.scatter(
    Y_test,
    Y_pred,
   


)

plt.plot(   #retta predizione perfetta
    [Y_test.min(),Y_test.max()], #asse x
    [Y_test.min(),Y_test.max()], #asse y
    color="red"

)
plt.xlabel("Prezzo Totale reale")
plt.ylabel("Prezzo totale predetto")
plt.grid()
plt.title("Regressione lineare")
plt.show()

#Valutazione del modello:

mse=mean_squared_error(Y_test,Y_pred)
rmse=root_mean_squared_error(Y_test,Y_pred)
range=(Y_test.max()-Y_test.min())

print(f"\nMSE:{mse}")
print(f"\nRMSE:{rmse}")
print(f"\nRange:{range}")
print(f"Rmse rispetto al range: {(rmse/range)*100:.2f}%")

#USANDO TUTTO IN LABEL ENCODER:
#ottengo che il modello sbaglia il 12.20% rispetto al range dei prezzi, 460 su 3770 accettabile per via delle poche features correlate a disposizione
#Il modello sbaglia di 460 euro rispetto ad un prezzo reale

#USANDO UN ORDINAL ENCODER PER LE VARIABILI ordinali (con map) [travel_insurance,cancelled li vedo come binari 0 e 1]:
#Passo a 12.18% è gia un miglioramento... forse bisognerebbe fare il one hot encoding sul nominale

