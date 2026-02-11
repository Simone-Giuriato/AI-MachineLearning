#---------------------Target: AGE---------------------
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.preprocessing import StandardScaler,LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,root_mean_squared_error

#Caricamneto dataset
file_path="Social.csv"
if not os.path.exists(file_path):
    raise FileNotFoundError(f"File csv non trovato al percorso {file_path}")

df=pd.read_csv(file_path,low_memory=False,na_values=['','-','NA','ND','n/a'],nrows=1000000)

#info dataset per capire con cosa sto lavorando
print(f"Grandezza del dataset: {df.shape}")
print(f"Prime 5 righe: {df.head()}")
df.info()

#data-cleaning
df.dropna(axis=1,how='all',inplace=True)    #elimino colonna con tutti Nan
df.dropna(axis=0,how='any',subset='Age',inplace=True)   #elimino righe con Nan nel traget

print(f"\nValori che assume il target: {df['Age'].value_counts()}")

#Linear Regression lavora con i numeri:
#Label Encoder (Gender,Academic_leve,Country,Most_used_paltform,Affects,Relatinship)
lencoder=LabelEncoder()

df['Gender']=lencoder.fit_transform(df['Gender'])
df['Academic_Level']=lencoder.fit_transform(df['Academic_Level'])
df['Country']=lencoder.fit_transform(df['Country'])
df['Most_Used_Platform']=lencoder.fit_transform(df['Most_Used_Platform'])
df['Affects_Academic_Performance']=lencoder.fit_transform(df['Affects_Academic_Performance'])
df['Relationship_Status']=lencoder.fit_transform(df['Relationship_Status'])

#ANDREBBE FATTA LA MATRICE DI CORELAZIONE, ELIMINANDO LE FETRES POCO CORRELATE (NEI CLASSIFICATORI LE LASCEREI POICHè CI POSSONO ESSERE ALTRE CONNESSIONI NON LINEARI, NASCOSTEE)

#salvo solo features numeriche
features=df.select_dtypes(include=[np.number]).columns.tolist()
features.remove("Student_ID")#non è utile
features.remove("Age")  #è numerico ma è anche target
df.dropna(axis=0,how="any",subset=features,inplace=True)    #rimuovo righe con Nan nelle colonne delel features utili (qui dovrebbero essere tutte piene)
print(f"\nFeatures numeriche: {features}")
df.info()

#PRE-PROCESSING

X=df[features]  #matrice con  features utili
Y=df["Age"] #matrice target
#split

X_train,X_test,Y_train,Y_test=train_test_split(
    X,Y,
    test_size=0.3,
    train_size=0.7,
    random_state=10
)

print(f"Dimensioni X_train: {X_train.shape}")
print(f"Dimensioni X_test: {X_test.shape}")

#Split,Standardizzo

scaler=StandardScaler()
X_train_scaled=scaler.fit_transform(X_train)    #impara e trasforma
X_test_scaled=scaler.transform(X_test)  #test solo trasforma

#Modello-->Regressione lienare

model=LinearRegression()
model.fit(X_train_scaled,Y_train)

Y_pred=model.predict(X_test_scaled)

plt.scatter(
    Y_test, # ASSE X → valori reali
    Y_pred, # ASSE Y → valori predetti
    color="Pink",
    label="Valori predeti"
    
)

plt.plot(   #retta ideale
    [Y_test.min(),Y_test.max()],
    [Y_test.min(),Y_test.max()],
    color="Purple",
    label="Valori ideali"
)

plt.title("Regressione Lineare")
plt.xlabel("Valori reali Age")
plt.ylabel("Valori predetti Age")
plt.legend()
plt.grid()
plt.show()



#Valutazione modello

mse=mean_squared_error(Y_test,Y_pred)
rmse=root_mean_squared_error(Y_test,Y_pred) #mi permette valutazione errore nell'unità di misura del target
range=Y_test.max()-Y_test.min()

print(f"\nMSE: {mse:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"Range: {range}")
print(f"Rmse in percentuale rispetto al range: {(rmse/range)*100:.2f}")






"""""
X = [X0, X1] = [Y_test.min(), Y_test.max()]
Y = [Y0, Y1] = [Y_test.min(), Y_test.max()]
Collega (X0,Y0) → (X1,Y1) con una linea

Risultato → diagonale perfetta
"""""