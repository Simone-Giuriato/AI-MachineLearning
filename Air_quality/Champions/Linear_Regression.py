#predico HomeGoals

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.preprocessing import LabelEncoder,StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,root_mean_squared_error
from sklearn.model_selection import train_test_split

file_path="champions_league.csv"
if not os.path.exists(file_path):
    raise FileNotFoundError(f"File csv non trovato al percoso{file_path}")

#Carciamneto dataset
df=pd.read_csv(file_path,low_memory=False,na_values=['','-','NA','ND','n/a'])

print(f"Grandezza dataset {df.shape}")
print(f"Prime righe dataset {df.info()}")
print("Info generali")
df.info()
target='HomeGoals'


#DATA CLEANING
df.dropna(axis=1,how="all",inplace=True)
df.dropna(axis=0,how="any",subset=target,inplace=True)

print("Valori che assumone features non numeriche:")
print("AwayTeam")
print(df['AwayTeam'].unique())
print("HomeTeam")
print(df['HomeTeam'].unique())
print("MatchResult")
print(df['MatchResult'].unique())


#noto che ci sono variabili categoriche--> converto in numero con labelEncoder
le_home=LabelEncoder()
le_away=LabelEncoder()
le_result=LabelEncoder()

df['HomeTeam']=le_home.fit_transform(df['HomeTeam'])
df['AwayTeam']=le_away.fit_transform(df['AwayTeam'])
df['MatchResult']=le_result.fit_transform(df['MatchResult'])


print("\nValori che assumone features non numeriche(post encoding):")
print("AwayTeam")
print(df['AwayTeam'].unique())
print("HomeTeam")
print(df['HomeTeam'].unique())
print("MatchResult")
print(df['MatchResult'].unique())

#MATRICE DI CORRELAZIONE LINEARE
numeric_cols=df.select_dtypes(include=[np.number]).columns.tolist()
corr=df[numeric_cols].corr()
sns.heatmap(corr,annot=True,cmap="Blues")
plt.title("Matrice correlazione lienare")
plt.show()

#Noto che ci sono feature poco correlate linearmente (|r|<0.2) che rimuovo perchè causano rumore e distorsione per il mio modello
numeric_cols.remove('HomeTeam')
numeric_cols.remove('AwayTeam')
numeric_cols.remove('AwayShots')
numeric_cols.remove('AwayShotsOnTarget')
numeric_cols.remove('AwayGoals')
numeric_cols.remove('AwayShotAccuracy')
#Giustamente noto che rimuovo le features che riguardano gli avversari... sono interessato ai gol della squadra di casa
numeric_cols.remove('HomeGoals')    #rimuovo tagret che essendo numerica faceva parte delle features numeriche in considerazione

print(f"\nFeatures numeriche {numeric_cols}")
df.dropna(axis=0,how="any",subset=numeric_cols,inplace=True)

print("Info dataset che userò post pulizia:")
df[numeric_cols].info()

#PRE-PROCESSING

X=df[numeric_cols]
Y=df[target]

#Split
X_train,X_test,Y_train,Y_test=train_test_split(
    X,Y,
    train_size=0.7,
    random_state=10
)

print(f"Grandezza campione allinemaneto : {X_train.shape}") #stessse colonne bene
print(f"Grandezza campione test : {X_test.shape}")


#scaling
scaler=StandardScaler()
X_train_scaled=scaler.fit_transform(X_train)
X_test_scaled=scaler.transform(X_test)

#MODELLO--> regressione lineare

model=LinearRegression()
model.fit(X_train_scaled,Y_train)

Y_pred=model.predict(X_test_scaled)
#Visualizzazione 


plt.scatter(    #grafico a dispersione
    Y_test, #asse x
    Y_pred, #asse y
    color="Blue"
)
plt.plot(   #Retta di riferimento
    [Y_test.min(),Y_test.max()],    #asse x
    [Y_test.min(),Y_test.max()],    #asse y
    color="Red"
)

plt.grid()
plt.xlabel("Valori reali")
plt.ylabel("Valori predetti")
plt.title("Regressione lineare")
plt.show()

#VALUTAZIONE MODELLO

mse=mean_squared_error(Y_test,Y_pred)
rmse=root_mean_squared_error(Y_test,Y_pred)
range=(Y_test.max()-Y_test.min())

print(f"\nMSE: {mse}")
print(f"RMSE: {rmse}")
print(f"range: {range}")
print(f"\n Rmse sul range: {(rmse/range)*100:.2f}%")




