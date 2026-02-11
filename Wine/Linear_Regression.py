# predico alchol

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.preprocessing import LabelEncoder,StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,root_mean_squared_error

file_path="5b_winequality-white_cat.csv"
if not os.path.exists(file_path):
    raise FileNotFoundError(f"File cvs non trovato al percorso {file_path}")

df=pd.read_csv(file_path,low_memory=False,na_values=['-','','NA','ND','n/a'], nrows=1000000)

print(f"Grandezza del dataset: {df.shape}")
print(f"\nPrime 5 righe del dataset{df.head()}")
print(f"\nInfo su features dataset {df.info()}")    #tutti float tranne quality_cat--> stringa


#Data-cleaning
df.dropna(axis=1,how="all",inplace=True)
df.dropna(axis=0,how="any",subset="alcohol",inplace=True)

#Label Encoding -->quality_cat da str a int per lavorare con regressione lineare
lencoder=LabelEncoder()
df['quality_cat']=lencoder.fit_transform(df['quality_cat'])
df.info()

numeric_cols=df.select_dtypes(include=[np.number]).columns.tolist()
numeric_cols.remove('alcohol')
df.dropna(axis=0,how="any",subset=numeric_cols,inplace=True)    #elimino righe con almeno un valore Nan (qua non serve perchè son tutte piene)
df.info()
print(f"Feature numeriche: {numeric_cols}")

#PRE-PROCESSING

X=df[numeric_cols]
Y=df['alcohol'] #target

#split
X_train,X_test,Y_train,Y_test=train_test_split(
    X,Y,
    train_size=0.7,
    random_state=10
)

print(f"Grandezza X_train {X_train.shape}")
print(f"Grandezza X_test: {X_test.shape}")

#Scale, standardizzo
scaler=StandardScaler()
X_train_scaled=scaler.fit_transform(X_train)
X_test_scaled=scaler.transform(X_test)

#ADDESTRO MODELLO
model=LinearRegression()
model.fit(X_train_scaled,Y_train)

Y_pred=model.predict(X_test_scaled)


plt.scatter(
    Y_test,  #asse X
    Y_pred, #asse Y
    label="Valori predetti"
)

plt.plot(
    [Y_test.min(),Y_test.max()],
    [Y_test.min(),Y_test.max()],
    color='red',
    label="Predizione perfetta"

)
plt.title("Regressione Lineare")
plt.xlabel("Alchol reale")
plt.ylabel("Alchol predetto")
plt.grid()
plt.legend()
plt.show()

#Valutazione del modello

mse=mean_squared_error(Y_test,Y_pred)
rmse=root_mean_squared_error(Y_test,Y_pred)
range=Y_test.max()-Y_test.min()


print(f"MSE:{mse:.2f}")
print(f"RMSE:{rmse:.2f}")
print(f"Range:{range}")
print(f"RMSE in percentuale rispetto al range:{(rmse/range)*100:.2f}%")






