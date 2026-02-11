#------------------CLASSIFICAZIONE su quality_cat--------------------
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier,plot_tree
from sklearn.metrics import accuracy_score,ConfusionMatrixDisplay,confusion_matrix


#caricamento dataset
file_path="5b_winequality-white_cat.csv"
if not os.path.exists(file_path):
    raise FileNotFoundError(f"File csv con percorso {file_path} non trovato")

df=pd.read_csv(file_path,low_memory=False,na_values=['','-','NA','ND','n/a'],nrows=100000)

print(f"Grandezza dataset: {df.shape}")
df.info()
print(df.head())

#DATA CLEANING
df.dropna(axis=1,how="all",inplace=True) #oinutile, non ho colonne vuote
df.dropna(axis=0,how="any",subset="quality_cat",inplace=True)

print(f"Nuove dimensioni:{df.shape}")

#Sbilanciato?
print(f"\nValori per ogni classe del target: {df['quality_cat'].value_counts().sort_index()}") #fprtemente sbilanciato
#raggruppo per ridurre sbilanciamento
df['quality_cat']=df["quality_cat"].replace({
    'A': 'Alta qualità',
    'B': 'Alta qualità',
    'C': 'Alta qualità',
    'D': 'Media qualità',
    'E': 'Bassa qualità',
    'F': 'Bassa qualità',
    'G': 'Bassa qualità'
})

print(f"\nValori per ogni target (post raggruppamento):{df['quality_cat'].value_counts().sort_index}") #meno sbilanciato


#PRE-PROCESSING
#Non serve Label Encoder ho gia tutto numerico
features_names=df.select_dtypes(include=[np.number]).columns.to_list() # considero come features le colonne numeriche (non serve ho giatutto numerico)
print(features_names)

features_names=df.select_dtypes(include=["int64","float64"]).columns.to_list() #fa la stessa cosa senza np
df.dropna(axis=0,how="any",subset=features_names,inplace=True)


#ADDESTRAMENTO

X=df[features_names]
Y=df["quality_cat"]

#split
X_train,X_test,Y_train,Y_test=train_test_split(
    X,Y,
    train_size=0.7,
    random_state=10
)

print(f"\nDimensione X_test: {X_test.shape}")
print(f"\nDimensione X_train: {X_train.shape}")

#Scale, standardizzo
scaler=StandardScaler()
X_train_scaled=scaler.fit_transform(X_train)
X_test_scaled=scaler.transform(X_test)

#MODEL--> albero decisionale

model=DecisionTreeClassifier(random_state=10)
model.fit(X_train_scaled,Y_train)

plt.figure(figsize=(20,10))
plot_tree(
    model,
    max_depth=2,
    feature_names=X.columns,
    
    class_names=[str(cls)
                for cls in sorted(Y.unique())],#serve per mettere l etichette sul grafico ad albero
    filled=True
)
plt.title("Albero Decisionale")
plt.show()

#Valutiamo il modello
Y_pred=model.predict(X_test_scaled)

#Accuracy
print(f"\nAccuracy del modello: {accuracy_score(Y_test,Y_pred)}")

#matrice di confusione
#plt.figure(figsize=(20,10)) NON SERVE
cm=confusion_matrix(Y_test,Y_pred, normalize="true")
disp=ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=df['quality_cat'].unique())
disp.plot(cmap="Blues")
plt.title("Matrice di Confusione normalizzata")
plt.show()






