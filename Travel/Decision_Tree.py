#target 'cancelled'
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score,ConfusionMatrixDisplay,confusion_matrix


#definisco il percorso del file
file_path="./travel.csv"

#verifico che il file sia presente al mio percorso:
if not os.path.exists(file_path):
    raise FileNotFoundError("File csv non è presente al percorso: ",file_path)

#carico il dataset in un dataframe di pandas
df=pd.read_csv(file_path,low_memory=False, na_values=['-','','NA','ND','n/a'])  #identifico come Nan i valori -,''... che rappresnetano vuoti/dati mancanti, ho poche righe(5000) le carico tutte

#Verifico informazioni sul dataset caricato
print(f"\nGrandezza dataset: {df.shape}")
print(f"Prime righe:{df.head()}")
print("\nInfo generali sul DataSet:")
df.info()   #noto avere features numeriche e nominali

target='cancelled' #target è stringa--> andrò ad utilizzare un albero decisionale

#DATA-CLEANING

df.dropna(axis=1,how="all",inplace=True)    #elimini le colonne vuote (qui non ce ne sono)
df.dropna(axis=0,how="any",subset=target,inplace=True)  #elimino le righe con Nan nel target

print(f"\nValori Assunti dal target: {df[target].value_counts()}")  #è abbastanza sbilanciato però non posso farci nulla

#Siccome Sklearn per gli Alberi Decisionali lavora con features numeriche devo andare a convertire, quelle che sono nominali
#Dato che gli alberi decisionali non ragionano sui valori o sulla distanza posso usare un Label Encoder per la semplificazione dell'encoding
#In un modello NON albero, andrebbe usato un Ordinal Encoding per le ordinali, e un One hot Encoding per le categoriche

le_country=LabelEncoder()
df['country']=le_country.fit_transform(df['country'])

le_traveltype=LabelEncoder()
df['travel_type']=le_traveltype.fit_transform(df['travel_type'])

le_transport=LabelEncoder()
df['transport']=le_transport.fit_transform(df['transport'])

le_season=LabelEncoder()
df['season']=le_season.fit_transform(df['season'])

le_travel_insurance=LabelEncoder()
df['travel_insurance']=le_travel_insurance.fit_transform(df['travel_insurance'])

#Per target nominale, l’encoding è opzionale. Sklearn lo gestisce anche se lasci stringhe.
le_target=LabelEncoder()
df['cancelled']=le_target.fit_transform(df['cancelled']) #converto target in numero per la amtrice di correlazione


#Matrice di correlazione

features_numeriche=df.select_dtypes(include=[np.number])    #seleziono dati del df delle sole colonne numeriche
corr=features_numeriche.corr()
plt.figure(figsize=(20,10))
sns.heatmap(corr,annot=True,cmap='Blues')
plt.title("Matrice di correlazione")
plt.show()

#Dalla matrice di correlazione potrei andare a togliere dal dataset quelle features che hanno poca correlazione lineari col target.
#Questo ragionamento è utile con la regressione lineare, ma con gli alberi potrebbero esserci relazioni extra lineari "nascoste" utili per gli alberi
#Non rimuovo feature con correlazione bassa perché gli alberi possono sfruttare pattern non lineari.
#elimino la features boking_id inutile per il mio modello
df.drop(columns='booking_id',inplace=True)


print("\nDataFrame post modifiche:")
df.info()
features_numeriche=df.select_dtypes(include=[np.number]).columns.tolist()   #seleziono nomi colonne numeriche
features_numeriche.remove('cancelled')
print(f"\nColonne numeriche: {features_numeriche}")


#PRE-PROCESSING

X=df[features_numeriche]    #Matrice delle X (solo numeriche)
Y=df[target]    #Matrice target (stringa o devo convertire???) [Qui numerico]
#print(Y.dtype)


#effettuto lo split 

X_train,X_test,Y_train,Y_test=train_test_split(
    X,Y,
    train_size=0.7, #effettuto una divisione con 70% dati per l'addestramento, e 30% per il test
    random_state=10
)

print(f"\nDimensione X_train: {X_train.shape}")
print(f"Dimensioni X_test: {X_test.shape}")

#effettuto lo Scaling, standardizzando i dati [caling non necessario per alberi, ma lo faccio per eventuali modelli futuri.]

scaler=StandardScaler()

X_train_scaled=scaler.fit_transform(X_train)    #stesso numero di colonne bene [se fosse diverso darebbe errore, o il modello non impara bene--> dovrei andare a ragruppare poichè magari dataset sbilanciato]
X_test_scaled=scaler.transform(X_test)

#MODELLO--> albero decisionale poichè traget è una categoria

model=DecisionTreeClassifier(random_state=10,max_depth=10)   #con max_depth 3 qua, l abero fa fino a 3 (3 domande)

model.fit(X_train_scaled,Y_train)   #addestro il modello sul campione di allenamento (scalato)

#Visualizzazione del modello


plot_tree(model,
          max_depth=3,  #max_depth qua, visualizzo 3 di profondità ma l albero in sè è piu grande [in base alla profndità messa su Decision Tree]
          feature_names=features_numeriche,
          class_names=le_target.classes_,
          filled=True
          
          )
plt.title("Albero decisonale: ")
plt.show()

#VALUTAZIONE DEL MODELLO:

Y_pred=model.predict(X_test_scaled)

#accuracy valutazione di quanto sbaglia
accuracy=accuracy_score(Y_test,Y_pred)

print(f"\nAccuracy: {accuracy:.2f}") #1.00 L’accuracy elevata è dovuta al forte sbilanciamento della classe target; l’albero tende a predire la classe dominante.

#Matrice di confusione
cm=confusion_matrix(Y_test,Y_pred,normalize="true")
disp=ConfusionMatrixDisplay(cm,display_labels=le_target.classes_)
disp.plot(cmap="Blues")
plt.title("Matrice di confusione Normalizzata")
plt.show()


#Il modello va in Overfitting... accuracy=1 e matrice di confusione perfetta perchè il dataset è profondamnete sbilanciato con yes e no
#Anche buttando a caso farebbe bene

#La profondità dell’albero indica quante domande consecutive l’albero può fare per arrivare a una decisione.
#Albero semplice → poche regole, decisioni generali--> evita overfitting ma non cattura pattern nascosti
#Albero complesso → molte regole, decisioni specifiche--> accuracy alta sul train, cattura pattern nascosti ma rischia overfitting non generalizzando bene


#Depth bassa → underfitting (errore alto su train)

#Depth alta → overfitting (modello troppo specifico, sensibile al rumore)




