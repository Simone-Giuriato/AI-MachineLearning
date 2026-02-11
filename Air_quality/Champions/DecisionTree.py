import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.preprocessing import LabelEncoder,StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier,plot_tree
from sklearn.metrics import accuracy_score,ConfusionMatrixDisplay,confusion_matrix


file_path="champions_league.csv"
if not os.path.exists(file_path):
    raise FileNotFoundError(f"File csv not found al percorso {file_path}")

#Caricamneto Dataset
df=pd.read_csv(file_path,low_memory=False,na_values=['','-','NA','ND','n/a'])   #sostituisco valori vuoti tipo - '' Na con Nan di pandas
#poche righe le carico tutte 120


#Info generali del dataset caricato
print(f"Grandezza Dataframe:{df.shape}")
print(f"Prime 5 righe {df.head()}")
print("Info generali:")
df.info()   #noto tipi di dato stringhe (object) e numerici

target='MatchResult' #tipo stringa, dovò classificare-->albero decisionale

#DATA-CLEANING
df.dropna(axis=1,how="all",inplace=True) #elimino colonne completamente nulle (qui non ne ho)
df.dropna(axis=0,how="any",subset=target,inplace=True) #elimino righe con valore Nan sul tagret, poichè alberi non lavorano con Nan

print(f"Valori assunti da target: {df[target].value_counts()}") #target è abbastanza bilanciato

#Che valori assumono le features non numeriche?
print("\nValori assunti da features NON numeriche:")
print("\nHomeTeam")
print(df['HomeTeam'].unique())
print("\nAwayTeam")
print(df['AwayTeam'].unique())
print("\MatchResult (target)")
print(df['MatchResult'].unique())

#Poichè sklearn con gli alberi decisionali lavora con valori numerici, dovrò convertire le variabili categoriche in numeri
#Per i modelli NON ad Albero la scelta corretta sarebbe:
    #Ordinal Encodign--> variabili categoriche ordinali
    #One hot Encoding--> varibili categoriche nominative
#Per un'analisi un pò più semplificata si userà LabelEncoder anche per le nominative poichè One Hot Encoding aggiugerebbe una colonna per ogni classe della features in questione, rischiando aumentare notevolmente la dimensione del dataframe

le_home=LabelEncoder()
df['HomeTeam']=le_home.fit_transform(df['HomeTeam'])

le_away=LabelEncoder()
df['AwayTeam']=le_away.fit_transform(df['AwayTeam'])

le_target=LabelEncoder()
df['MatchResult']=le_target.fit_transform(df['MatchResult'])

print("\nValori assunti da features NON numeriche (post encoder):")
print("\nHomeTeam")
print(df['HomeTeam'].unique())
print("\nAwayTeam")
print(df['AwayTeam'].unique())
print("\MatchResult (target)")
print(df['MatchResult'].unique())


#MATRICE DI CORRELAZIONE LINEARE
numeric_cols=df.select_dtypes(include=[np.number])
corr=numeric_cols.corr()
sns.heatmap(corr,annot=True,cmap="Blues")
plt.title("Matrice di correlazione lineare")
plt.show()

#Potrei andare ad eliminare le feature poco correlate linearmente poichè potrebbero generare rumore che distorce il modello.
#MA precluderei la possibilità agli alberi di scovare pattern nascosti (extra lineari). Motivo per cui ritengo la amtrice di correlazione lineare sostanzilmente inutile in questo caso
#DOPO LA prima esecuzione il modello è andato in overfitting ho riguardato le etichette ed è chiaro che avendo a disposizione go squadra di casa e trasferta con una regola basica sapeva il riusltato, infatti hanno una correlazione "alta" col target

numeric_cols=df.select_dtypes(include=[np.number]).columns.tolist()

numeric_cols.remove(target) #target avevo covertito in numerico, lo toglo dalle feature che userò per matrice X
numeric_cols.remove('HomeGoals')    #rimuovo per evitare Overifitting, dataleakege: modello capisce la regola: golHome>golAway--> MatchResult:Home o vicevrsaa
numeric_cols.remove('AwayGoals')     #rimuovo per evitare Overifitting, dataleakege
print(f"Features numeriche: {numeric_cols}")

df.dropna(axis=0,how="any",subset=numeric_cols,inplace=True)    #rimuovo righe con Nan nelle features numeriche che userò per il modello

print("\nInfo post pulizia:")
df.info()

#PRE-PROCESSING

X=df[numeric_cols]
Y=df[target]

#Split

X_train,X_test,Y_train,Y_test=train_test_split(
    X,Y,
    train_size=0.7, #splitto il set, in 70% per l'allenmento e 30% per il test
    random_state=10
)

print(f"\nGrandezza X_train {X_train.shape}")   #noto stesso numero di colonne, quindi l'albero può addestrarsi bene associando le features corrispondenti
print(f"\nGrandezza X_test: {X_test.shape}")    #noto un corretto split 70% e 30%

#Scaling,standardizzo i dati (con alberi decisonali si potrebbe fare a meno)
scaler=StandardScaler()
X_train_scaled=scaler.fit_transform(X_train)
X_test_scaled=scaler.transform(X_test)

#MODEL-->Albero Decisionale

model=DecisionTreeClassifier(random_state=10)   #profondità massima, senza vincoli
model.fit(X_train_scaled,Y_train)   #addestro modello

#Visualizzazione del modello
plt.figure(figsize=(15,10))
plot_tree(
    model,
    max_depth=2,        #disegno depth=3
    feature_names=X.columns,
    class_names=le_target.classes_,
    filled=True
)
plt.title("Albero Decisionale (max_depth)")
plt.show()

#VALUTAZIONE del modello

Y_pred=model.predict(X_test_scaled)

accuracy=accuracy_score(Y_test,Y_pred)

print(f"\nAccuracy: {accuracy*100:.2f}%")   #78% accuracy

#matrice di confusione
cm=confusion_matrix(Y_test,Y_pred,normalize="true")

disp=ConfusionMatrixDisplay(cm, display_labels=le_target.classes_)
disp.plot(cmap="Blues")
plt.title("Matrice di confusione normalizzata")
plt.show()

#L'accuracy è buona, probabilmente modello in overfitting perchè dataset piccolo e semplice
#Nonostante ciò la matrice di confusione evidenzia che il modello fatica e distingue preggio e cìvittore caslinga













