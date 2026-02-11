#------------------------CLassificazione su Most Used Platform-------------------------------

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler,LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier,plot_tree
from sklearn.metrics import accuracy_score,ConfusionMatrixDisplay,confusion_matrix

#Caricamento Dataset
file_path="Social.csv"
if not os.path.exists(file_path):
    raise FileNotFoundError(f"File csv non trovato al percorso {file_path}")

df=pd.read_csv(file_path,low_memory=False,na_values=['','-','NA','ND','n/a'],nrows=10000000)

#stampo alcune info per capire con cosa ho a che fare
print(f"\nGrandezza dataset: {df.shape}") #ho 705 righe, dataset piccolo
print(f"\nPrime 5 righe: {df.head()}")
df.info()

#Data cleaning
df.dropna(axis=1,how='all',inplace=True)    #elimino le colonne tutte vuote (qui non ne ho)
df.dropna(axis=0,how="any",subset="Most_Used_Platform",inplace=True)    #elimino righe che hanno almeno Nan nel target
print(f"\nValori che assume target: {df['Most_Used_Platform'].value_counts()}") #è sbilanciato soprattutto tra i più usati e meno, ma è difficle fare raggruppamenti

#si potrebbe fare un raggruppamento sui social meno usati ma andrei a perdere dati che sono gia pochi

#Alberi non lavorano con feature valori nominabili ma devo convertire in numeri (Gender,Academic_level,Country,Affects_Academic_performance,Relationship_status)
#Label encoder

# Label Encoder
# Poiché il modello utilizzato è un albero decisionale, le variabili categoriali
# vengono codificate tramite LabelEncoder per convertire i valori stringa in interi.
# Nei modelli ad albero il LabelEncoder è adatto anche per variabili nominali,
# in quanto l’algoritmo non assume relazioni di ordine o distanza tra i valori.
#
# In generale, per modelli non ad albero, sarebbe preferibile distinguere tra:
# - variabili ordinali → ordinal encoding
# - variabili nominali → one-hot encoding
#
# Il one-hot encoding può aumentare significativamente la dimensionalità del dataset,
# motivo per cui in questo contesto si preferisce l’uso del LabelEncoder.
lencoder=LabelEncoder()

df["Gender"]=lencoder.fit_transform(df['Gender'])
df["Academic_Level"]=lencoder.fit_transform(df["Academic_Level"])
df["Country"]=lencoder.fit_transform(df["Country"])
df["Affects_Academic_Performance"]=lencoder.fit_transform(df["Affects_Academic_Performance"])
df["Relationship_Status"]=lencoder.fit_transform(df["Relationship_Status"])

lencoder_target=LabelEncoder() #encoder specifico per il target che utile dopo nella amtrice di confusione per avere il nome delle categorie
df["Most_Used_Platform"]=lencoder_target.fit_transform(df["Most_Used_Platform"])   #sarebbe corretto fare label encoder anche su target(poi va tolto dalle features poichè diventa numero)
#si potrebbe evitare sul target poichè lo fa gli alberi, ma meglio farlo (il prof non lo fa sul target)
#QUI NON FACCIO MA LA PROCEDURA SAREBBE: ENCODER IN NUMERO SUL TARGET, MATRIC EDI CORRELAZIONE COSI DA TOGLIERE COSE INUTILI E POI lencoder.classe_ nella matrice di confusione in display_labels

df.info()

#faccio una lista delle feature numeriche, andando ad escludere il target (numero dopo label encodr)
features=df.select_dtypes(include=[np.number]).columns.tolist()
df.dropna(axis=0, how='any', subset=features, inplace=True)   #elimino righe con valori Nan nelle fetures
#rimuovo studentiId poichè inutile ai fini dell'addestramento
features.remove("Student_ID")
features.remove("Most_Used_Platform")   #tolgo target, se non lo facessi il modello copierebbe andando abfare prestazioni quasi perfette
print(f"\nLista di features su cui addestrare: {features}")
print("\nInfo post pulizia: ")


df.info()

#PRE-PROCESSING

X=df[features]  #matrice delle feature per addestare e testare
Y=df["Most_Used_Platform"]  #target
#Split

X_train,X_test,Y_train,Y_test=train_test_split(
    X,Y,
    train_size=0.7,
    test_size=0.3,
    random_state=10 
)

print(f"Grandezza campione d'allenamento: {X_train.shape}")
print(f"Grandezza campione di test:{X_test.shape}")

#Scale, standardizzo i dati
scaler=StandardScaler()
X_train_sclaed=scaler.fit_transform(X_train)    
X_test_scaled=scaler.transform(X_test)

#Modello-->albero decisonale classificatore

model=DecisionTreeClassifier(random_state=10)
model.fit(X_train_sclaed,Y_train)       #faccio studare il mio modello

plt.figure(figsize=(20,10))
plot_tree(
    model,
    max_depth=2,#Impostazione di max_depth = 4 (qui 2) per limitare la complessità del modello e ridurre il rischio di overfitting (che impari a memeoria sul train facendo fatica a generalizzare)
    feature_names=features,
    class_names=[str(cls) for cls in sorted(Y.unique())],
    filled=True
)
plt.title("Albero Decisionale")
plt.show()

#Valutazione del modello

Y_pred=model.predict(X_test_scaled)

#accuracy
print(f"Accuracy del modello: {accuracy_score(Y_test,Y_pred):.2f}")

#Matrice di confusione normalizzata
cm=confusion_matrix(Y_test,Y_pred,normalize="true")
disp=ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=lencoder_target.classes_)   #display_labels=lencoder_target.classes_ per far venire fuori i nomi e non numeri nella matrice nelle labels
disp.plot(cmap="Greens")
plt.title("Matrice di confusione normalizzata")
plt.show()



#Si noti che snapacht non viene mai predetto correttamente
#Facebook viene predetto Instagram  il 30% a causa anche dello sbilanciamento dei dati
