import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.preprocessing import LabelEncoder,StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier,plot_tree
from sklearn.metrics import accuracy_score,ConfusionMatrixDisplay,confusion_matrix


#individuazione del file_path
file_path="./drug200.csv"
if not os.path.exists(file_path):
    raise FileNotFoundError(f"File csv non trovato al percorso {file_path}")

#CARICAMENTO DATASET
df=pd.read_csv(file_path,low_memory=False,na_values=['','-','ND','NA','n/a']) #carico intero dataset tanto poche righe, sostituisco i valori nulli con Nan di pandas

#info generali su Dataset
print(f"\nGrandezza Dataset: {df.shape}")
print(f"Prime righe dataset {df.head()}")
print("Info generali sul dataset caricato:")
df.info()   #dataset abbastanza piccolo (200 righe), noto presenza di valori int e stringhe
target='Drug'   #è una stringa, quindi devo classificare userò modello ad Albero decisionale


#DATA-CLEANING
df.dropna(axis=1,how="all",inplace=True)    #elimino eventuali colonne nulle
df.dropna(axis=0,how="any",subset=target,inplace=True)  #elimino righe con valore Nan su target

#Conteggio valori del target
print(f"\nIl target assume questi valori: {df[target].value_counts()}")
#Il target è abbastanza sbilanciato (DrugX e DrugY molto più pressnti del resto)


print("\nValori assunti da features:")
print("\nSex:")
print(df['Sex'].unique())
print("\nBP:")
print(df['BP'].unique())
print("\nCholesterol:")
print(df['Cholesterol'].unique())
print("\nDrug:")
print(df['Drug'].unique())

#Label Encoder
#Osservo la presenza di valori categorici(stringhe), poichè sklearn alberi non lavora con le stringhe devo convertire in numero
#la giusta manipolazione (per modelli non ad Albero come la regressione lineare) vedrebbe:
    #Ordinal Encoding-->valori ordinali
    #One Hot Encodign--> valori nominali
#Per semplificare l'elaborazione utilizzo il Label Encoder, poichè One Hot Encoding andrebbe ad aggiungermi una colonna per ogni classe correndo rischio di far "scoppiare" il dataset


le_Sex=LabelEncoder()
df['Sex']=le_Sex.fit_transform(df['Sex'])

le_BP=LabelEncoder()
df['BP']=le_BP.fit_transform(df['BP'])

le_Cholesterol=LabelEncoder()
df['Cholesterol']=le_Cholesterol.fit_transform(df['Cholesterol'])

le_target=LabelEncoder()
df['Drug']=le_target.fit_transform(df['Drug'])


print("\nValori assunti da features (post encoding):")
print("\nSex:")
print(df['Sex'].unique())
print("\nBP:")
print(df['BP'].unique())
print("\nCholesterol:")
print(df['Cholesterol'].unique())
print("\nDrug:")
print(df['Drug'].unique())

#Matrice di correlazione
features_num=df.select_dtypes(include=[np.number])
corr=features_num.corr()
sns.heatmap(corr,annot=True,cmap="Blues")
plt.title("Matrice di correlazione lineare")
plt.show()

#Potrei andare a togliere dalla mia prossima elaborazione quelle etichette che provocherebbero rumore (|r|<0.2), ma potrebbero avere 
#relazione nascosti o pattern extra che gli albero sono in grado ci percepire lo stesso, motivo per cui le lascio (non ho dati come medie da eliminare causa di data leakage)


features_num=df.select_dtypes(include=[np.number]).columns.tolist() #lista di nomi delle features numeriche
features_num.remove(target) #avevo convertito target in numero lo tolgo, poichè mi serviràper la matrice X
print(f"Features numeriche: {features_num}")
df.dropna(axis=0,how="all",subset=features_num,inplace=True)    #rimuovo eventuali righe con valori Nan nelle fetures numeriche di mio interesse (poiche gl alberi non le tollerano)
print("Info dataset post Data Cleaning:")
df.info()


#PRE-PROCESSING
X=df[features_num]
Y=df[target]

#Splito il mio insieme di dati in 70% campione d'allenmento e 30% per il test

X_train,X_test,Y_train,Y_test=train_test_split(
    X,Y,
    train_size=0.7,  
    test_size=0.3,
    random_state=10
)

print(f"\nGrandezza campione d'allenamento {X_train.shape}")    #noto stesse colonne/stesse feature, e potrà addestrarsi bene
print(f"\nGrandezza campione di test {X_test.shape}")       #se numero dovesse essere diverso o ho pochi valori di qualche classe, o random state


#Scaling, standardizzo i dati
scaler=StandardScaler()

X_train_scaled=scaler.fit_transform(X_train)
X_test_scaled=scaler.transform(X_test)

#MODELLO--> uso Alberi decisonale poichè devo classificare una categoria
model=DecisionTreeClassifier(random_state=10)   #non metto un limite alla profondità lascio il massimo
model.fit(X_train_scaled,Y_train)

#Visualizzazione del modello
plot_tree(
    model,
    max_depth=3,    #mostro fino profondità 3
    class_names=le_target.classes_,
    feature_names=features_num,
    filled=True

)

plt.title("Albero Decisionale")
plt.show()

#VALUTAZIONE DEL MODELLO

Y_pred=model.predict(X_test_scaled)

accuracy=accuracy_score(Y_test,Y_pred)

print(f"\nAccuracy del modello: {accuracy*100:2f}%")

#Matrice di confusione
cm=confusion_matrix(Y_test,Y_pred,normalize="true")
disp=ConfusionMatrixDisplay(cm,display_labels=le_target.classes_)
disp.plot(cmap="Blues")
plt.title("Matrice di confusione (no max_depth)")
plt.show()

#Il target è abbastanza sbilanciato: DrugX e DrugY sono molto più presenti
#Dalla matrice di confusione noto che confonde DrugA e DrugB per il resto il modello si comporta abbastazna bene come testimoniato dall'accuracy elevata

#e se il modello andasse in Overfitting? Imparando a Memoria le relazioni facendo fatica a generalizzare?--> diminuisco la profondità massima del mio albero


print("\n\n\nMODELLO AD ALBERO CON MAX_DEPTH=3:")

#MODELLO--> max_depth=3
model=DecisionTreeClassifier(random_state=10,max_depth=3)
model.fit(X_train_scaled,Y_train)

#Visualizzazione del modello
plot_tree(
    model,  #mostro profondità masassima che sarà 3
    class_names=le_target.classes_,
    feature_names=features_num,
    filled=True

)

plt.title("Albero Decisionale")
plt.show()

#VALUTAZIONE DEL MODELLO

Y_pred=model.predict(X_test_scaled)

accuracy=accuracy_score(Y_test,Y_pred)

print(f"\nAccuracy del modello con max_depth=3: {accuracy*100:2f}%")

#Matrice di confusione
cm=confusion_matrix(Y_test,Y_pred,normalize="true")
disp=ConfusionMatrixDisplay(cm,display_labels=le_target.classes_)
disp.plot(cmap="Blues")
plt.title("Matrice di confusione con max_depth=3")
plt.show()
#Con profondità minore, cala l'accuracy poichè riduco quello che è il fenomeno dell'overfitting



