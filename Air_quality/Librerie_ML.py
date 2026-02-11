#Esploreremo tre librerie fondamentali per lavorare nel campo del **Machine Learning con Python**:

#- `pandas`: per la gestione dei dati
#- `scikit-learn`: per il caricamento del dataset e il calcolo delle metriche
#- `matplotlib`: per la visualizzazione grafica

#Utilizzeremo i dati dell'Indice di Qualità dell'Aria di Taiwan 2016~2024** per effettuare la prima analisi del dataset.

#Cosa ho nel dataset?
## 1. Importing libraries and loading the airquality dataset  
#**Column description** in `air_quality.csv` (25 in total):

#- date: date and time of the measurement (e.g. “2024-08-31 23:00”)  
#- sitename, county, siteid: details about the monitoring station  
#- aqi: Air Quality Index, continuous target for possible regression  
#- pollutant: main pollutant
#- status: qualitative classification (Good→5 … Hazardous→0)  
#- so2, co, o3, o3_8hr, co_8hr, no2, nox, no, pm10, pm2.5: instant and 8-hour average pollutants' concentrations  
#- pm10_avg, pm2.5_avg, so2_avg: moving average of the main pollutants  
#- windspeed, winddirec: wind speed and direction  
#- longitude, latitude: geographical coordinates  



### 1. Importazione delle librerie e caricamento del dataset sulla qualità dell'aria
#**Descrizione delle colonne** in `air_quality.csv` (25 in totale):

#- data: data e ora della misurazione (es. "2024-08-31 23:00")
#- nome del sito, contea, siteid: dettagli sulla stazione di monitoraggio
#- aqi: indice di qualità dell'aria, target continuo per possibile regressione
#- inquinante: inquinante principale
#- stato: classificazione qualitativa (Buono→5 … Pericoloso→0)
#- so2, co, o3, o3_8hr, co_8hr, no2, nox, no, pm10, pm2.5: concentrazioni medie istantanee e su 8 ore degli inquinanti
#- pm10_avg, pm2.5_avg, so2_avg: media mobile dei principali inquinanti
#- velocità del vento, direzione del vento: velocità e direzione del vento
#- longitudine latitudine: coordinate geografiche


# EXERCISE

#1. Load the csv file into a DataFrame. Print the shape of the resulting DataFrame.
#2. From the resulting DataFrame, remove the columns where the missing values are more than the 5% of the total rows. Print the shape of the resulting DataFrame.
#3. From the resulting DataFrame count and remove all the rows where we have at least 3 columns with a missing value. Print the shape of the resulting DataFrame.
#4. Write the statistics of the resulting DataFrame.
#5. Show the heatmap of the missing values using Seaborn.
#6. Plot the aqi vs date for the first 365 rows for the site name 'Hukou'


# ESERCIZII (ita)

#1. Caricare il file csv in un DataFrame. 
#2. Dal DataFrame risultante, rimuovere le colonne in cui i valori mancanti sono superiori al 5% delle righe totali. Stampare la forma del DataFrame risultante.
#3. Dal DataFrame risultante, contare e rimuovere tutte le righe in cui sono presenti almeno 3 colonne con un valore mancante. Stampare la forma del DataFrame risultante.
#4. Scrivere le statistiche del DataFrame risultante.
#5. Mostrare la mappa di calore dei valori mancanti utilizzando Seaborn.
#6. Rappresentare graficamente l'aqi in base alla data per le prime 365 righe per il nome del sito 'Hukou'

import os  # per gestire file e cartelle
import pandas as pd  # per manipolare dati in DataFrame
import matplotlib.pyplot as plt  # per creare grafici base
import seaborn as sns  # per grafici avanzati e heatmap
from sklearn.model_selection import train_test_split  # per dividere dati in train/test
from sklearn.preprocessing import StandardScaler  # per normalizzare/scalare dati
from sklearn.tree import DecisionTreeClassifier, plot_tree  # albero decisionale e visualizzazione
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay  # metriche ML

#1 Caricare csv in DataFrame
file_path="data/air_quality.csv"  # percorso del file CSV
if not os.path.exists(file_path):
    raise FileNotFoundError(f"Il file csv non esiste: {file_path}")  # errore chiaro se il file non esiste

# leggere il CSV in un DataFrame
df = pd.read_csv(
    file_path, 
    low_memory=False,  # legge tutto insieme per tipi di colonna coerenti
    na_values=['-', 'NA', 'ND', 'n/a', ''],  # valori da considerare NaN
    nrows=1000000  # legge solo le prime 1 milione di righe
)

#2 Percentuale di valori mancanti e dimensioni dataset
print("\nPercentuale di valori mancanti per colonna (prima di rimuovere cose)")
missing_perc = df.isnull().mean() * 100  # calcola % valori NaN per colonna
print(missing_perc)
print(df.shape)  # dimensioni DataFrame (righe, colonne)

#3 Dropping colonne con troppi valori mancanti
print("\n\nDropping colonne con valori mancanti")
df = df.loc[:, missing_perc < 5]  # seleziona solo colonne con <5% valori mancanti
print(df.shape)  # dimensioni aggiornate

#4 Contare righe con almeno 3 valori mancanti
count_three_missing = (df.isna().sum(axis=1) >= 3).sum()  # conta righe con >=3 NaN
print("\nNumero di righe con più di 3 missing values:", count_three_missing)

# Eliminare righe con 3 o più valori mancanti
print("\n\n\nDropping rows... ")
df = df[df.isna().sum(axis=1) < 3]  # mantiene solo righe con <3 NaN
print(df.shape)  # dimensioni aggiornate

#5 Info e statistiche descrittive
print("\n\nDataSet info:")
print(df.info())  # informazioni generali (tipi, NaN)

print("\nStatistiche Descrittive:")
print(df.describe(include='all'))  # statistiche base per tutte le colonne

print("\nPercentage of missing values per column (after dropping):")
print(df.isnull().mean() * 100)  # % di valori mancanti aggiornati

#6 Heatmap dei valori mancanti
plt.figure(figsize=(12,6))
sns.heatmap(df.isnull(), cbar=False, cmap='viridis')  # mappa dei NaN
plt.title('Missing Values Map')
plt.show()

#7 Plot AQI nel tempo per una stazione specifica
df_plot = df[df['sitename'] == 'Hukou']  # seleziona stazione Hukou
df_plot = df_plot.head(365)  # solo primi 365 giorni

plt.figure(figsize=(20, 5))
plt.plot(df_plot['date'], df_plot['aqi'], marker='o', linestyle='-', linewidth=2)  # trend AQI
plt.title("AQI Trend Over Time", fontsize=14)
plt.xlabel("Date", fontsize=12)
plt.ylabel("AQI", fontsize=12)
plt.xticks(rotation=45, fontsize=8)  # ruota date per leggibilità
plt.grid(True)
plt.show()







