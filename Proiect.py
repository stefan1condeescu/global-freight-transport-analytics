import pandas as pd
import numpy as np

df_original = pd.read_csv("date_transport.csv")

# cream fisierul de lucru (cel luat de pe World Bank va ramane nemodificat in structura proiectului)
df_work = df_original.copy()

# eliminarea coloanelor inutile
df_work = df_work.drop(columns=["Series Code"])

# curatarea randurilor de la finalul fisierului World Bank
df_work = df_work.dropna(subset=['Series Name', 'Country Name'])

# inlocuirea .. cu NaN
df_work = df_work.replace("..", np.nan)

# conversia coloanelor cu ani in float
columns_ani = [col for col in df_work.columns if "[" in col]
for col in columns_ani:
    df_work[col] = pd.to_numeric(df_work[col], errors="coerce")

# rezultate dupa pasul 1
print(df_work.head())
print(df_work.info())

# transformarea tabelului din format wide in format long prin folosirea functiei melt
df_work = df_work.melt(id_vars=['Country Name', 'Series Name', 'Country Code'],
                       var_name='An',
                       value_name='Valoare')

# curatam coloana an
df_work['An'] = df_work['An'].str.split(' ').str[0].astype(int)

# pivotarea tabelului
# transformam valorile din Series Name in coloane individuale
df_work = df_work.pivot_table(index=['Country Name', 'An', 'Country Code'],
                              columns='Series Name',
                              values='Valoare').reset_index()

# MERGE (unirea cu regiunile) din fisierul country_metadata
df_meta = pd.read_csv('country_metadata.csv')
# selectam doar coloanele care ne intereseaza
df_meta = df_meta[['Code', 'Region', 'Income Group']]

# unim tabelele folosind codul tarii
df_work = pd.merge(df_work, df_meta, left_on='Country Code', right_on='Code', how='left')

# stergem coloana 'Code' care e duplicat acum si 'Country Code' pt ca nu avem nevoie de ea
df_work = df_work.drop(columns=['Code', "Country Code"])

df_work.to_csv('date_transport_work.csv', index=False)
