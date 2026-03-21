import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
from altair import BoxPlot
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler, MinMaxScaler


st.set_page_config(layout="wide")

st.title("Analiza privind transportul de mǎrfuri aerian în raport cu cel feroviar, PIB şi populația")

#citim si vizualizam setul de date
df=pd.read_csv("date_transport_work.csv")
st.header("1. Vizualizarea setului de date")
st.dataframe(df.head(20))

### TRATAREA VALORILOR LIPSA

#gasim coloanele cu valori lipsa si afisam numarul valorilor lipsa din fiecare
st.header("2. Identificarea valorilor lipsa")
missing_values=df.isnull().sum()

#pastram doar coloanele care au valori lipsa
missing_values=missing_values[missing_values > 0]
if missing_values.empty:
    st.success("Nu exista valori lipsa in setul de date.")
else:
    st.warning(f"Au fost gasite {len(missing_values)} coloane cu valori lipsa.")

    st.dataframe(
        pd.DataFrame({
            "Coloana":missing_values.index,
            "Numar valori lipsa":missing_values.values,
            "Procent": (missing_values.values/len(df)*100).round(2)
        })
    )

#vizualizam randurile unde lipsesc valori
st.subheader("Vizualizare randuri cu valori lipsa")
if missing_values.empty:
    st.info("Nu exista coloane cu valori lipsa pentru analiza.")
else:
    col_missing = list(missing_values.index)

    col_selected = st.selectbox(
        "Alege o coloana pentru vizualizare:",
        col_missing,
    )

    st.write("Randurile unde lipsesc valori pentru coloana selectata:")
    st.dataframe(df[df[col_selected].isnull()])

st.header("Tratarea valorilor lipsa")
#modurile de tratare disponibile in functie de tipul de variabile:
#pentru variabile numerice: media, mediana, forward fill, backward fill
#pentru variabile categoriale: moda / "Necunoscut"
#pentru putine valori lipsa: se pot sterge randurile afectate

st.subheader("Moduri de tratare disponibile in functie de tipul variabilei")

tip_selectat = st.radio(
    "Alege tipul variabilei:",
    ["Numerice", "Categoriale"]
)

if tip_selectat == "Numerice":
    st.write("Pentru variabile numerice, putem folosi urmatoarele metode:")
    st.markdown("""
    - **Media** – potrivita cand valorile sunt distribuite relativ uniform  
    - **Mediana** – potrivita cand exista valori extreme  
    - **Forward fill** – completeaza cu valoarea anterioara  
    - **Backward fill** – completeaza cu valoarea urmatoare  
    - **Stergerea randurilor** – doar daca numarul valorilor lipsa este foarte mic  
    """)
else:
    st.write("Pentru variabile categoriale, putem folosi urmatoarele metode:")
    st.markdown("""
    - **Moda** – completeaza cu cea mai frecventa valoare  
    - **'Necunoscut'** – util cand nu vrem sa presupunem o categorie  
    - **Stergerea randurilor** – doar daca lipsesc foarte putine valori  
    """)

#aplicam modul de tratare eficient in cazul nostru:
# cream o copie a setului de date
df_tratat = df.copy()

# ordonam datele dupa tara si an
df_tratat = df_tratat.sort_values(by=["Country Name", "An"])

# coloanele numerice cu valori lipsa
coloane_de_tratat = [
    "Air transport, freight (million ton-km)",
    "Railways, goods transported (million ton-km)"
]

# aplicam forward fill si backward fill in cadrul fiecarei tari
for col in coloane_de_tratat:
    df_tratat[col] = df_tratat.groupby("Country Name")[col].ffill()
    df_tratat[col] = df_tratat.groupby("Country Name")[col].bfill()

# daca mai raman valori lipsa, completam cu mediana coloanei
for col in coloane_de_tratat:
    df_tratat[col] = df_tratat[col].fillna(df_tratat[col].median())

#verificam setul de date dupa tratare
st.subheader("Verificarea valorilor lipsa dupa tratare")

missing_after = df_tratat.isnull().sum()
missing_after = missing_after[missing_after > 0]

if missing_after.empty:
    st.success("Toate valorile lipsa au fost tratate cu succes.")
else:
    st.warning("Mai exista valori lipsa in urmatoarele coloane:")
    st.dataframe(
        pd.DataFrame({
            "Coloana": missing_after.index,
            "Numar valori lipsa": missing_after.values
        })
    )

st.subheader("Setul de date dupa tratarea valorilor lipsa")
# verificam daca mai exista valori lipsa
missing_after = df_tratat.isnull().sum()
missing_after = missing_after[missing_after > 0]

if missing_after.empty:
    st.success("Nu mai exista valori lipsa in setul de date tratat.")
else:
    st.warning("Mai exista valori lipsa in setul de date tratat.")

    st.dataframe(
        pd.DataFrame({
            "Coloana": missing_after.index,
            "Numar valori lipsa": missing_after.values,
            "Procent": (missing_after.values / len(df_tratat) * 100).round(2)
        })
    )

with st.expander("Afiseaza setul de date dupa tratare"):
    st.dataframe(df_tratat)

#putem salva setul de date prelucrat:
csv = df_tratat.to_csv(index=False).encode("utf-8")

st.download_button(
    label="Descarca setul de date tratat",
    data=csv,
    file_name="date_transport_curatat.csv",
    mime="text/csv"
)


### TRATAREA VALORILOR EXTREME

st.header("3. Identificarea valorilor extreme")

# coloanele numerice relevante pentru analiza
coloane_outlieri = [
    "Air transport, freight (million ton-km)",
    "Railways, goods transported (million ton-km)",
    "GDP (current US$)",
    "Population, total"
]

st.subheader("Analiza outlierilor cu metoda IQR")
#pastram doar coloanele care exista efectiv in setul de date
coloane_outlieri = [col for col in coloane_outlieri if col in df_tratat.columns]

col_outlier = st.selectbox(
    "Alege coloana pentru analiza outlierilor:",
    coloane_outlieri
)

#calcul IQR
Q1 = df_tratat[col_outlier].quantile(0.25)
Q3 = df_tratat[col_outlier].quantile(0.75)
IQR = Q3 - Q1

limita_inf = max(0, Q1 - 1.5 * IQR)
limita_sup = Q3 + 1.5 * IQR

#identificare outlieri
outlieri = df_tratat[(df_tratat[col_outlier] < limita_inf) | (df_tratat[col_outlier] > limita_sup) ]

nr_outlieri = len(outlieri)
procent_outlieri = round(nr_outlieri / len(df_tratat) * 100, 2)

st.subheader("Rezultatele identificarii outlierilor")

col1, col2 = st.columns(2)

with col1:
    st.metric("Limita inferioara", f"{limita_inf:,.0f}")
    st.metric("Numar outlieri", nr_outlieri)

with col2:
    st.metric("Limita superioara", f"{limita_sup:,.0f}")
    st.metric("Procent outlieri", f"{procent_outlieri:.2f}%")

st.write("Randurile identificate ca outlieri:")
st.dataframe(outlieri)

#cream grafice de tip boxplot si histogram pentru o vizualizare mai clara a valorilor extreme

st.subheader("Vizualizarea outlierilor prin intermediul graficelor BoxPlot si Histogram")

col1, col2 = st.columns(2)

with col1:
    fig_box = px.box(
        df_tratat,
        y=col_outlier,
        title=f"Boxplot pentru {col_outlier} (log scale)"
    )

    fig_box.update_yaxes(type="log")

    st.plotly_chart(fig_box, use_container_width=True)
with col2:
    fig_hist = px.histogram(
        df_tratat,
        x=col_outlier,
        nbins=30,
        title=f"Histograma pentru {col_outlier}"
    )
    fig_hist.update_layout(height=400)
    st.plotly_chart(fig_hist, use_container_width=True)


### CODIFICAREA VARIABILELOR CATEGORIALE

st.header("4. Codificarea variabilelor categoriale")

#identificam coloanele categoriale
coloane_categorice = df_tratat.select_dtypes(include=["object"]).columns.tolist()

#excludem coloanele pe care nu vrem sa le codificam
#Country Name este un identificator, nu o variabila explicativa, astfel codificarea acesteia nu ar aduce informatie relevanta pt modele sau analiza statistica
coloane_excluse = ["Country Name"]
coloane_categorice = [col for col in coloane_categorice if col not in coloane_excluse]

st.subheader("Coloane categoriale disponibile")
if coloane_categorice:
    st.write(coloane_categorice)
else:
    st.info("Nu exista coloane categoriale disponibile.")

#alegerea coloanelor
coloane_selectate = st.multiselect(
    "Alege coloanele pentru codificare:",
    coloane_categorice,
    default=coloane_categorice
)

#alegerea metodei
metoda_codificare = st.selectbox(
    "Alege metoda de codificare:",
    ["One-Hot Encoding", "Label Encoding"]
)

from sklearn.preprocessing import LabelEncoder

if coloane_selectate:
    df_codificat = df_tratat.copy()

    if metoda_codificare == "One-Hot Encoding":
        df_codificat = pd.get_dummies(
            df_codificat,
            columns=coloane_selectate,
            drop_first=False
        )

    elif metoda_codificare == "Label Encoding":
        le = LabelEncoder()
        for col in coloane_selectate:
            df_codificat[col] = le.fit_transform(df_codificat[col].astype(str))

    #rezultate
    st.subheader("Rezultatul codificarii")

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Nr. coloane initial", df_tratat.shape[1])

    with col2:
        st.metric("Nr. coloane dupa codificare", df_codificat.shape[1])

    st.write("Primele randuri:")
    st.dataframe(df_codificat.head(20))

    #descarcare
    csv_codificat = df_codificat.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="Descarca setul de date codificat",
        data=csv_codificat,
        file_name="date_transport_codificat.csv",
        mime="text/csv"
    )

else:
    st.warning("Selecteaza cel putin o coloana.")

### SCALAREA DATELOR

st.header("5. Metode de scalare")

coloane_numerice_scalare = [
    "Air transport, freight (million ton-km)",
    "Railways, goods transported (million ton-km)",
    "GDP (current US$)",
    "Population, total"
]

coloane_numerice_scalare = [col for col in coloane_numerice_scalare if col in df_tratat.columns]

st.subheader("Coloane numerice disponibile pentru scalare")
st.write(", ".join(coloane_numerice_scalare))

coloane_selectate_scalare = st.multiselect(
    "Alege coloanele pe care vrei sa le scalezi:",
    coloane_numerice_scalare,
    default=coloane_numerice_scalare
)

metoda_scalare = st.selectbox(
    "Alege metoda de scalare:",
    ["StandardScaler", "MinMaxScaler"]
)

if coloane_selectate_scalare:
    df_scalat = df_tratat.copy()

    if metoda_scalare == "StandardScaler":
        scaler = StandardScaler()
    else:
        scaler = MinMaxScaler()

    df_scalat[coloane_selectate_scalare] = scaler.fit_transform(df_scalat[coloane_selectate_scalare])

    st.subheader("Rezultatele scalarii")

    col1, col2 = st.columns(2)

    with col1:
        st.write("Date inainte de scalare:")
        st.dataframe(df_tratat[coloane_selectate_scalare].head(10))

    with col2:
        st.write("Date dupa scalare:")
        st.dataframe(df_scalat[coloane_selectate_scalare].head(10))

else:
    st.warning("Selecteaza cel putin o coloana pentru scalare.")

### PRELUCRARI STATISTICE, GRUPARI SI AGREGARI DE DATE

st.header("6. Prelucrari statistice, grupari si agregari pe date")

#1.prelucrari statistice: count, mean, std, min, max, mediana
st.subheader("A. Statistici descriptive generale")
coloane_numerice = [
    "Air transport, freight (million ton-km)",
    "Railways, goods transported (million ton-km)",
    "GDP (current US$)",
    "Population, total"
]

#pastram doar coloanele care exista in setul de date
coloane_numerice = [col for col in coloane_numerice if col in df_tratat.columns]

if coloane_numerice:
    statistici = df_tratat[coloane_numerice].describe().T.round(2)
    st.dataframe(statistici)
else:
    st.info("Nu exista coloane numerice disponibile pentru analiza.")

#2.grupare si agregare pe date - pe an
st.subheader("B. Evolutia in timp a transportului de marfuri")
if "An" in df_tratat.columns:
    agregare_an = df_tratat.groupby("An").agg({
        "Air transport, freight (million ton-km)": "mean",
        "Railways, goods transported (million ton-km)": "mean",
        "GDP (current US$)": "mean",
        "Population, total": "mean"
    }).round(2)

    st.write("Valorile medii anuale pentru principalii indicatori:")
    st.dataframe(agregare_an)

    st.line_chart(
        agregare_an[[
            "Air transport, freight (million ton-km)",
            "Railways, goods transported (million ton-km)"
        ]]
    )
else:
    st.info("Nu exista coloana 'An' in setul de date.")

#3.grupare si agregare pe date - pe region
st.subheader("C. Comparatia indicatorilor pe regiuni geografice")

if "Region" in df_tratat.columns:
    agregare_regiune = df_tratat.groupby("Region").agg({
        "Air transport, freight (million ton-km)": "mean",
        "Railways, goods transported (million ton-km)": "mean",
        "GDP (current US$)": "mean",
        "Population, total": "mean"
    }).round(2)

    st.write("Valorile medii ale principalilor indicatori, pe regiuni:")
    st.dataframe(agregare_regiune)

    st.write("Comparatia dintre regiunile geografice pentru transportul aerian si feroviar:")
    st.bar_chart(
        agregare_regiune[[
            "Air transport, freight (million ton-km)",
            "Railways, goods transported (million ton-km)"
        ]]
    )
else:
    st.info("Nu exista coloana 'Region' in setul de date.")

### FILTRARE SI SORTARE

st.header("7. Filtrare si sortare dupa transportul aerian de marfa")

df_filtrat = df_tratat.copy()

#valori min si max
min_val = float(df_filtrat["Air transport, freight (million ton-km)"].min())
max_val = float(df_filtrat["Air transport, freight (million ton-km)"].max())

#slider
interval = st.slider(
    "Selecteaza intervalul transportului aerian (million ton-km):",
    min_value=min_val,
    max_value=max_val,
    value=(1.0, max_val)
)

#filtrare
df_filtrat = df_filtrat[
    (df_filtrat["Air transport, freight (million ton-km)"] >= interval[0]) &
    (df_filtrat["Air transport, freight (million ton-km)"] <= interval[1])
]

#sortare
ordine = st.radio(
    "Alege ordinea:",
    ["Descrescator", "Crescator"]
)

if ordine == "Descrescator":
    df_filtrat = df_filtrat.sort_values(
        by="Air transport, freight (million ton-km)",
        ascending=False
    )
else:
    df_filtrat = df_filtrat.sort_values(
        by="Air transport, freight (million ton-km)",
        ascending=True
    )

#afisare
st.write(f"Numar observatii: {len(df_filtrat)}")
st.dataframe(df_filtrat.head(20))

### FUNCTII DE GRUP SI ANALIZA COTELOR DE PIATA

st.header("8. Analiza cotelor de piata pe regiuni (Functii de grup)")

# cream o copie pentru analiza
df_group = df_tratat.copy()

# calculam totalul pe regiune si an folosind transform('sum')
df_group['Total_Regiune_An'] = df_group.groupby(['Region', 'An'])['Air transport, freight (million ton-km)'].transform('sum')

# calculam ponderea % fiecarei tari in totalul regiunii sale
df_group['Pondere_in_Regiune'] = (df_group['Air transport, freight (million ton-km)'] / df_group['Total_Regiune_An'] * 100).round(2)

an_selectat = st.selectbox("Selecteaza anul pentru analiza cotelor de piata:", sorted(df_group['An'].unique(), reverse=True))

df_an = df_group[df_group['An'] == an_selectat].sort_values(by='Pondere_in_Regiune', ascending=False)

st.write(f"Topul tarilor cu cea mai mare influenta in transportul aerian regional in anul {an_selectat}:")
st.dataframe(df_an[['Country Name', 'Region', 'Air transport, freight (million ton-km)', 'Pondere_in_Regiune']].head(10))

fig_share = px.bar(df_an.head(15), x='Country Name', y='Pondere_in_Regiune', color='Region',
                   title="Cota de piata a tarilor in cadrul regiunii lor (%)")
st.plotly_chart(fig_share, use_container_width=True)

