import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
from altair import BoxPlot

st.set_page_config(layout="wide")

st.title("Analiza privind transportul de mǎrfuri aerian în raport cu cel feroviar, PIB şi populația")

#citim si vizualizam setul de date
df=pd.read_csv("date_transport_work.csv")
st.header("1. Vizualizarea setului de date")
st.dataframe(df.head(20))

#---------TRATAREA VALORILOR LIPSA------------

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


#------TRATAREA VALORILOR EXTREME----------
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

limita_inf = Q1 - 1.5 * IQR
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


