import streamlit as st
import pandas as pd
import joblib
import base64
import folium
from streamlit_folium import st_folium
from branca.colormap import linear
import time
from streamlit_autorefresh import st_autorefresh
import requests
from datetime import datetime
import replicate
import os
import numpy as np
from tensorflow.keras.models import load_model
# -------- CONFIGURATION DE LA PAGE ----------
st.set_page_config(page_title="APPLICATION MULTI PAGE", layout="wide")



st.markdown("""
    <style>
    section[data-testid="stSidebar"] label {
        color: white !important;
    }
    </style>
""", unsafe_allow_html=True)


# -------- LOADER ANIMÉ ----------
st.markdown("""
<style>
#overlay {
    position: fixed;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    background-color: #010203;
    display: flex;
    justify-content: center;
    align-items: center;
    z-index: 9999;
    animation: fadeOut 1s ease-in-out 5s forwards;
}

@keyframes fadeOut {
    to {
        opacity: 0;
        visibility: hidden;
    }
}

.glow-text {
    font-size: 5rem;
    font-weight: bold;
    font-family: monospace;
    color: #111;
    display: flex;
    gap: 0.3em;
}

.glow-text span {
    color: #111;
    position: relative;
    display: inline-block;
}

.glow-text span::after {
    content: attr(data-letter);
    position: absolute;
    top: 0;
    left: 0;
    color: white;
    text-shadow: 0 0 8px #0ff, 0 0 20px #0ff;
    opacity: 0;
    animation: lightup 3s ease-in-out forwards;
}

.glow-text span:nth-child(1)::after { animation-delay: 0.3s; }
.glow-text span:nth-child(2)::after { animation-delay: 0.6s; }
.glow-text span:nth-child(3)::after { animation-delay: 0.9s; }
.glow-text span:nth-child(4)::after { animation-delay: 1.2s; }
.glow-text span:nth-child(5)::after { animation-delay: 1.5s; }
.glow-text span:nth-child(6)::after { animation-delay: 1.8s; }
.glow-text span:nth-child(7)::after { animation-delay: 2.1s; }
.glow-text span:nth-child(8)::after { animation-delay: 2.4s; }

@keyframes lightup {
    0% { opacity: 0; }
    50% { opacity: 1; }
    100% { opacity: 0; }
}
</style>

<div id="overlay">
  <div class="glow-text">
    <span data-letter="J">J</span>
    <span data-letter="O">O</span>
    <span data-letter="H">H</span>
    <span data-letter="A">A</span>
    <span data-letter="N">N</span>
    <span data-letter="N">N</span>
    <span data-letter="2.">2.</span>
    <span data-letter="0">0</span>
  </div>
</div>
""", unsafe_allow_html=True)

time.sleep(5)





# MENU DE NAVIGATION VERTICAL
page = st.selectbox("📌 Choisissez une section :", [
    "📊 Données & Prédiction Crédit",
    "🗺️ Carte des Monnaies dans le Monde",
])

# === PAGE 1 : DONNÉES & PRÉDICTION ===
if page == "📊 Données & Prédiction Crédit":
    # -------- IMAGE AVATAR --------
    def get_base64_image(image_path):
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode()

    image_path = "src/JOE.jpg"
    image_data = get_base64_image(image_path)

    # -------- CSS AVATAR ET SIDEBAR --------


    st.markdown(f"""
    <style>
    section[data-testid="stSidebar"] {{
        background-color: #00333d;
        color: white !important;
    }}

    .sidebar-content {{
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        margin-top: 60px;
    }}

    .avatar-container {{
        width: 240px;
        height: 240px;
        border-radius: 50%;
        border: 6px solid white;
        overflow: hidden;
        animation: spin3D 8s ease-in-out infinite;
    }}

    .avatar-container img {{
        width: 100%;
        height: 100%;
        object-fit: cover;
    }}

    .name {{
        margin-top: 20px;
        font-size: 28px;
        color: white;
        font-weight: bold;
        text-align: center;
    }}

    @keyframes spin3D {{
        0%   {{ transform: rotateY(0deg) rotateX(0deg); }}
        40%  {{ transform: rotateY(180deg) rotateX(30deg); }}
        80%  {{ transform: rotateY(360deg) rotateX(0deg); }}
        100% {{ transform: rotateY(360deg) rotateX(0deg); }}
    }}

    .pulse-wrapper {{
        position: fixed;
        top: 33%;
        right: 20px;
        transform: translateY(-50%);
        z-index: 999;
    }}

    .pulse-circle {{
        background-color: #00333d;
        border-radius: 50%;
        width: 226px;
        height: 226px;
        display: flex;
        justify-content: center;
        align-items: center;
        color: white;
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
        position: relative;
        text-align: center;
        flex-direction: column;
    }}

    .pulse-circle .temperature {{
        font-size: 27px;
        margin-bottom: 6px;
    }}

    .pulse-circle .location {{
        font-size: 20px;
        margin-top: 6px;
    }}

    .pulse-circle .date {{
        font-size: 29px;
        margin-bottom: 5px;
    }}

    .pulse-ring {{
        position: absolute;
        top: 0;
        left: 0;
        width: 226px;
        height: 226px;
        border-radius: 50%;
        background-color: #40E0D0;
        opacity: 0.4;
        z-index: 1;
        animation: pulse 2s infinite;
    }}

    @keyframes pulse {{
        0% {{ transform: scale(1); opacity: 0.5; }}
        70% {{ transform: scale(1.6); opacity: 0; }}
        100% {{ transform: scale(1); opacity: 0; }}
    }}
    </style>
    """, unsafe_allow_html=True)

    # -------- AFFICHAGE AVATAR SIDEBAR --------
    with st.sidebar:
        st.markdown(f"""
        <div class="sidebar-content">
            <div class="avatar-container">
                <img src="data:image/png;base64,{image_data}" alt="Avatar">
            </div>
            <div class="name">JOHANN IABD</div>
        </div>
        """, unsafe_allow_html=True)

    # -------- Titre --------
    st.title("PRESENTATION DE MON APPLICATION")

    # --------- DONNÉES METEO ---------
    st_autorefresh(interval=3600000, key="auto_refresh")

    API_KEY = "4997fb908b935664f8b5234881c8145f"
    lat, lon = 3.848, 11.502
    weather_url = f"https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={API_KEY}"

    response = requests.get(weather_url)
    if response.status_code == 200:
        data = response.json()
        temperature = f"{data['main']['temp'] - 273.15:.2f}°C"
        location = data['name']
    else:
        temperature = "N/A"
        location = "Inconnue"

    date_now = datetime.now().strftime("%a. %d %b").upper()

    # -------- AFFICHAGE METEO --------
    st.markdown(f"""
    <div class="pulse-wrapper">
        <div class="pulse-ring"></div>
        <div class="pulse-circle">
            <div class="temperature">{temperature}</div>
            <div class="date">{date_now}</div>
            <div class="location">{location}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # -------- CHARGEMENT DES DONNÉES --------
    @st.cache_data
    def load_data():
        return pd.read_csv("df_final.csv")

    df = load_data()

    # -------- BARRE LATÉRALE DE FILTRES --------
    st.sidebar.header("Filtres")

    months_loan_duration = st.sidebar.slider("Durée du prêt (mois)", int(df["months_loan_duration"].min()), int(df["months_loan_duration"].max()), (int(df["months_loan_duration"].min()), int(df["months_loan_duration"].max())))
    age = st.sidebar.slider("Âge", int(df["age"].min()), int(df["age"].max()), (int(df["age"].min()), int(df["age"].max())))
    existing_credits = st.sidebar.selectbox("Nombre de crédits existants", sorted(df["existing_credits"].unique()))
    default = st.sidebar.selectbox("Défaut de paiement", sorted(df["default"].unique()))
    checking_balance = st.sidebar.multiselect("Solde du compte courant (encodé)", sorted(df["checking_balance_encoded"].unique()), default=sorted(df["checking_balance_encoded"].unique()))
    savings_balance = st.sidebar.multiselect("Solde du compte épargne (encodé)", sorted(df["savings_balance_encoded"].unique()), default=sorted(df["savings_balance_encoded"].unique()))

    # -------- FILTRAGE --------
    filtered_df = df[
        (df["months_loan_duration"].between(*months_loan_duration)) &
        (df["age"].between(*age)) &
        (df["existing_credits"] == existing_credits) &
        (df["default"] == default) &
        (df["checking_balance_encoded"].isin(checking_balance)) &
        (df["savings_balance_encoded"].isin(savings_balance))
    ]

    # -------- AFFICHAGE DES DONNÉES --------
    st.markdown(
            f"""
            <div id="donnees-filtrees"><h3>📋 DATAFRAME CREDIT</h3></div>
            """,
            unsafe_allow_html=True
        )
    st.markdown("### 📊 Données filtrées")
    st.dataframe(filtered_df.style.background_gradient(cmap='Blues'))



    def image_to_base64(path):
        with open(path, "rb") as img_file:
            encoded = base64.b64encode(img_file.read()).decode()
        return f"data:image/jpeg;base64,{encoded}"

    # === Chargement des images ===
    img1 = image_to_base64("G1.PNG")
    img2 = image_to_base64("G2.PNG")
    img3 = image_to_base64("G3.PNG")
    img4 = image_to_base64("G4.PNG")
    img5 = image_to_base64("POWER BI 1.PNG")
    img6 = image_to_base64("POWER B2 1.PNG")
    img7 = image_to_base64("POWER BI 3.PNG")

    # === Style CSS pour galerie ===
    st.markdown(
        """
        <style>
        .image-scroll-container {
            display: flex;
            overflow-x: auto;
            padding: 10px;
            white-space: nowrap;
            width: 100%;
            height: 450px;
            margin-bottom: 20px;
        }
        .image-scroll-container img {
            width: 800px;
            height: 400px;
            margin-right: 15px;
            border-radius: 10px;
            box-shadow: 0 4px 8px rgba(0,0,0,0.2);
            transition: transform 0.2s;
        }
        .image-scroll-container img:hover {
            transform: scale(1.02);
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    # === Bloc 1 ===
    st.markdown(
            f"""
            <div id="seaborn"><h3>📊 Graphiques fait avec Seaborn</h3></div>
            """,
            unsafe_allow_html=True
        )
    if st.checkbox("UNE PRESENTATION DES GRAPHS FAITS SUR VSCODE AVEC SEABORN", value=True):
        st.markdown(
            f"""
            <div class="image-scroll-container">
                <img src="{img1}" alt="Image 1">
                <img src="{img2}" alt="Image 2">
                <img src="{img3}" alt="Image 3">
                <img src="{img4}" alt="Image 4">
            </div>
            """,
            unsafe_allow_html=True
        )

    # === Bloc 2 ===
    st.markdown(
            f"""
            <div id="powerbi"><h3>📈 Graphiques fait avec POWER BI</h3></div>
            """,
            unsafe_allow_html=True
        )
    if st.checkbox("UN SOUVENIR AVEC LES DASHBOARD POWER BI VOUS POUVEZ NAVIGUER HORIZONTALEMENT", value=True):
        st.markdown(
            f"""

            <div class="image-scroll-container">
                <img src="{img5}" alt="Image 5">
                <img src="{img6}" alt="Image 6">
                <img src="{img7}" alt="Image 7">
            </div>
            """,
            unsafe_allow_html=True
        )


   # Chargement des modèles
    model_xgb = joblib.load("JOHN_XGBOOST.pkl")
    model_ann = load_model("JOHANN_ann.h5")

    st.markdown("""
        <div id="titre"><h3>🔮 FORMULAIRE DE PREDICTION</h3></div>
    """, unsafe_allow_html=True)

    st.title("💳 Prédiction de la classe du montant de crédit")
    st.markdown("""
    Ce modèle prédit une **classe de montant (`amount_class`)** en fonction des caractéristiques numériques liées à un crédit.

    ### 🧮 Classes prédictives :
    - **0** : A1 (Très petit) ≤ 1000  
    - **1** : A2 (Petit) 1001 à 2500  
    - **2** : B1 (Moyen) 2501 à 4000  
    - **3** : B2 (Assez grand) 4001 à 6000  
    - **4** : C1 (Grand) 6001 à 10000  
    - **5** : C2 (Très grand) > 10000
    """)

    # Formulaire utilisateur
    with st.form("formulaire"):
        model_choice = st.selectbox("Choisissez le modèle de prédiction :", ["XGBoost", "ANN (JOHANN_ann.h5)"])

        months_loan_duration = st.number_input("Durée du prêt (mois)", 1, 100, 6)
        installment_rate = st.slider("Taux de mensualité", 1, 4, 4)
        residence_history = st.slider("Historique de résidence", 1, 4, 2)
        age = st.number_input("Âge", 18, 100, 30)
        existing_credits = st.number_input("Crédits existants", 0, 10, 1)
        default = st.selectbox("Défaut de paiement ?", [0, 1])
        dependents = st.slider("Personnes à charge", 0, 10, 1)
        checking_balance_encoded = st.selectbox("Solde compte courant (encodé)", [0, 1, 2, 3])
        savings_balance_encoded = st.selectbox("Solde épargne (encodé)", [0, 1, 2, 3, 4])
        employment_length_encoded = st.selectbox("Ancienneté emploi (encodé)", [0, 1, 2, 3, 4])

        submit = st.form_submit_button("🔎 Prédire la classe du montant")

    if submit:
        input_data = pd.DataFrame([[months_loan_duration,
                                    installment_rate,
                                    residence_history,
                                    age,
                                    existing_credits,
                                    default,
                                    dependents,
                                    checking_balance_encoded,
                                    savings_balance_encoded,
                                    employment_length_encoded]],
                                    columns=[
                                        'months_loan_duration',
                                        'installment_rate',
                                        'residence_history',
                                        'age',
                                        'existing_credits',
                                        'default',
                                        'dependents',
                                        'checking_balance_encoded',
                                        'savings_balance_encoded',
                                        'employment_length_encoded'
                                    ])
    
        # Choix du modèle
        if model_choice == "XGBoost":
            prediction = model_xgb.predict(input_data)[0]

        else:  # ANN
            # ANN nécessite souvent une normalisation, assure-toi que ton modèle a été entraîné avec les mêmes valeurs
            prediction_proba = model_ann.predict(input_data)
            prediction = np.argmax(prediction_proba, axis=1)[0]

        # Dictionnaire des classes explicatives
        classes = {
            0: "A1 (Très petit)",
            1: "A2 (Petit)",
            2: "B1 (Moyen)",
            3: "B2 (Assez grand)",
            4: "C1 (Grand)",
            5: "C2 (Très grand)"
        }

        st.success(f"✅ Classe prédite du montant : **{classes[prediction]}** avec le modèle : **{model_choice}**")
 

    

# === PAGE 2 : CARTE INTERACTIVE ===
elif page == "🗺️ Carte des Monnaies dans le Monde":
    

    st.markdown("<h1 style='color:#1f77b4;'>🗺️ Carte Satellite : Capital, Pays et Monnaies</h1>", unsafe_allow_html=True)

    # === Données enrichies avec continents ===
    data = [
        {"country": "France", "capital": "Paris", "currency": "Euro", "continent": "Europe", "lat": 48.8566, "lon": 2.3522, "image_path": "Screenshot.png"},
        {"country": "United States", "capital": "Washington, D.C.", "currency": "US Dollar", "continent": "North America", "lat": 38.8951, "lon": -77.0364, "image_path": "src/Washington.png"},
        {"country": "Japan", "capital": "Tokyo", "currency": "Yen", "continent": "Asia", "lat": 35.6762, "lon": 139.6503, "image_path": "src/Tokyo.png"},
        {"country": "Brazil", "capital": "Brasília", "currency": "Real", "continent": "South America", "lat": -15.7939, "lon": -47.8828, "image_path": "src/Brazilia.png"},
        {"country": "India", "capital": "New Delhi", "currency": "Rupee", "continent": "Asia", "lat": 28.6139, "lon": 77.2090, "image_path": "src/newdelly.png"},
        {"country": "Canada", "capital": "Ottawa", "currency": "Canadian Dollar", "continent": "North America", "lat": 45.4215, "lon": -75.6972, "image_path": "src/Ottawa.png"},
        {"country": "Germany", "capital": "Berlin", "currency": "Euro", "continent": "Europe", "lat": 52.5200, "lon": 13.4050, "image_path": "src/Berlin.png"},
        {"country": "Australia", "capital": "Canberra", "currency": "Australian Dollar", "continent": "Oceania", "lat": -35.2809, "lon": 149.1300, "image_path": "src/cambera.png"},
        {"country": "South Africa", "capital": "Pretoria", "currency": "Rand", "continent": "Africa", "lat": -25.7479, "lon": 28.2293, "image_path": "src/Pretoria.png"},
        {"country": "Mexico", "capital": "Mexico City", "currency": "Peso", "continent": "North America", "lat": 19.4326, "lon": -99.1332, "image_path": "src/Mexico.png"},

        # Vos pays obligatoires
        {"country": "Russia", "capital": "Moscow", "currency": "Ruble", "continent": "Europe", "lat": 55.7558, "lon": 37.6173, "image_path": "src/Moscou.png"},
        {"country": "China", "capital": "Beijing", "currency": "Yuan", "continent": "Asia", "lat": 39.9042, "lon": 116.4074, "image_path": "src/Beijing.png"},
        {"country": "Israel", "capital": "Jerusalem", "currency": "Shekel", "continent": "Asia", "lat": 31.7683, "lon": 35.2137, "image_path": "src/Jérusalem.png"},
        {"country": "South Korea", "capital": "Seoul", "currency": "Won", "continent": "Asia", "lat": 37.5665, "lon": 126.9780, "image_path": "src/Seoul.png"},
        {"country": "Cameroon", "capital": "Yaoundé", "currency": "CFA Franc", "continent": "Africa", "lat": 3.8480, "lon": 11.5021, "image_path": "src/Yaounde.png"},
        {"country": "Côte d'Ivoire", "capital": "Yamoussoukro", "currency": "CFA Franc", "continent": "Africa", "lat": 6.8276, "lon": -5.2893, "image_path": "src/Yamoussoukro.png"},
        {"country": "Nigeria", "capital": "Abuja", "currency": "Naira", "continent": "Africa", "lat": 9.0579, "lon": 7.4951, "image_path": "src/Abuja.png"},
        {"country": "United Kingdom", "capital": "London", "currency": "Pound Sterling", "continent": "Europe", "lat": 51.5074, "lon": -0.1278, "image_path": "src/London.png"},
        {"country": "Sweden", "capital": "Stockholm", "currency": "Swedish Krona", "continent": "Europe", "lat": 59.3293, "lon": 18.0686, "image_path": "src/Suède.png"},
        {"country": "Finland", "capital": "Helsinki", "currency": "Euro", "continent": "Europe", "lat": 60.1695, "lon": 24.9354, "image_path": "src/Helsinki.png"},
        {"country": "Switzerland", "capital": "Bern", "currency": "Swiss Franc", "continent": "Europe", "lat": 46.9481, "lon": 7.4474, "image_path": "src/Bern.png"},
        {"country": "Singapore", "capital": "Singapore", "currency": "Singapore Dollar", "continent": "Asia", "lat": 1.3521, "lon": 103.8198, "image_path": "src/Singapour.png"},

        # Autres pays Afrique, Europe, Asie, Amérique du Nord et Sud
        {"country": "Kenya", "capital": "Nairobi", "currency": "Shilling", "continent": "Africa", "lat": -1.2921, "lon": 36.8219, "image_path": "src/Nairobi.png"},
        {"country": "Ghana", "capital": "Accra", "currency": "Cedi", "continent": "Africa", "lat": 5.6037, "lon": -0.1870, "image_path": "src/Accra.png"},
        {"country": "Algeria", "capital": "Algiers", "currency": "Dinar", "continent": "Africa", "lat": 36.7538, "lon": 3.0588, "image_path": "src/Alger.png"},
        {"country": "Senegal", "capital": "Dakar", "currency": "CFA Franc", "continent": "Africa", "lat": 14.6928, "lon": -17.4467, "image_path": "src/Dakar.png"},
        {"country": "Malaysia", "capital": "Kuala Lumpur", "currency": "Ringgit", "continent": "Asia", "lat": 3.1390, "lon": 101.6869, "image_path": "src/Kuala.png"},
    ]


    df = pd.DataFrame(data)

    # Fonction d'encodage d'image
    def encode_image(image_path):
        with open(image_path, "rb") as img_file:
            img_b64 = base64.b64encode(img_file.read()).decode()
        return f'''
        <div style="width: 100%; height: 200px; overflow: hidden; border-radius: 10px; box-shadow: 0px 2px 6px rgba(0,0,0,0.2); margin-top:10px;">
            <img src="data:image/jpeg;base64,{img_b64}" 
                style="width: 100%; height: 100%; object-fit: cover;"/>
        </div>
        '''

    # === BARRE LATÉRALE DE FILTRES ===
    st.sidebar.markdown("<h2 style='color:white;background-color:#1f77b4;padding:10px;border-radius:10px;'>🎯 Filtres</h2>", unsafe_allow_html=True)

    continent_choice = st.sidebar.selectbox("🌐 Choisissez un continent", ["Tous"] + sorted(df["continent"].unique()))
    filtered_df = df.copy()

    if continent_choice != "Tous":
        filtered_df = filtered_df[filtered_df["continent"] == continent_choice]

    country_choice = st.sidebar.selectbox("🏳️ Choisissez un pays", ["Tous"] + sorted(filtered_df["country"].unique()))

    if country_choice != "Tous":
        filtered_df = filtered_df[filtered_df["country"] == country_choice]

    # === Si aucun pays sélectionné, on affiche le monde entier sinon un seul pays ===
    if not filtered_df.empty:
        map_center = [filtered_df.iloc[0]["lat"], filtered_df.iloc[0]["lon"]]
    else:
        map_center = [0, 0]

    # === Création de la carte ===
    m = folium.Map(location=map_center, zoom_start=3, tiles="Esri.WorldImagery")
    folium.TileLayer("CartoDB PositronOnlyLabels").add_to(m)

    # === Ajout des marqueurs ===
    for _, row in filtered_df.iterrows():
        image_html = encode_image(row["image_path"])
        popup_html = f"""
        <div style="
            width: 300px;
            height: auto;
            padding: 10px;
            border-radius: 15px;
            background-color: white;
            box-shadow: 0px 4px 8px rgba(0,0,0,0.3);
            font-family: Arial, sans-serif;
            text-align: center;
        ">
            <h4 style='margin: 5px 0; color: #2c3e50;'>{row['country']}</h4>
            <p style='margin: 0.2em 0;'>🏛️ <b>Capitale :</b> {row['capital']}</p>
            <p style='margin: 0.2em 0;'>💱 <b>Monnaie :</b> {row['currency']}</p>
            {image_html}
        </div>
        """

        folium.Marker(
            location=[row["lat"], row["lon"]],
            popup=folium.Popup(popup_html, max_width=300),
            tooltip=row["capital"],
            icon=folium.Icon(color="blue", icon="info-sign")
        ).add_to(m)

    # === Affichage dans Streamlit ===
    st_data = st_folium(m, width=1400, height=800)

    # === Affichage complet des données ===
    with st.expander("📋 Voir toutes les données disponibles"):
        st.dataframe(df)


