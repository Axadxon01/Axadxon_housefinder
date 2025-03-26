import streamlit as st
import pandas as pd
import sqlite3
import folium
from folium.plugins import MarkerCluster
from streamlit_folium import st_folium
import joblib
import xgboost
from sklearn.model_selection import train_test_split
import io
import random
import os

# === Config ===
st.set_page_config(page_title="🏠 House Finder", layout="wide")
BASE_PATH = os.path.dirname(__file__)
DATASET_PATH = os.path.join(BASE_PATH, "AmesHousing.csv")
DATABASE_NAME = os.path.join(BASE_PATH, "houses.db")
MODEL_FILE = os.path.join(BASE_PATH, "house_price_model.pkl")
DEFAULT_COORDINATES = [42.0347, -93.6200]

# === Language Selection ===
language = st.sidebar.selectbox("🌐 Choose Language", ["English", "O‘zbek", "Русский"])
translations = {
    "English": {
        "welcome": "👋 Welcome to Akhadkhon's House Finder!",
        "search_button": "🔎 Search Houses",
        "filters_header": "🔍 Set Your Filters",
        "price_label": "💰 Max Budget",
        "bedrooms_label": "🛏 Bedrooms",
        "year_label": "🏗 Min Year Built",
        "garage_label": "🚗 Garage Spaces",
        "lot_label": "📏 Min Lot Area",
        "quality_label": "🔧 Min Overall Quality",
        "result_found": "✅ Found {} matching houses",
        "no_match": "😕 No exact matches found. Showing sample data.",
        "farewell": "👋 Goodbye! See you soon.",
        "predict_title": "🧠 Predict House Price",
        "gr_liv_area": "📐 Living Area (sq ft)",
        "bedroom_abvgr": "🛏 Bedrooms Above Ground",
        "year_built": "🏗 Year Built",
        "garage_cars": "🚗 Garage Spaces",
        "lot_area": "📏 Lot Area",
        "overall_qual": "🔧 Overall Quality",
        "predict_button": "Predict Price",
        "predicted_price": "🏷️ Estimated Price: ${:,}",
        "model_missing": "❌ Model is not loaded properly."
    },
    "O‘zbek": {
        "welcome": "👋 Akhadkhon'ning Uy Qidiruv Ilovasiga xush kelibsiz!",
        "search_button": "🔎 Uylarni qidiring",
        "filters_header": "🔍 Filtrlarni sozlang",
        "result_found": "✅ {} ta mos uy topildi",
        "no_match": "😕 Mos uy topilmadi. Namuna ko‘rsatilmoqda.",
        "farewell": "👋 Xayr! Yana kutamiz.",
        "predict_title": "🧠 Uy narxini bashorat qiling",
        "predict_button": "Narxni bashorat qilish",
        "predicted_price": "🏷️ Taxminiy narx: ${:,}",
        "model_missing": "❌ Modelni yuklab bo‘lmadi."
    }
}
t = translations[language]

# === Welcome Message ===
st.markdown(
    f"<div style='background-color:#e0f7fa;padding:10px;border-radius:10px;text-align:center;animation: fadeIn 2.5s;'>{t['welcome']}</div><style>@keyframes fadeIn{{0%{{opacity:0;}}100%{{opacity:1;}}}}</style>",
    unsafe_allow_html=True
)

# === Load Data ===
@st.cache_data
def load_dataframe():
    try:
        return pd.read_csv(DATASET_PATH)
    except:
        st.error("❌ Dataset not found.")
        return pd.DataFrame()

df = load_dataframe()

@st.cache_resource
def load_model():
    try:
        return joblib.load(MODEL_FILE)
    except:
        if not df.empty:
            X = df[['Gr Liv Area', 'Bedroom AbvGr', 'Year Built', 'Garage Cars', 'Lot Area', 'Overall Qual']]
            y = df['SalePrice']
            model = xgboost.XGBRegressor(n_estimators=200)
            model.fit(X, y)
            joblib.dump(model, MODEL_FILE)
            return model
        return None

model = load_model()
# === Filter + Search UI ===
st.sidebar.header(t["filters_header"])
budget = st.sidebar.number_input(t["price_label"], value=250000)
bedrooms = st.sidebar.number_input(t["bedrooms_label"], value=3)
year = st.sidebar.number_input(t["year_label"], value=2000)
garage = st.sidebar.number_input(t["garage_label"], value=1)
lot_size = st.sidebar.number_input(t["lot_label"], value=5000)
quality = st.sidebar.slider(t["quality_label"], 1, 10, 5)

def filter_houses():
    if df.empty:
        return pd.DataFrame()
    result = df[
        (df['SalePrice'] <= budget) &
        (df['Bedroom AbvGr'] == bedrooms) &
        (df['Year Built'] >= year) &
        (df['Garage Cars'] == garage) &
        (df['Lot Area'] >= lot_size) &
        (df['Overall Qual'] >= quality)
    ]
    return result

# === SEARCH ACTION ===
if st.sidebar.button(t["search_button"]):
    with st.spinner("🔎 Searching..."):
        results = filter_houses()
        if not results.empty:
            st.success(t["result_found"].format(len(results)))
            st.dataframe(results)
        else:
            st.warning(t["no_match"])
            st.dataframe(df.sample(5))

# === AI PRICE PREDICTION ===
st.markdown("---")
st.subheader(t["predict_title"])

with st.form("predict_form"):
    col1, col2 = st.columns(2)
    with col1:
        area = st.number_input(t["gr_liv_area"], 500, 10000, 1500)
        bedroom = st.slider(t["bedroom_abvgr"], 1, 10, 3)
        year_built = st.number_input(t["year_built"], 1900, 2025, 2005)
    with col2:
        garage_cars = st.slider(t["garage_cars"], 0, 4, 1)
        lot = st.number_input(t["lot_area"], 1000, 30000, 8000)
        qual = st.slider(t["overall_qual"], 1, 10, 5)
    submitted = st.form_submit_button(t["predict_button"])
    if submitted:
        if model:
            price = model.predict([[area, bedroom, year_built, garage_cars, lot, qual]])[0]
            st.success(t["predicted_price"].format(int(price)))
        else:
            st.error(t["model_missing"])

# === Farewell Message ===
st.markdown(
    f"<div style='text-align:center;margin-top:20px;color:gray;font-style:italic;animation: fadeIn 2.5s;'>{t['farewell']}</div>",
    unsafe_allow_html=True
)
