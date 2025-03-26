import streamlit as st
import pandas as pd
import sqlite3
import folium
from folium.plugins import MarkerCluster
from streamlit_folium import st_folium
import joblib
import xgboost
import numpy as np
from sklearn.model_selection import train_test_split
import os

# === Config ===
st.set_page_config(page_title="🏠 House Finder", layout="wide")
BASE_PATH = ""
DATASET_PATH = os.path.join(BASE_PATH, "AmesHousing.csv")
DATABASE_NAME = os.path.join(BASE_PATH, "houses.db")
MODEL_FILE = os.path.join(BASE_PATH, "house_price_model.pkl")
DEFAULT_COORDINATES = [42.0347, -93.6200]

# === Language Fallback ===
language = st.sidebar.selectbox("🌐 Choose Language", ["English", "O‘zbek", "Русский"])

default_translations = {
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
}

translations = {
    "English": default_translations,
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
    },
    "Русский": {
        "welcome": "👋 Добро пожаловать в House Finder от Akhadkhon!",
        "search_button": "🔎 Поиск домов",
        "filters_header": "🔍 Установите фильтры",
        "result_found": "✅ Найдено {} подходящих домов",
        "no_match": "😕 Ничего не найдено. Показываем пример.",
        "farewell": "👋 До свидания! Ждем снова.",
        "predict_title": "🧠 Прогноз цены на дом",
        "predict_button": "Предсказать цену",
        "predicted_price": "🏷️ Предполагаемая цена: ${:,}",
        "model_missing": "❌ Не удалось загрузить модель."
    }
}

def translate(key):
    return translations.get(language, {}).get(key, default_translations.get(key, key))

# === Welcome Message ===
st.markdown(f"<div style='background-color:#e0f7fa;padding:10px;border-radius:10px;text-align:center;'>{translate('welcome')}</div>", unsafe_allow_html=True)

# === Load Dataset ===
@st.cache_data
def load_data():
    try:
        return pd.read_csv(DATASET_PATH)
    except:
        st.error("❌ Dataset not found.")
        return pd.DataFrame()

df = load_data()

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

# === Filter Sidebar ===
st.sidebar.header(translate("filters_header"))
budget = st.sidebar.number_input(translate("price_label"), value=250000)
bedrooms = st.sidebar.number_input(translate("bedrooms_label"), value=3)
year = st.sidebar.number_input(translate("year_label"), value=2000)
garage = st.sidebar.number_input(translate("garage_label"), value=1)
lot_size = st.sidebar.number_input(translate("lot_label"), value=5000)
quality = st.sidebar.slider(translate("quality_label"), 1, 10, 5)

# === Filter Function ===
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

# === Search Button ===
if st.sidebar.button(translate("search_button")):
    with st.spinner("🔎 Searching..."):
        results = filter_houses()
        if not results.empty:
            st.success(translate("result_found").format(len(results)))
            st.dataframe(results)
        else:
            st.warning(translate("no_match"))
            st.dataframe(df.sample(5))

# === Prediction Section ===
st.markdown("---")
st.subheader(translate("predict_title"))

with st.form("predict_form"):
    col1, col2 = st.columns(2)
    with col1:
        area = st.number_input(translate("gr_liv_area"), 500, 10000, 1500)
        bedroom = st.slider(translate("bedroom_abvgr"), 1, 10, 3)
        year_built = st.number_input(translate("year_built"), 1900, 2025, 2005)
    with col2:
        garage_cars = st.slider(translate("garage_cars"), 0, 4, 1)
        lot = st.number_input(translate("lot_area"), 1000, 30000, 8000)
        qual = st.slider(translate("overall_qual"), 1, 10, 5)
    submitted = st.form_submit_button(translate("predict_button"))
    if submitted:
        if model:
            price = model.predict(np.array([[area, bedroom, year_built, garage_cars, lot, qual]]))[0]
            st.success(translate("predicted_price").format(int(price)))
        else:
            st.error(translate("model_missing"))

# === Goodbye message ===
st.markdown(f"<div style='text-align:center;margin-top:20px;color:gray;font-style:italic;'>{translate('farewell')}</div>", unsafe_allow_html=True)
