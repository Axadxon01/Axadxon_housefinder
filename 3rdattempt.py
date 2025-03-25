
# 🏠 House Finder - Streamlit App (PWA tayyor)

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

# Set page config as the first Streamlit command
st.set_page_config(page_title="🏠 House Finder", layout="wide")

# === CONFIG ===
BASE_PATH = r"C:\Users\admin\capstone project"
DATABASE_NAME = os.path.join(BASE_PATH, "houses.db")
DATASET_PATH = os.path.join(BASE_PATH, "AmesHousing.csv")
MODEL_FILE = os.path.join(BASE_PATH, "house_price_model.pkl")
DEFAULT_COORDINATES = [42.0347, -93.6200]  # Ames, Iowa

# Function to generate approximate coordinates based on neighborhood
def get_neighborhood_coord(neighborhood):
    base_lat = 42.0347  # Center of Ames, Iowa
    base_lng = -93.6200
    random.seed(hash(neighborhood))  # Consistent offset for each neighborhood
    lat_offset = random.uniform(-0.02, 0.02)
    lng_offset = random.uniform(-0.02, 0.02)
    return [base_lat + lat_offset, base_lng + lng_offset]

@st.cache_data
def load_dataframe():
    try:
        return pd.read_csv(DATASET_PATH)
    except FileNotFoundError:
        st.error(f"❌ Dataset file not found at {DATASET_PATH}. Please ensure 'AmesHousing.csv' is in the directory.")
        return pd.DataFrame()  # Return empty DataFrame to avoid crashes

@st.cache_resource
def load_or_train_model():
    try:
        return joblib.load(MODEL_FILE)
    except FileNotFoundError:
        if df.empty:
            st.error("❌ Cannot train model: Dataset is missing.")
            return None
        X = df[['Gr Liv Area', 'Bedroom AbvGr', 'Year Built', 'Garage Cars', 'Lot Area', 'Overall Qual']]
        y = df['SalePrice']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = xgboost.XGBRegressor(n_estimators=200, learning_rate=0.1)
        model.fit(X_train, y_train)
        joblib.dump(model, MODEL_FILE)
        return model

# Load data
df = load_dataframe()

# Initialize database connection in the main thread
if not df.empty:
    conn = sqlite3.connect(DATABASE_NAME)
    df.to_sql("houses", conn, if_exists="replace", index=False)
else:
    conn = None

model = load_or_train_model()

# Filter houses with parameterized queries
def filter_houses(budget, bedrooms, year, garage, lot_size, quality):
    if conn is None:
        return pd.DataFrame()
    query = "SELECT * FROM houses WHERE SalePrice <= ? AND [Year Built] >= ? AND [Lot Area] >= ? AND [Overall Qual] >= ?"
    params = [budget, year, lot_size, quality]
    if bedrooms != -1:
        query += " AND [Bedroom AbvGr] = ?"
        params.append(bedrooms)
    if garage != -1:
        query += " AND [Garage Cars] = ?"
        params.append(garage)
    return pd.read_sql(query, conn, params=tuple(params))

# === UI ===
st.title("🏠 Akhadkhon's House Finder App")

# Sidebar with improved input widgets
st.sidebar.header("🔍 Set Your Filters")
budget = st.sidebar.number_input("💰 Max Budget", min_value=50000, max_value=1000000, value=250000, step=10000)
bedrooms_options = ["Any", 1, 2, 3, 4, 5, 6]
bedrooms_input = st.sidebar.selectbox("🛏 Bedrooms", options=bedrooms_options, index=3)
bedrooms = -1 if bedrooms_input == "Any" else int(bedrooms_input)
year = st.sidebar.number_input("🏗 Min Year Built", min_value=1800, max_value=2023, value=2000, step=1)
garage_options = ["Any", 0, 1, 2, 3, 4]
garage_input = st.sidebar.selectbox("🚗 Garage Spaces", options=garage_options, index=2)
garage = -1 if garage_input == "Any" else int(garage_input)
lot_size = st.sidebar.number_input("📏 Min Lot Area", min_value=0, max_value=100000, value=5000, step=100)
quality = st.sidebar.slider("🔧 Min Overall Quality", min_value=1, max_value=10, value=5)

if st.sidebar.button("🔎 Search Houses"):
    if df.empty or conn is None:
        st.error("❌ Cannot search: Dataset is missing or database connection failed.")
    else:
        result_df = filter_houses(budget, bedrooms, year, garage, lot_size, quality)
        st.success(f"✅ Found {len(result_df)} matching houses")

        if not result_df.empty:
            # Display filtered results
            st.dataframe(result_df[['SalePrice', 'Neighborhood', 'Year Built', 'Bedroom AbvGr', 'Garage Cars', 'Overall Qual']])

            # Map with approximate coordinates
            m = folium.Map(location=DEFAULT_COORDINATES, zoom_start=12)
            marker_cluster = MarkerCluster().add_to(m)
            for _, row in result_df.iterrows():
                lat, lng = get_neighborhood_coord(row['Neighborhood'])
                folium.Marker([lat, lng], popup=f"${row['SalePrice']}").add_to(marker_cluster)
            st_folium(m, width=800, height=500)

            # === Export section ===
            st.subheader("⬇️ Export Results")
            csv = result_df.to_csv(index=False).encode('utf-8')
            st.download_button("⬇️ Download CSV", data=csv, file_name='filtered_houses.csv', mime='text/csv')

            try:
                excel_buffer = io.BytesIO()
                result_df.to_excel(excel_buffer, index=False)
                st.download_button("⬇️ Download Excel", data=excel_buffer.getvalue(),
                                   file_name="filtered_houses.xlsx",
                                   mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
            except ImportError:
                st.warning("❗ Excel eksporti uchun `openpyxl` kerak: `pip install openpyxl`")
        else:
            st.warning("😕 No houses found. Try adjusting your filters.")
else:
    st.info("ℹ️ Use the sidebar to set filters and search.")

    # === Price trend chart ===
    if not df.empty:
        st.subheader("📈 Avg. House Price by Neighborhood")
        avg_prices = df.groupby("Neighborhood")["SalePrice"].mean().sort_values(ascending=False)
        st.bar_chart(avg_prices)

# Clean up
if conn is not None:
    conn.close()

    