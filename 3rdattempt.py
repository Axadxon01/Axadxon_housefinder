
import streamlit as st
import pandas as pd
import sqlite3
import joblib
import folium
from folium.plugins import MarkerCluster
from streamlit_folium import st_folium
import yaml
from yaml.loader import SafeLoader
import streamlit_authenticator as stauth
from datetime import datetime

# Page setup
st.set_page_config("Ultimate House Finder", layout="wide")
st.title("🏠 Ultimate House Finder by Akhadkhon")

# Load Auth
with open("config.yaml") as file:
    config = yaml.load(file, Loader=SafeLoader)
authenticator = stauth.Authenticate(
    config["credentials"], config["cookie"]["name"], config["cookie"]["key"],
    config["cookie"]["expiry_days"], config["preauthorized"]
)
name, auth_status, username = authenticator.login("Login", "main")

if auth_status:
    authenticator.logout("Logout", "sidebar")
    st.sidebar.success(f"Welcome, {name}!")

    conn = sqlite3.connect("houses.db")
    df = pd.read_sql("SELECT * FROM houses", conn)

    # House filters
    st.sidebar.header("Filter Houses")
    budget = st.sidebar.slider("Max Budget", 50000, 1000000, 250000, 10000)
    year = st.sidebar.slider("Min Year Built", 1900, 2025, 2000)
    quality = st.sidebar.slider("Overall Qual", 1, 10, 5)

    filtered_df = df[(df["SalePrice"] <= budget) & (df["Year Built"] >= year) & (df["Overall Qual"] >= quality)]

    # Main content
    st.subheader("🔍 Filtered Houses")
    st.dataframe(filtered_df)

    st.subheader("🗺 Real Map View (Google-like)")
    center = [42.0347, -93.62]
    m = folium.Map(location=center, zoom_start=12)
    marker_cluster = MarkerCluster().add_to(m)
    for _, row in filtered_df.iterrows():
        lat = row.get("Latitude", center[0]) + 0.001
        lon = row.get("Longitude", center[1]) + 0.001
        folium.Marker([lat, lon], popup=f"${row['SalePrice']} - {row['Neighborhood']}").add_to(marker_cluster)
    st_folium(m, width=1000)

    st.subheader("📊 Admin Panel")
    st.markdown("Average price by neighborhood")
    avg_price = df.groupby("Neighborhood")["SalePrice"].mean().sort_values(ascending=False)
    st.bar_chart(avg_price)

    st.markdown("📈 Newest 5 Listings")
    latest = df.sort_values("Year Built", ascending=False).head(5)
    st.dataframe(latest)

    st.subheader("➕ Submit New House Listing")
    with st.form("new_listing", clear_on_submit=True):
        price = st.number_input("Sale Price", 50000, 1000000, 250000)
        area = st.number_input("Gr Liv Area", 500, 5000, 1500)
        beds = st.selectbox("Bedrooms", [1, 2, 3, 4, 5])
        year_built = st.number_input("Year Built", 1900, 2025, 2005)
        garage = st.selectbox("Garage Cars", [0, 1, 2, 3])
        lot = st.number_input("Lot Area", 1000, 20000, 7000)
        qual = st.slider("Overall Qual", 1, 10, 6)
        nbhd = st.text_input("Neighborhood", "CollgCr")
        submit = st.form_submit_button("Submit")
        if submit:
            conn.execute("""INSERT INTO houses (SalePrice, [Gr Liv Area], [Bedroom AbvGr], [Year Built],
                [Garage Cars], [Lot Area], [Overall Qual], Neighborhood)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)""", (price, area, beds, year_built, garage, lot, qual, nbhd))
            conn.commit()
            st.success("✅ Listing added!")

else:
    st.warning("Please login to access the full app.")
