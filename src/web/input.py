import streamlit as st
import numpy as np
import pandas as pd
import pickle
import base64
import datetime as date

# Load the trained model

model = pickle.load(open('/home/karan/Templates/Crop-Recommendation-system/src/Notebooks/Crop_Yield_Prediction.pkl','rb'))

preprocesser = pickle.load(open('/home/karan/Templates/Crop-Recommendation-system/src/Notebooks/preprocessor.pkl','rb'))

items_list =['Maize','Potatoes','Rice, paddy','Sorghum','Soybeans','Wheat','Cassava','Sweet potatoes','Plantains and others','Yams']


def prediction(Year, average_rain_fall_mm_per_year, pesticides_tonnes, avg_temp, Area, Item):
    features = np.array([[Year, average_rain_fall_mm_per_year, pesticides_tonnes, avg_temp, Area, Item]], dtype=object)
    transformed_features = preprocesser.transform(features)
    predicted_yield = model.predict(transformed_features).reshape(1, -1)
    return predicted_yield[0]


# Streamlit app
def main():

    st.title("Crop Recommendation System")

    # Get the date input from the user with a specific label and default value
    selected_date = st.date_input("Select a date", min_value=date.now(), max_value=date.date(2025,12,31))

    # Clamp the year between 2023 and 2025
    clamped_date = selected_date.replace(year=max(2023, min(selected_date.year, 2025)))

    # Extract month, day, and year into separate variables
    selected_month = clamped_date.month
    selected_day = clamped_date.day
    selected_year = clamped_date.year

    print(selected_month, selected_day, selected_year)

    average_rain_fall_mm_per_year = st.number_input("Average Rainfall (mm per year)", value=1485.0)
    pesticides_tonnes = st.number_input("Pesticides (tonnes)", value=121.0)
    avg_temp = st.number_input("Average Temperature", value=16.37)
    Area = st.text_input("Area", "India")
    # Item = st.text_input("Item", "Maize")
    Item = st.selectbox("Select Crop", items_list)

    
    # Load the CSV file
    df = pd.read_csv('/home/karan/Templates/Crop-Recommendation-system/src/web/location.csv')

    # Get unique states from the DataFrame
    states = df['State'].unique()

    # Display state selector
    selected_state = st.selectbox("Select a State", states)

    # Filter districts based on the selected state
    filtered_districts = df[df['State'] == selected_state]['District'].unique()

    # Display district selector
    selected_district = st.selectbox("Select a District", filtered_districts)

    # Get coordinates based on the selected state and district
    selected_location = df[(df['State'] == selected_state) & (df['District'] == selected_district)]

    # Display the selected state, district, and coordinates
    if not selected_location.empty:
        latitude = selected_location['Latitude'].values[0]
        longitude = selected_location['Longitude'].values[0]
    else:
        st.warning("Please select a state and district.")






    # Make prediction
    if st.button("Predict Yield"):
        Area = "India"
        print(selected_year,average_rain_fall_mm_per_year,pesticides_tonnes,avg_temp,Area,Item)
        result = prediction(selected_year, average_rain_fall_mm_per_year, pesticides_tonnes, avg_temp, Area, Item)
        st.success(f"Predicted Yield: {result[0]}")

if __name__ == "__main__":
    main()
