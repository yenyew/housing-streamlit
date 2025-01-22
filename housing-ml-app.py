import streamlit as st
import pandas as pd
import joblib  # or pickle, depending on how your model is saved

# Title of the app
st.write("""
# House Price Prediction App
This app predicts the price of a house based on user input!
""")

# Sidebar for user inputs
st.sidebar.header('User Input Parameters')

def user_input_features():
    area = st.sidebar.slider('Area (sq ft)', 1000, 20000, 10000)
    bedrooms = st.sidebar.slider('Number of Bedrooms', 1, 5, 3)
    bathrooms = st.sidebar.slider('Number of Bathrooms', 1, 3, 2)
    floors = st.sidebar.slider('Number of Floors', 1, 4, 1)
    main_road = st.sidebar.selectbox('Main Road (1: Yes, 0: No)', [1, 0])
    guestroom = st.sidebar.selectbox('Guestroom (1: Yes, 0: No)', [1, 0])
    basement = st.sidebar.selectbox('Basement (1: Yes, 0: No)', [1, 0])
    parking = st.sidebar.slider('Parking Spaces (0-4)', 0, 4, 2)
    prefarea = st.sidebar.selectbox('Preferred Area (1: Yes, 0: No)', [1, 0])

    data = {
        'area': area,
        'bedrooms': bedrooms,
        'bathrooms': bathrooms,
        'floors': floors,
        'main_road': main_road,
        'guest_room': guestroom,
        'basement': basement,
        'parking': parking,
        'prefarea': prefarea
    }
    features = pd.DataFrame(data, index=[0])
    return features

# Get user input features
df = user_input_features()

# Display user inputs
st.subheader('User Input Parameters')
st.write(df)

# Load your pre-trained model (assuming it's saved as 'Gradient_Boosting_Regressor.pkl')
model = joblib.load('Gradient_Boosting_Regressor.pkl')  # Use your model's path here

# Make a prediction based on user input
prediction = model.predict(df)

# Display the prediction
st.subheader('Predicted House Price')
st.write(f"${prediction[0]:,.2f}")

# Optionally show the model's features or coefficients (if it's a linear model)
if hasattr(model, 'coef_'):
    st.subheader('Model Coefficients')
    coeff_df = pd.DataFrame(model.coef_, df.columns, columns=['Coefficient'])
    st.write(coeff_df)
