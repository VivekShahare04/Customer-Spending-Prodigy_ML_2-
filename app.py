import streamlit as st
import pandas as pd
import pickle
import sklearn

# Load the model from the pickle file
@st.cache(allow_output_mutation=True)
def load_model():
    with open('kmeans_model.pkl1', 'rb') as f:
        model = pickle.load(f)
    return model

model = load_model()

st.title("Mall Customers Cluster Predictor")

st.sidebar.header("User Input Parameters")

def user_input_features():
    annual_income = st.sidebar.slider('Annual Income (k$)', 0, 150, 50)
    spending_score = st.sidebar.slider('Spending Score (1-100)', 0, 100, 50)
    data = {'Annual Income (k$)': annual_income,
            'Spending Score (1-100)': spending_score}
    features = pd.DataFrame(data, index=[0])
    return features

df = user_input_features()

st.subheader('User Input parameters')
st.write(df)

# Predict the cluster
prediction = model.predict(df)

st.subheader('Prediction')
st.write(f"Predicted Cluster: {prediction[0]}")
