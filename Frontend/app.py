import streamlit as st
import os
from dotenv import load_dotenv
import requests

# Load environment variables from .env file
load_dotenv()

# Get the API URL from the .env file
API_URL = os.getenv("API_URL")

# Page configuration
st.set_page_config(
    page_title="Medical Insurance Cost Prediction",
    page_icon="💰",
    layout="centered",
)

# Page title
st.title("Medical Insurance Cost Prediction")

# Description
st.info(
    "This application predicts the cost of medical insurance based on user inputs. "
    "Please fill in the details below and click 'Predict' to see the estimated insurance cost."
)

st.subheader("Please provide the following information:")

# Create two columns
col1, col2 = st.columns(2)

# First column
with col1:
    age = st.number_input(
        "Age",
        min_value=0,
        max_value=120,
        value=30
    )

    bmi = st.number_input(
        "BMI (Body Mass Index)",
        min_value=10.0,
        max_value=100.0,
        value=25.0
    )

    children = st.number_input(
        "Number of Children",
        min_value=0,
        max_value=15,
        value=0
    )

# Second column
with col2:
    sex = st.selectbox(
        "Sex",
        options=["Male", "Female"]
    )

    smoker = st.selectbox(
        "Smoker",
        options=["Yes", "No"]
    )

    region = st.selectbox(
        "Region",
        options=[
            "Northeast",
            "Northwest",
            "Southeast",
            "Southwest"
        ]
    )

# Prediction button
if st.button("Predict"):

    # Check that the API URL exists before making the request
    if not API_URL:
        st.error(
            "API_URL is not set. Please set the API_URL "
            "environment variable in your .env file."
        )
        st.stop()

    # Prepare the input data
    payload = {
        "age": age,
        "sex": sex,
        "bmi": bmi,
        "children": children,
        "smoker": smoker,
        "region": region
    }

    # Send the data to the FastAPI backend
    try:
        response = requests.post(
            API_URL,
            json=payload,
            timeout=20
        )

        # Raise an error if the API returns a bad status code
        response.raise_for_status()

        # Convert JSON response into a Python dictionary
        result = response.json()

        # Get the predicted insurance charge
        prediction = result["predicted_charges"]

        # Display the result
        st.success(
            f"The predicted medical insurance cost is: ${prediction:,.2f}"
        )

    except requests.RequestException as e:
        st.error(
            "Unable to connect to the prediction API. "
            "Please make sure your FastAPI backend is running."
        )
        st.caption(f"Error details: {e}")

    except KeyError:
        st.error(
            "The API response does not contain 'predicted_charges'. "
            "Please check your FastAPI backend response."
        )

    except ValueError:
        st.error(
            "The API returned an invalid JSON response."
        )