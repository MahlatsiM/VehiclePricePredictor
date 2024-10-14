! pip install seaborn
! pip install plotly
! pip install matplotlib

import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt

# Set page configuration for better styling
st.set_page_config(page_title="Car Price Predictor", page_icon="🏎", layout="centered")

# Style the front page
st.markdown("""
    <style>
    body {
        background-image: url('https://example.com/path-to-your-background-image.jpg');  /* Add your image URL here */
        background-size: cover; 
        background-position: center;
        color: white;
    }
    .header {
        font-size: 48px;
        font-weight: bold;
        text-align: center;
        margin-top: 50px;
        text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.7);
    }
    .subheader {
        font-size: 24px;
        font-weight: bold;
        margin-top: 20px;
        text-align: center;
        text-shadow: 1px 1px 3px rgba(0, 0, 0, 0.5);
    }
    .content {
        font-size: 18px;
        line-height: 1.6;
        text-align: justify;
        margin: 10px 20px;
        background-color: rgba(0, 0, 0, 0.6);
        padding: 20px;
        border-radius: 10px;
    }
    .footer {
        margin-top: 40px;
        font-size: 16px;
        text-align: center;
        text-shadow: 1px 1px 3px rgba(0, 0, 0, 0.5);
    }
    .highlight {
        color: #f39c12;
        font-weight: bold;
    }
    .feature {
        margin-top: 20px;
        padding: 10px;
        border: 1px solid #f39c12;
        border-radius: 5px;
        background-color: rgba(0, 0, 0, 0.7);
    }
    </style>
""", unsafe_allow_html=True)

# Add header
st.markdown('<div class="header">Car Price Prediction Project🏎</div>', unsafe_allow_html=True)

# Welcome message
st.markdown("""
    <div class="content">
    Welcome to the <span class="highlight">Car Price Predictor</span>, a powerful machine learning-based web application designed to help you analyze and predict car prices based on various factors like <span class="highlight">mileage, engine power, fuel type, age</span>, and more.
    </div>
""", unsafe_allow_html=True)

# Explanation of how it works
st.markdown('<div class="subheader">How It Works</div>', unsafe_allow_html=True)
st.markdown("""
    <div class="content">
    <ol>
        <li>
            <span style="font-size: 16px; font-weight: bold;">Provide Car Details</span><br>
            Share the specific details of the vehicle you’d like to get a price prediction for, including model, year, and mileage.
        </li><br>
        <li>
            <span style="font-size: 16px; font-weight: bold;">Receive Your Prediction</span><br>
            Our machine learning model will analyze the information provided and deliver an expected price range for your vehicle.
        </li>
    </ol>
    </div>
""", unsafe_allow_html=True)


# Features of the application
st.markdown('<div class="subheader">Features</div>', unsafe_allow_html=True)
features = [
    "🔍 Compare used car prices to dealership book values.",
    "📈 Track car prices and get insights from past vehicle data."
]
for feature in features:
    st.markdown(f'<div class="feature">{feature}</div>', unsafe_allow_html=True)

st.markdown('<div class="subheader">Frequently Asked Questions</div>', unsafe_allow_html=True)

faq = [
    {
        "question": "How do car dealerships determine the price of a car?",
        "answer": "As you can imagine, every car dealership doesn’t have the expertise or resources to invest in a team of automotive data scientists to analyse historic car sales data and estimate a fair price for each of their cars. That’s why car dealerships outsource this responsibility to third-party automotive data providers like TransUnion, Lightstone, DiskDrive, and a few others. These automotive data providers provide car dealerships with book, trade, and retail values to help them price their cars."
    },
    {
        "question": "What's the difference between a car's book value, trade value, and retail value?",
        "answer": (
            "* **Book value**: The price a dealership is recommended to buy a car for.\n"
            "* **Trade value**: As the name suggests, this is the trade-in value of a car and it’s the same as the above mentioned 'book value'.\n"
            "* **Retail value**: The price a dealership is recommended to sell a car for."
       )
    },
    {
        "question": "Who determines a car's book/trade/retail value?",
        "answer": "Various entities determine these values, including automotive publishers (like Kelley Blue Book or Edmunds), auction companies, and car dealerships, using historical data, market trends, and current conditions."
    },
    {
        "question": "How is a car’s book/trade/retail value calculated?",
        "answer": "These values are calculated using algorithms that take into account factors such as the car's make, model, year, mileage, condition, location, and market demand. Data is often aggregated from sales, auctions, and dealer transactions."
    },
    {
        "question": "Where can I learn more about the Car Market in South Africa?",
        "answer": (
            "Below are well known South African Car review YouTube channels that offer comprehensive vehicle reviews and car buying tips: \n\n "
            "+ **MrHowMuch**: https://www.youtube.com/@mrhowmuch \n"
            "+ **Muzi Sambo**: https://www.youtube.com/@MuziSambo \n"
            "+ **Reba S. Cars**: https://www.youtube.com/@RebaSCars \n"
                   )
    }
]

# Display FAQ with dropdowns
for item in faq:
    with st.expander(item["question"], expanded=False):
        st.write(item["answer"], unsafe_allow_html=True)


# Call to Action
st.markdown('<div class="subheader">Get Started</div>', unsafe_allow_html=True)
st.markdown("""
    <div class="content">
    Use the prediction tool by entering specific details about a car and let the machine learning models do the work. Start exploring now to <span class="highlight">predict your next car price</span> with ease.
    </div>
""", unsafe_allow_html=True)

# Footer Section
st.markdown("""
    <div class="footer">
    Created by Mahlatsi Malise Mashilo as part of the Data Science Capstone Project.
    </div>
""", unsafe_allow_html=True)

# Create a sidebar for navigation
st.sidebar.title("Navigation")
st.sidebar.success("Select a page above")
