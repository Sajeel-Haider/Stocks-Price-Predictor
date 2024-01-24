import streamlit as st
import time as t

st.set_option('deprecation.showPyplotGlobalUse', False)

#with st.spinner("Loading..."):
#    t.sleep(3)

##############################################################################################
#Side Bar Content

st.sidebar.title("About the project")
st.sidebar.warning("Presenting you this machine learning project. This project contains the data set of S&P 500 Stocks from 1957-2017 containing open, low, high, close as attributes. Its target varaible is Closing Price")

st.sidebar.title("About S&P 500 Stock")
st.sidebar.info("The S&P 500 is a market-weighted index representing 500 leading U.S. companies. It's a vital gauge for the U.S. economy's health, used by investors and analysts to track market trends and guide investment decisions.")

st.sidebar.markdown("Collaborators")
collaborators = [
    {"name": "Ahmad Abdullah", "linkedin_url": "https://www.linkedin.com/in/ahmad-abdullah-240920179/"},
    {"name": "Ahsan Naveed", "linkedin_url": "https://www.linkedin.com/in/ahsan-naveed-805a5a230/"},
    {"name": "Sajeel Haider", "linkedin_url": "https://www.linkedin.com/in/sajeel-haider-863398228/"}
]

for collaborator in collaborators:
    st.sidebar.markdown(
        f'<div style="display: flex; align-items: center; justify-content: space-between; margin-top: 10px">'
        f'    <div>{collaborator["name"]} </div>'
        f'    <a href="{collaborator["linkedin_url"]}" target="_blank" rel="noopener noreferrer" style="text-decoration: none;">'
        f'        <img src="https://img.shields.io/badge/LinkedIn-Connect-blue?style=for-the-badge&logo=linkedin" alt="LinkedIn">'
        f'    </a>'
        f'</div>',
        unsafe_allow_html=True
    )

##############################################################################################
#Main Content

st.title("S&P 500 Stock Price Predictor")

st.button("Description ")
st.info("This machine learning model is based on predicting the stock price")

st.button("Journey ")
st.warning("This project has its own limitation and boundaries, we didnt realized it until until we got indepth in this project. Even when selecting the data we were confused where to find it and also pondered on whether this data is feasible to work on or not. After we got our hands on the dataset it was time to analyze it and check its feasible, the opening, closing, high and low attributes were enough to train our model to predict the curve but then came the issue of whether selecting the whole data or just a window of it, after consultation and analyzing our problem statement we ended up selecting the entire dataset. After plotting the data and using correlation maps we that there were needs of some more features, feature engineering was hectic job in this entire scnerio selecting the right features were necessary for predicting accurately and for training on the dataset")


#############################################################################################
#Graph content 
    
import pandas as pd
import numpy as np
#import matplotlib.pyplot as plt
import seaborn as sns
import datetime
from sklearn.preprocessing import MinMaxScaler

df = pd.read_csv('./S&P.csv')
df = df.iloc[::-1]
df = df.reset_index(drop=True)
print(df)

