from io import BytesIO
import joblib
import pandas as pd
import streamlit as st
from function import *
from lightgbm import LGBMClassifier

# Genel Sayfa Ayarları
st.set_page_config(layout="centered", page_title="Dataliners Hava Yolları",
                   page_icon="images/airplane.ico")

#Background Resminin Ayarlanması
img = get_img_as_base64("./images/background.jpg")
page_bg_img = f"""
<style>
[data-testid="stAppViewContainer"] > .main {{
background-image:url("data:image/png;base64,{img}");
background-size: cover;
background-position: center top;
background-repeat: no-repeat;
background-attachment: local;
}}
}}
{{
[data-testid="stVerticalBlockBorderWrapper"]{{
background-color: rgba(38, 38, 54, 0.3); 
border-radius: 16px;
}}
.st-ds {{
background-color: rgba(38, 39, 48, 0);
}}
[.data-testid="stColorBlock"]{{
background-color: rgba(38, 39, 10;
}}
</style>
"""
st.markdown(page_bg_img, unsafe_allow_html=True)

# Sayfa Başlığı ve Yazı Stili
st.markdown("""
    <style>
        .title {
            text-align: center;
            font-family: Yellow peace;
            font-weight: lighter;
            color: rgba(43, 45, 49);
            font-size: 2.5rem;
            padding-bottom: 20px;
        }
        .me {
            text-align: center;
            font-family: Yellow peace;
            color: rgba(43, 45, 49);
            font-size: 1 rem;
            padding: 0;
            margin: 0;
        }

    </style>
""", unsafe_allow_html=True)
st.markdown("<h1 class='title'> Miuul Airlines R&D </h1>", unsafe_allow_html=True)

# Ana Ekran Giriş Sayfası
st.image("./images/a2.png")
st.markdown("<p class='me'>Miuul Airlines</p>", unsafe_allow_html=True)
st.markdown("<p class='me'>Passenger Satisfaction Forecasting System</p>", unsafe_allow_html=True)
st.markdown("<p class='me'>1.3.0</p>", unsafe_allow_html=True)