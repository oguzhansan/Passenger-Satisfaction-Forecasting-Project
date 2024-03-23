import joblib
import streamlit as st
from matplotlib import pyplot as plt

from function import *

st.set_page_config(layout="centered", page_title="Dataliners Hava Yolları",
                   page_icon="images/airplane.ico")


img = get_img_as_base64("./images/background.jpg")

page_bg_img = f"""
<style>
[data-testid="stAppViewContainer"] > .main {{
background-image: url("data:image/png;base64,{img}");
background-size: 180%;
background-position: top left;
background-repeat: no-repeat;
background-attachment: local;
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

taba, tabb, tabc, tabd, tab1, tab2 = st.tabs(["____", "____", "____", "____", "🗃️ Data Upload & Download",
                                                                              "‍📊️ Data Analyze"])
taba.image("./images/analyze.png")

st.sidebar.header("Miuul Airlines R&D")

uploaded_files = tab1.file_uploader("Choose a file", accept_multiple_files=True)


bigData = bigdats(uploaded_files)

tab1.write(bigData)

if tab1.button("PREDICTIONS"):
    bigDataPred = save(bigData)

    new_model = joblib.load("model/lgbm.pkl")
    pred = new_model.predict(bigDataPred)

    bigData["Predictions"] = pred

    bigData['Predictions'].replace({0: 'neutral or dissatisfied', 1: 'satisfied'}, inplace=True)
    tab1.write(bigData)

    href = download_excel(bigData)
    tab1.markdown(f'<a href="{href}" download="dataset.xlsx"><button>Download Excel File</button></a>',
                  unsafe_allow_html=True)

    with tab2:
        col1, col2 = st.columns(2)
        gender_counts = bigData['Gender'].value_counts()

        if bigData is not None:
            # 'Gender' sütununu kontrol edin
            if 'Gender' in bigData.columns:
                gender_counts = bigData['Gender'].value_counts()

                fig = plt.figure(figsize=(4, 3))
                fig.patch.set_alpha(0)
                # Grafik oluştur
                plt.bar(gender_counts.index, gender_counts.values, color=['blue', 'darkblue'])
                plt.xlabel('Gender')
                plt.ylabel('Count')
                plt.title('Gender Distribution')
                plt.xticks(rotation=45)
                # Streamlit'te grafik göster
                col1.pyplot(fig)
            else:
                st.error("Veri kümesinde 'Gender' sütunu bulunamadı.")
        else:
            st.error("Veri yüklenemedi veya boş.")

        class_counts = bigData['Class'].value_counts()

        fig2 = plt.figure(figsize=(4, 3))
        fig2.patch.set_alpha(0)
        # Grafik oluştur
        plt.bar(class_counts.index, class_counts.values, color=['red', 'orange', "purple"])
        plt.xlabel('Class')
        plt.ylabel('Count')
        plt.title('Class Distribution')
        plt.xticks(rotation=45)
        col2.pyplot(fig2)

        customer_counts = bigData['Customer Type'].value_counts()

        fig3 = plt.figure(figsize=(4, 3))
        fig3.patch.set_alpha(0)
        # Grafik oluştur
        plt.bar(customer_counts.index, customer_counts.values, color=['Pink', 'Green'])
        plt.xlabel('Customer Type')
        plt.ylabel('Count')
        plt.title('Customer Distribution')
        plt.xticks(rotation=45)
        col1.pyplot(fig3)

        type_counts = bigData['Type of Travel'].value_counts()

        fig4 = plt.figure(figsize=(4, 3))
        fig4.patch.set_alpha(0)
        # Grafik oluştur
        plt.bar(type_counts.index, type_counts.values, color=['yellow', 'grey', "black"])
        plt.xlabel('Type of Travel')
        plt.ylabel('Count')
        plt.title('Travel Distribution')
        plt.xticks(rotation=45)
        col2.pyplot(fig4)