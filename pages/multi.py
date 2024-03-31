import joblib
import streamlit as st
from matplotlib import pyplot as plt
from function import *

# Genel Sayfa Ayarları
st.set_page_config(layout="centered", page_title="Dataliners Hava Yolları",
                   page_icon="images/airplane.ico")


# Background Resminin Ayarlanması
img = get_img_as_base64("./images/Fearless - 7.jpeg")
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

[data-testid="stHeader"]
{{background: rgba(56,97,142,0.3);}}
{{[data-testid="stVerticalBlockBorderWrapper"]
{{background-color: rgba(38, 38, 54, 0.3); border-radius: 16px;}}

[.data-testid="stColorBlock"]{{
    background-color: rgba(38, 39, 10;
}}
</style>
"""
st.markdown(page_bg_img, unsafe_allow_html=True)
st.markdown(
    f"""
    <style>
        section[data-testid="stSidebar"] {{
            width: 200px !important; 
        }}
    </style>
    """,
    unsafe_allow_html=True,
)
# Sayfa Başlığı ve Yazı Stili
st.markdown("""
    <style>
        .title {
            text-align: center;
            font-family: Yellow peace;
            font-weight: lighter;
            color: rgba(91, 162, 194);
            font-size: 2.5rem;
            padding-bottom: 20px;
        }
        .me {
            text-align: center;
            font-family: Yellow peace;
            color: rgba(94, 78, 207);
            font-size: 1 rem;
            padding: 0;
            margin: 0;
        }

    </style>
""", unsafe_allow_html=True)
st.markdown("<h1 class='title'> Miuul Airlines R&D </h1>", unsafe_allow_html=True)

# Sayfa Düzenine Tabların Eklenmesi
(taba, tabb, tabc, tabd, tabe, tabf, tabg,
 tabh, tabj, tabk, tabl, tabm, tabn, tab1, tab2) = st.tabs(["____", "____", "____", "____",
                                                                  "____", "____", "____", "____",
                                                                  "____", "____", "____", "____",
                                                                  "____",
                                                                  "🗃️ Data Upload & Download",
                                                                  "‍📊️ Data Analyze"])

# Ana Ekran Giriş Sayfası
taba.image("./images/Fearless - 1.png")


st.sidebar.markdown("<br>", unsafe_allow_html=True)
st.sidebar.image("./images/b7.png")

# Veri seti yükleme
uploaded_files = tab1.file_uploader("Choose a file", accept_multiple_files=True)
bigData = bigdats(uploaded_files)

# Yüklenen veri setinin ön izlemesi
tab1.write(bigData)


# Tahminleme İşlemlerin Gerçekleştiği Fonksiyon
if tab1.button("PREDICTIONS"):
    bigDataPred = save(bigData)

    # Model Yükleme
    new_model = joblib.load("model/lgbm.pkl")
    pred = new_model.predict(bigDataPred)

    # Tahmin Sonuçlarının Veri Setine İşlenmesi
    bigData["Predictions"] = pred
    bigData['Predictions'].replace({0: 'neutral or dissatisfied', 1: 'satisfied'}, inplace=True)

    # Hazırlanan Veri Setinin Ekrana Yansıtılması
    tab1.write(bigData)

    # Excel Formatında İndirme Butonu Oluşturma
    href = download_excel(bigData)
    tab1.markdown(f'<a href="{href}" download="dataset.xlsx"><button>Download Excel File</button></a>',
                  unsafe_allow_html=True)

    # Grafiklerin Ekrana Yansıtılması
    with tab2:
        col1, col2 = st.columns(2)

        gender_counts = bigData['Gender'].value_counts()

        fig = plt.figure(figsize=(4, 3))
        fig.patch.set_alpha(0)

        plt.bar(gender_counts.index, gender_counts.values, color=['blue', 'darkblue'])
        plt.xlabel('Gender')
        plt.ylabel('Count')
        plt.title('Gender Distribution')
        plt.xticks(rotation=45)

        col1.pyplot(fig)

        class_counts = bigData['Class'].value_counts()

        fig2 = plt.figure(figsize=(4, 3))
        fig2.patch.set_alpha(0)

        plt.bar(class_counts.index, class_counts.values, color=['red', 'orange', "purple"])
        plt.xlabel('Class')
        plt.ylabel('Count')
        plt.title('Class Distribution')
        plt.xticks(rotation=45)
        col2.pyplot(fig2)

        customer_counts = bigData['Customer Type'].value_counts()

        fig3 = plt.figure(figsize=(4, 3))
        fig3.patch.set_alpha(0)

        plt.bar(customer_counts.index, customer_counts.values, color=['Pink', 'Green'])
        plt.xlabel('Customer Type')
        plt.ylabel('Count')
        plt.title('Customer Distribution')
        plt.xticks(rotation=45)
        col1.pyplot(fig3)

        type_counts = bigData['Type of Travel'].value_counts()

        fig4 = plt.figure(figsize=(4, 3))
        fig4.patch.set_alpha(0)

        plt.bar(type_counts.index, type_counts.values, color=['yellow', 'grey', "black"])
        plt.xlabel('Type of Travel')
        plt.ylabel('Count')
        plt.title('Travel Distribution')
        plt.xticks(rotation=45)
        col2.pyplot(fig4)

# Sayfa Footer HTML Kod Uygulaması
with open("style/footer.html", "r", encoding="utf-8") as pred:
    footer_html = f"""{pred.read()}"""
    st.markdown(footer_html, unsafe_allow_html=True)
