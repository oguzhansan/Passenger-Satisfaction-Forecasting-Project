import streamlit as st
from function import *
from st_pages import Page, show_pages
# General Page Settings
st.set_page_config(layout="centered", page_title="Dataliners Hava Yolları",
                   page_icon="images/airplane.ico")

# Pages Desing to Side Bar
show_pages(
    [
        Page("homepage.py", "Home", "🏠"),
        Page("pages/single.py", "Single Data Entry", "📃"),
        Page("pages/multi.py", "Multiple Data Entry", "📂")
    ]
)

st.sidebar.markdown("<br>", unsafe_allow_html=True)

st.sidebar.image("./images/b7.png")

# Background Resminin Ayarlanması
img = get_img_as_base64("./images/Fearless - 7.jpeg")
page_bg_img = f"""
<style>
[data-testid="stAppViewContainer"] > .main {{
background-image:url("data:image/png;base64,{img}");
background-size: cover;
background-position: center top;
background-repeat: no-repeat;
background-attachment: local;
}}
[data-testid="stHeader"]
{{background: rgba(56,97,142,0.3);}}
{{[data-testid="stVerticalBlockBorderWrapper"]
{{background-color: rgba(38, 38, 54, 0.3); border-radius: 16px;}}
.st-ds 
{{background-color: rgba(38, 39, 48, 0);}}
[.data-testid="stColorBlock"]
{{background-color: rgba(38, 39, 10;}}
</style>
"""
st.markdown(page_bg_img, unsafe_allow_html=True)

# Set the width to your desired value

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

# Ana Ekran Giriş Sayfası
st.image("./images/Fearless - 3.png")

# Sayfa Footer HTML Kod Uygulaması
with open("style/footer.html", "r", encoding="utf-8") as pred:
    footer_html = f"""{pred.read()}"""
    st.markdown(footer_html, unsafe_allow_html=True)
