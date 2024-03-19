import joblib
from function import *



st.set_page_config(layout="centered", page_title="Dataliners Hava Yolları",
                   page_icon="artitects/5929224_airplane_earth_global_globe_trave_icon.ico")

st.title("DataLiners Hava Yolları")



img = get_img_as_base64("artitects/ross-parmly-rf6ywHVkrlY-unsplash.jpg")
page_bg_img = f"""
<style>
[data-testid="stAppViewContainer"] > .main {{
background-image:url("data:image/png;base64,{img}");
background-size: 100%;
background-position: center top;
background-repeat: no-repeat;
background-attachment: local;
}}

[data-testid="stHeader"] {{
background: rgba(0,0,0,0);
}}

[data-testid="stToolbar"] {{
right: 2rem;
}}
</style>
"""
st.markdown(page_bg_img, unsafe_allow_html=True)




# Feature inputs
tab1, tab2, tab3, tab4 ,tab5 = st.tabs(["🛬 Temel Uçuş Bilgileri", "💺 Uçuş İçi Hizmet",
                                  "🧳 Operasyonel Hizmet", "🧑🏻‍💻 Online Hizmet", "📩 Submit"])

tab1col1, tab1col2 = tab1.columns(2)
tab1col3, tab1col4 = tab1.columns(2)

tab2col1, tab2col2 = tab2.columns(2)
tab2col3, tab2col4 = tab2.columns(2)




Gender = tab1col1.selectbox("Your Gender", ["Male", "Female"])
Customer_Type = tab1col2.selectbox("Customer Type", ["Loyal Customer", "disloyal Customer"])
Age = tab1.slider("Your Age", 7, 85, step=1)
Type_of_Travel = tab1col3.radio("Travel Type", ["Business travel", "Personal Travel"])
Class = tab1col4.radio("Class", ["Business", "Eco Plus", "Eco"])
Flight_Distance = tab1.slider("Distance", 31, 4983, step=1)


kategorik_degerler = {
    "Hiç memnun değilim 😢": 0,
    "Memnun değilim 😕": 1,
    "İdare eder 😐": 2,
    "Ortalama 🙂": 3,
    "Memnunum 😊": 4,
    "Çok memnunum 😄": 5
}


####### Uçuş içi Hizmet###########


Inflight_service = tab2col1.selectbox("INFLIGHT SERVICE", list(kategorik_degerler.keys()))
Inflight_wifi_service = tab2col3.selectbox("INFLIGHT WIFI SERVICE",list(kategorik_degerler.keys()))
Inflight_entertainment = tab2col2.selectbox("INFLIGHT ENTERTAINMENT",list(kategorik_degerler.keys()))
Food_and_drink = tab2col4.selectbox("FOOD AND DRINK", list(kategorik_degerler.keys()))
Seat_comfort = tab2.selectbox("SEAT COMFORT", list(kategorik_degerler.keys()))

Inflight_service = kategorik_degerler[Inflight_service]
Inflight_wifi_service = kategorik_degerler[Inflight_wifi_service]
Inflight_entertainment = kategorik_degerler[Inflight_entertainment]
Food_and_drink = kategorik_degerler[Food_and_drink]
Seat_comfort = kategorik_degerler[Seat_comfort]


########## Operasyonel Hizmet ###############

Departure_Arrival_time_convenient = tab3.selectbox("Departure Arrival Time", list(kategorik_degerler.keys()))
Gate_location = tab3.selectbox("Gate_location", list(kategorik_degerler.keys()))
On_board_service = tab3.selectbox("On_board_service", list(kategorik_degerler.keys()))
Baggage_handling = tab3.selectbox("Baggage_handling", list(kategorik_degerler.keys()))
Cleanliness = tab3.selectbox("Cleanliness", list(kategorik_degerler.keys()))

# Delay Minutes
tab3col1, tab3col2 = tab3.columns(2)

Departure_Delay_in_Minutes = tab3col1.number_input("Departure_Delay_in_Minutes", 0, 1600)
Arrival_Delay_in_Minutes = tab3col2.number_input("Arrival_Delay_in_Minutes", 0, 1600)


Departure_Arrival_time_convenient = kategorik_degerler[Departure_Arrival_time_convenient]
Gate_location = kategorik_degerler[Gate_location]
On_board_service = kategorik_degerler[On_board_service]
Baggage_handling = kategorik_degerler[Baggage_handling]
Cleanliness = kategorik_degerler[Cleanliness]



########## Online Hizmet ###################

Ease_of_Online_booking = tab4.selectbox("Ease_of_Online_booking", list(kategorik_degerler.keys()))
Online_boarding = tab4.selectbox("Online_boarding", list(kategorik_degerler.keys()))
Leg_room_service = tab4.selectbox("Leg_room_service", list(kategorik_degerler.keys()))
Checkin_service = tab4.selectbox("Checkin_service", list(kategorik_degerler.keys()))


Ease_of_Online_booking = kategorik_degerler[Ease_of_Online_booking]
Online_boarding = kategorik_degerler[Online_boarding]
Leg_room_service = kategorik_degerler[Leg_room_service]
Checkin_service = kategorik_degerler[Checkin_service]






with open("style/pred.html", "r", encoding="utf-8") as pred:
    pred_html = f"""{pred.read()}"""
    st.markdown(pred_html, unsafe_allow_html=True)


tab5col1, tab5col2 = tab5.columns(2)

# Genişlik ve yükseklik ayarları
button_width = 200
button_height = 100



# Butonun stilini belirle
st.markdown(
    f"""
    <style>
    .stButton > button {{
        width: {button_width}px;
        height: {button_height}px;
        background-color: #87CEEB; /* Gökyüzü mavisi renk */
        border: none;
        color: black;
        padding: 15px 32px;
        text-align: center;
        text-decoration: none;
        display: inline-block;
        margin: 4px 2px;
        cursor: pointer;
        border-radius: 24px; /* Yuvarlak kenarlık */
        box-shadow: 0 6px 12px 0 rgba(0,0,0,0.2), 0 6px 20px 0 rgba(0,0,0,0.19); /* Gölge efekti */
        transition-duration: 0.4s;
    }}

    .stButton > button:hover {{
        background-color: #add8e6; /* Hover efekti ile daha koyu bir mavi */
        color:black;
    
    }}
    </style>
    """,
    unsafe_allow_html=True
)


if tab5col1.button("PREDICT"):





    fd = pd.DataFrame({'Gender': [Gender],
                       'Customer Type': [Customer_Type],
                       'Age': [Age],
                       'Type of Travel': [Type_of_Travel],
                       'Class': [Class],
                       'Flight Distance': [Flight_Distance],
                       'Inflight wifi service': [Inflight_wifi_service],
                       'Departure/Arrival time convenient': [Departure_Arrival_time_convenient],
                       'Ease of Online booking': [Ease_of_Online_booking],
                       'Gate location': [Gate_location],
                       'Food and drink': [Food_and_drink],
                       'Online boarding': [Online_boarding],
                       'Seat comfort': [Seat_comfort],
                       'Inflight entertainment': [Inflight_entertainment],
                       'On-board service': [On_board_service],
                       'Leg room service': [Leg_room_service],
                       'Baggage handling': [Baggage_handling],
                       'Checkin service': [Checkin_service],
                       'Inflight service': [Inflight_service],
                       'Cleanliness': [Cleanliness],
                       'Departure Delay in Minutes': [Departure_Delay_in_Minutes],
                       'Arrival Delay in Minutes': [Arrival_Delay_in_Minutes]})


    df = pd.read_csv("data.csv")


    input_df = pred_data(df, fd)



    new_model = joblib.load("lgbm.pkl")
    pred = new_model.predict(input_df)


    file_ = open("png/dissatisfied.png", "rb")
    contents = file_.read()
    data_url = base64.b64encode(contents).decode("utf-8")
    file_.close()

    file_ = open("png/satisfied.png", "rb")
    contents = file_.read()
    data_url2 = base64.b64encode(contents).decode("utf-8")
    file_.close()


    if pred[0] == 0:
        tab5col2.markdown(
        f'<img src="data:image/gif;base64,{data_url}" alt="cat gif">',
        unsafe_allow_html=True)

    else:
        tab5col2.markdown(
        f'<img src="data:image/gif;base64,{data_url2}" alt="cat gif">',
        unsafe_allow_html=True)
