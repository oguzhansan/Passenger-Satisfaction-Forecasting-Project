import base64
import io
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from lightgbm import LGBMClassifier

# Girilen Input Verisinin Tahminini Bir Dataframe Olarak Döndüren Fonksiyon
def pred_data(df, df_input):
    def label_encoder(dataframe, cat_cols):
        le = LabelEncoder()
        for col in cat_cols:
            dataframe[col] = le.fit_transform(dataframe[col])
        return dataframe


    df['Arrival Delay in Minutes'].fillna(df['Arrival Delay in Minutes'].median(), inplace=True)
    df['satisfaction'].replace({'neutral or dissatisfied': 0, 'satisfied': 1}, inplace=True)
    df = df.drop(columns=["Unnamed: 0"])
    df = df.drop(columns=["id"])
    df = pd.concat([df, df_input], ignore_index=True)


    def grab_col_names(dataframe, cat_th=20, car_th=40):
        cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
        num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and
                       dataframe[col].dtypes != "O"]
        cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and
                       dataframe[col].dtypes == "O"]
        cat_cols = cat_cols + num_but_cat
        cat_cols = [col for col in cat_cols if col not in cat_but_car]

        # num_cols
        num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
        num_cols = [col for col in num_cols if col not in num_but_cat]

        return cat_cols, num_cols, cat_but_car


    # Gender Loyality
    df.loc[(df['Gender'] == "Male") & (
            df['Customer Type'] == "Loyal Customer"), 'NEW_CUSTOMER_GENDER'] = "Male Loyal"
    df.loc[(df['Gender'] == "Female") & (
            df['Customer Type'] == "Loyal Customer"), 'NEW_CUSTOMER_GENDER'] = "Female Loyal"

    df.loc[(df['Gender'] == "Male") & (
            df['Customer Type'] == "disloyal Customer"), 'NEW_CUSTOMER_GENDER'] = "Male Unloyal"
    df.loc[(df['Gender'] == "Female") & (
            df['Customer Type'] == "disloyal Customer"), 'NEW_CUSTOMER_GENDER'] = "Female Unloyal"

    # Age Categorization
    df.loc[(df['Age'] >= 7) & (df['Age'] < 25), 'NEW_AGE_CAT'] = "young"
    df.loc[(df['Age'] >= 25) & (df['Age'] < 40), 'NEW_AGE_CAT'] = "mature"
    df.loc[(df['Age'] >= 40) & (df['Age'] < 65), 'NEW_AGE_CAT'] = "middle_age"
    df.loc[(df['Age'] >= 65) & (df['Age'] < 95), 'NEW_AGE_CAT'] = "old_age"

    # Age x Gender
    df.loc[(df['NEW_AGE_CAT'] == "young") & (df['Gender'] == "Male"), 'NEW_AGE_Gender'] = "young Male"
    df.loc[(df['NEW_AGE_CAT'] == "young") & (df['Gender'] == "Female"), 'NEW_AGE_Gender'] = "young Female"

    df.loc[(df['NEW_AGE_CAT'] == "mature") & (df['Gender'] == "Male"), 'NEW_AGE_Gender'] = "mature Male"
    df.loc[(df['NEW_AGE_CAT'] == "mature") & (df['Gender'] == "Female"), 'NEW_AGE_Gender'] = "mature Female"

    df.loc[(df['NEW_AGE_CAT'] == "middle_age") & (
            df['Gender'] == "Male"), 'NEW_AGE_Gender'] = "middle_age Male"
    df.loc[(df['NEW_AGE_CAT'] == "middle_age") & (
            df['Gender'] == "Female"), 'NEW_AGE_Gender'] = "middle_age Female"

    df.loc[(df['NEW_AGE_CAT'] == "old_age") & (
            df['Gender'] == "Male"), 'NEW_AGE_Gender'] = "old_age Male"
    df.loc[(df['NEW_AGE_CAT'] == "old_age") & (
            df['Gender'] == "Female"), 'NEW_AGE_Gender'] = "old_age Female"

    # Travel Type x Lotality
    df.loc[(df['Type of Travel'] == "Personal Travel") & (
            df['Customer Type'] == "Loyal Customer"), 'NEW_CUSTOMER_TRAVEL_TYPE'] = "Personal Loyal"
    df.loc[(df['Type of Travel'] == "Business travel") & (
            df['Customer Type'] == "Loyal Customer"), 'NEW_CUSTOMER_TRAVEL_TYPE'] = "Business Loyal"

    df.loc[(df['Type of Travel'] == "Personal Travel") & (
            df['Customer Type'] == "disloyal Customer"), 'NEW_CUSTOMER_TRAVEL_TYPE'] = "Personal Unloyal"
    df.loc[(df['Type of Travel'] == "Business travel") & (
            df['Customer Type'] == "disloyal Customer"), 'NEW_CUSTOMER_TRAVEL_TYPE'] = "Business Unloyal"

    # New Customer Travel Type x Gender
    df.loc[(df['NEW_CUSTOMER_TRAVEL_TYPE'] == "Personal Loyal") & (
            df['Gender'] == "Male"), 'NEW_CUSTOMER-TRAVEL_GENDER'] = "Personal Loyal Male"
    df.loc[(df['NEW_CUSTOMER_TRAVEL_TYPE'] == "Personal Loyal") & (
            df['Gender'] == "Female"), 'NEW_CUSTOMER-TRAVEL_GENDER'] = "Personal Loyal Female"

    df.loc[(df['NEW_CUSTOMER_TRAVEL_TYPE'] == "Business Loyal") & (
            df['Gender'] == "Male"), 'NEW_CUSTOMER-TRAVEL_GENDER'] = "Business Loyal Male"
    df.loc[(df['NEW_CUSTOMER_TRAVEL_TYPE'] == "Business Loyal") & (
            df['Gender'] == "Female"), 'NEW_CUSTOMER-TRAVEL_GENDER'] = "Business Loyal Female"

    df.loc[(df['NEW_CUSTOMER_TRAVEL_TYPE'] == "Personal Unloyal") & (
            df['Gender'] == "Male"), 'NEW_CUSTOMER-TRAVEL_GENDER'] = "Personal Unloyal Male"
    df.loc[(df['NEW_CUSTOMER_TRAVEL_TYPE'] == "Personal Unloyal") & (
            df['Gender'] == "Female"), 'NEW_CUSTOMER-TRAVEL_GENDER'] = "Personal Unloyal Female"

    df.loc[(df['NEW_CUSTOMER_TRAVEL_TYPE'] == "Business Unloyal") & (
            df['Gender'] == "Male"), 'NEW_CUSTOMER-TRAVEL_GENDER'] = "Business Unloyal Male"
    df.loc[(df['NEW_CUSTOMER_TRAVEL_TYPE'] == "Business Unloyal") & (
            df['Gender'] == "Female"), 'NEW_CUSTOMER-TRAVEL_GENDER'] = "Business Unloyal Female"

    # New Customer Travel Type x Class
    df.loc[(df['NEW_CUSTOMER_TRAVEL_TYPE'] == "Personal Loyal") & (
            df['Class'] == "Eco"), 'NEW_TRAVEL_TYPE_CLASS'] = "Personal Loyal Eco"
    df.loc[(df['NEW_CUSTOMER_TRAVEL_TYPE'] == "Personal Loyal") & (
            df['Class'] == "Eco Plus"), 'NEW_TRAVEL_TYPE_CLASS'] = "Personal Loyal Eco Plus"
    df.loc[(df['NEW_CUSTOMER_TRAVEL_TYPE'] == "Personal Loyal") & (
            df['Class'] == "Business"), 'NEW_TRAVEL_TYPE_CLASS'] = "Personal Loyal Business"

    df.loc[(df['NEW_CUSTOMER_TRAVEL_TYPE'] == "Business Loyal") & (
            df['Class'] == "Eco"), 'NEW_TRAVEL_TYPE_CLASS'] = "Business Loyal Eco"
    df.loc[(df['NEW_CUSTOMER_TRAVEL_TYPE'] == "Business Loyal") & (
            df['Class'] == "Eco Plus"), 'NEW_TRAVEL_TYPE_CLASS'] = "Business Loyal Eco Plus"
    df.loc[(df['NEW_CUSTOMER_TRAVEL_TYPE'] == "Business Loyal") & (
            df['Class'] == "Business"), 'NEW_TRAVEL_TYPE_CLASS'] = "Business Loyal Business"

    df.loc[(df['NEW_CUSTOMER_TRAVEL_TYPE'] == "Personal Unloyal") & (
            df['Class'] == "Eco"), 'NEW_TRAVEL_TYPE_CLASS'] = "Personal Unloyal Eco"
    df.loc[(df['NEW_CUSTOMER_TRAVEL_TYPE'] == "Personal Unloyal") & (
            df['Class'] == "Eco Plus"), 'NEW_TRAVEL_TYPE_CLASS'] = "Personal Unloyal Eco Plus"
    df.loc[(df['NEW_CUSTOMER_TRAVEL_TYPE'] == "Personal Unloyal") & (
            df['Class'] == "Business"), 'NEW_TRAVEL_TYPE_CLASS'] = "Personal Unloyal Business"

    df.loc[(df['NEW_CUSTOMER_TRAVEL_TYPE'] == "Business Unloyal") & (
            df['Class'] == "Eco"), 'NEW_TRAVEL_TYPE_CLASS'] = "Business Unloyal Eco"
    df.loc[(df['NEW_CUSTOMER_TRAVEL_TYPE'] == "Business Unloyal") & (
            df['Class'] == "Eco Plus"), 'NEW_TRAVEL_TYPE_CLASS'] = "Business Unloyal Eco Plus"
    df.loc[(df['NEW_CUSTOMER_TRAVEL_TYPE'] == "Business Unloyal") & (
            df['Class'] == "Business"), 'NEW_TRAVEL_TYPE_CLASS'] = "Business Unloyal Business"

    # New Delay Absolute
    df["NEW_DELAY_GAP"] = abs(
        df["Departure Delay in Minutes"] - df["Arrival Delay in Minutes"])

    # Flight Distance Segmentation
    df.loc[(df['Flight Distance'] <= 1500), 'NEW_DISTANCE_SEGMENTATION'] = "kısa mesafe"
    df.loc[(df['Flight Distance'] > 1500), 'NEW_DISTANCE_SEGMENTATION'] = "uzun mesafe"

    # Based Service Score
    df["NEW_FLIGHT_SITUATION"] = (df["Inflight wifi service"] + df["Food and drink"] +
                                  df["Seat comfort"] + df["Inflight entertainment"] +
                                  df["Leg room service"]) / 25
    df["NEW_OPERATIONAL"] = (df["Departure/Arrival time convenient"] + df["Cleanliness"] +
                             df["Baggage handling"] + df["Gate location"]) / 20
    df["NEW_ONLINE"] = (df["Ease of Online booking"] + df["Online boarding"] + df[
        "Checkin service"]) / 15
    df["NEW_PERSONAL_BEHAVIOR"] = (df["On-board service"] + df["Inflight service"]) / 10

    cat_cols, num_cols, cat_but_car = grab_col_names(df, cat_th=20, car_th=40)

    df = label_encoder(df, cat_cols)

    X_scaled = StandardScaler().fit_transform(df[num_cols])
    df[num_cols] = pd.DataFrame(X_scaled, columns=df[num_cols].columns)

    X = df.drop(["satisfaction"], axis=1)

    # Inputun Ölçeklenmiş ve Encode Edilmiş Halinin Modele Hazırlanması
    df1 = X.iloc[-1]

    l1 = [float(df1[0]), float(df1[1]), float(df1[2]), float(df1[3]), int(df1[4]), int(df1[5]),
          int(df1[6]), int(df1[7]), int(df1[8]), int(df1[9]), int(df1[10]), int(df1[11]), int(df1[12]),
          int(df1[13]), int(df1[14]), int(df1[15]), int(df1[16]), int(df1[17]), int(df1[18]), int(df1[19]),
          int(df1[20]), int(df1[21]), int(df1[22]), int(df1[23]), int(df1[24]), int(df1[25]), int(df1[26]),
          int(df1[27]), int(df1[28]), int(df1[29]), int(df1[30]), int(df1[31]), int(df1[32]), int(df1[33])]

    # Input Verilerinden Tek Satırlık DataFrame Oluşturma

    l2 = np.array(l1).reshape(1, -1)
    input_df = pd.DataFrame(l2)

    return input_df


# Arka Plan Yükleme Fonksiyonu
def get_img_as_base64(file):
    with open(file, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()


# Toplu Veri Girişinin Ölçeklenmesi ve Encode Edilmesini Sağlayan Fonksiyon
def save(bigData):
            def label_encoder(dataframe, cat_cols):
                le = LabelEncoder()
                for col in cat_cols:
                    dataframe[col] = le.fit_transform(dataframe[col])
                return dataframe

            df = pd.read_csv("data/data.csv")
            df['Arrival Delay in Minutes'].fillna(df['Arrival Delay in Minutes'].median(), inplace=True)
            df['satisfaction'].replace({'neutral or dissatisfied': 0, 'satisfied': 1}, inplace=True)
            df = df.drop(columns=["Unnamed: 0"])
            df = df.drop(columns=["id"])

            df = pd.concat([df, bigData], ignore_index=True)

            def grab_col_names(dataframe, cat_th=20, car_th=40):
                cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
                num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and
                               dataframe[col].dtypes != "O"]
                cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and
                               dataframe[col].dtypes == "O"]
                cat_cols = cat_cols + num_but_cat
                cat_cols = [col for col in cat_cols if col not in cat_but_car]

                # num_cols
                num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
                num_cols = [col for col in num_cols if col not in num_but_cat]

                return cat_cols, num_cols, cat_but_car

                # Gender Loyality


            df.loc[(df['Gender'] == "Male") & (
                    df['Customer Type'] == "Loyal Customer"), 'NEW_CUSTOMER_GENDER'] = "Male Loyal"
            df.loc[(df['Gender'] == "Female") & (
                    df['Customer Type'] == "Loyal Customer"), 'NEW_CUSTOMER_GENDER'] = "Female Loyal"

            df.loc[(df['Gender'] == "Male") & (
                    df['Customer Type'] == "disloyal Customer"), 'NEW_CUSTOMER_GENDER'] = "Male Unloyal"
            df.loc[(df['Gender'] == "Female") & (
                    df['Customer Type'] == "disloyal Customer"), 'NEW_CUSTOMER_GENDER'] = "Female Unloyal"

            # Age Categorization
            df.loc[(df['Age'] >= 7) & (df['Age'] < 25), 'NEW_AGE_CAT'] = "young"
            df.loc[(df['Age'] >= 25) & (df['Age'] < 40), 'NEW_AGE_CAT'] = "mature"
            df.loc[(df['Age'] >= 40) & (df['Age'] < 65), 'NEW_AGE_CAT'] = "middle_age"
            df.loc[(df['Age'] >= 65) & (df['Age'] < 95), 'NEW_AGE_CAT'] = "old_age"

            # Age x Gender
            df.loc[(df['NEW_AGE_CAT'] == "young") & (df['Gender'] == "Male"), 'NEW_AGE_Gender'] = "young Male"
            df.loc[(df['NEW_AGE_CAT'] == "young") & (df['Gender'] == "Female"), 'NEW_AGE_Gender'] = "young Female"

            df.loc[(df['NEW_AGE_CAT'] == "mature") & (df['Gender'] == "Male"), 'NEW_AGE_Gender'] = "mature Male"
            df.loc[(df['NEW_AGE_CAT'] == "mature") & (df['Gender'] == "Female"), 'NEW_AGE_Gender'] = "mature Female"

            df.loc[(df['NEW_AGE_CAT'] == "middle_age") & (
                    df['Gender'] == "Male"), 'NEW_AGE_Gender'] = "middle_age Male"
            df.loc[(df['NEW_AGE_CAT'] == "middle_age") & (
                    df['Gender'] == "Female"), 'NEW_AGE_Gender'] = "middle_age Female"

            df.loc[(df['NEW_AGE_CAT'] == "old_age") & (
                    df['Gender'] == "Male"), 'NEW_AGE_Gender'] = "old_age Male"
            df.loc[(df['NEW_AGE_CAT'] == "old_age") & (
                    df['Gender'] == "Female"), 'NEW_AGE_Gender'] = "old_age Female"

            # Travel Type x Lotality
            df.loc[(df['Type of Travel'] == "Personal Travel") & (
                    df['Customer Type'] == "Loyal Customer"), 'NEW_CUSTOMER_TRAVEL_TYPE'] = "Personal Loyal"
            df.loc[(df['Type of Travel'] == "Business travel") & (
                    df['Customer Type'] == "Loyal Customer"), 'NEW_CUSTOMER_TRAVEL_TYPE'] = "Business Loyal"

            df.loc[(df['Type of Travel'] == "Personal Travel") & (
                    df['Customer Type'] == "disloyal Customer"), 'NEW_CUSTOMER_TRAVEL_TYPE'] = "Personal Unloyal"
            df.loc[(df['Type of Travel'] == "Business travel") & (
                    df['Customer Type'] == "disloyal Customer"), 'NEW_CUSTOMER_TRAVEL_TYPE'] = "Business Unloyal"

            # New Customer Travel Type x Gender
            df.loc[(df['NEW_CUSTOMER_TRAVEL_TYPE'] == "Personal Loyal") & (
                    df['Gender'] == "Male"), 'NEW_CUSTOMER-TRAVEL_GENDER'] = "Personal Loyal Male"
            df.loc[(df['NEW_CUSTOMER_TRAVEL_TYPE'] == "Personal Loyal") & (
                    df['Gender'] == "Female"), 'NEW_CUSTOMER-TRAVEL_GENDER'] = "Personal Loyal Female"

            df.loc[(df['NEW_CUSTOMER_TRAVEL_TYPE'] == "Business Loyal") & (
                    df['Gender'] == "Male"), 'NEW_CUSTOMER-TRAVEL_GENDER'] = "Business Loyal Male"
            df.loc[(df['NEW_CUSTOMER_TRAVEL_TYPE'] == "Business Loyal") & (
                    df['Gender'] == "Female"), 'NEW_CUSTOMER-TRAVEL_GENDER'] = "Business Loyal Female"

            df.loc[(df['NEW_CUSTOMER_TRAVEL_TYPE'] == "Personal Unloyal") & (
                    df['Gender'] == "Male"), 'NEW_CUSTOMER-TRAVEL_GENDER'] = "Personal Unloyal Male"
            df.loc[(df['NEW_CUSTOMER_TRAVEL_TYPE'] == "Personal Unloyal") & (
                    df['Gender'] == "Female"), 'NEW_CUSTOMER-TRAVEL_GENDER'] = "Personal Unloyal Female"

            df.loc[(df['NEW_CUSTOMER_TRAVEL_TYPE'] == "Business Unloyal") & (
                    df['Gender'] == "Male"), 'NEW_CUSTOMER-TRAVEL_GENDER'] = "Business Unloyal Male"
            df.loc[(df['NEW_CUSTOMER_TRAVEL_TYPE'] == "Business Unloyal") & (
                    df['Gender'] == "Female"), 'NEW_CUSTOMER-TRAVEL_GENDER'] = "Business Unloyal Female"

            # New Customer Travel Type x Class
            df.loc[(df['NEW_CUSTOMER_TRAVEL_TYPE'] == "Personal Loyal") & (
                    df['Class'] == "Eco"), 'NEW_TRAVEL_TYPE_CLASS'] = "Personal Loyal Eco"
            df.loc[(df['NEW_CUSTOMER_TRAVEL_TYPE'] == "Personal Loyal") & (
                    df['Class'] == "Eco Plus"), 'NEW_TRAVEL_TYPE_CLASS'] = "Personal Loyal Eco Plus"
            df.loc[(df['NEW_CUSTOMER_TRAVEL_TYPE'] == "Personal Loyal") & (
                    df['Class'] == "Business"), 'NEW_TRAVEL_TYPE_CLASS'] = "Personal Loyal Business"

            df.loc[(df['NEW_CUSTOMER_TRAVEL_TYPE'] == "Business Loyal") & (
                    df['Class'] == "Eco"), 'NEW_TRAVEL_TYPE_CLASS'] = "Business Loyal Eco"
            df.loc[(df['NEW_CUSTOMER_TRAVEL_TYPE'] == "Business Loyal") & (
                    df['Class'] == "Eco Plus"), 'NEW_TRAVEL_TYPE_CLASS'] = "Business Loyal Eco Plus"
            df.loc[(df['NEW_CUSTOMER_TRAVEL_TYPE'] == "Business Loyal") & (
                    df['Class'] == "Business"), 'NEW_TRAVEL_TYPE_CLASS'] = "Business Loyal Business"

            df.loc[(df['NEW_CUSTOMER_TRAVEL_TYPE'] == "Personal Unloyal") & (
                    df['Class'] == "Eco"), 'NEW_TRAVEL_TYPE_CLASS'] = "Personal Unloyal Eco"
            df.loc[(df['NEW_CUSTOMER_TRAVEL_TYPE'] == "Personal Unloyal") & (
                    df['Class'] == "Eco Plus"), 'NEW_TRAVEL_TYPE_CLASS'] = "Personal Unloyal Eco Plus"
            df.loc[(df['NEW_CUSTOMER_TRAVEL_TYPE'] == "Personal Unloyal") & (
                    df['Class'] == "Business"), 'NEW_TRAVEL_TYPE_CLASS'] = "Personal Unloyal Business"

            df.loc[(df['NEW_CUSTOMER_TRAVEL_TYPE'] == "Business Unloyal") & (
                    df['Class'] == "Eco"), 'NEW_TRAVEL_TYPE_CLASS'] = "Business Unloyal Eco"
            df.loc[(df['NEW_CUSTOMER_TRAVEL_TYPE'] == "Business Unloyal") & (
                    df['Class'] == "Eco Plus"), 'NEW_TRAVEL_TYPE_CLASS'] = "Business Unloyal Eco Plus"
            df.loc[(df['NEW_CUSTOMER_TRAVEL_TYPE'] == "Business Unloyal") & (
                    df['Class'] == "Business"), 'NEW_TRAVEL_TYPE_CLASS'] = "Business Unloyal Business"

            # New Delay Absolute
            df["NEW_DELAY_GAP"] = abs(
                df["Departure Delay in Minutes"] - df["Arrival Delay in Minutes"])

            # Flight Distance Segmentation
            df.loc[(df['Flight Distance'] <= 1500), 'NEW_DISTANCE_SEGMENTATION'] = "kısa mesafe"
            df.loc[(df['Flight Distance'] > 1500), 'NEW_DISTANCE_SEGMENTATION'] = "uzun mesafe"

            # Based Service Score
            df["NEW_FLIGHT_SITUATION"] = (df["Inflight wifi service"] + df["Food and drink"] +
                                          df["Seat comfort"] + df["Inflight entertainment"] +
                                          df["Leg room service"]) / 25
            df["NEW_OPERATIONAL"] = (df["Departure/Arrival time convenient"] + df["Cleanliness"] +
                                     df["Baggage handling"] + df["Gate location"]) / 20
            df["NEW_ONLINE"] = (df["Ease of Online booking"] + df["Online boarding"] + df[
                "Checkin service"]) / 15
            df["NEW_PERSONAL_BEHAVIOR"] = (df["On-board service"] + df["Inflight service"]) / 10

            bh = len(bigData)

            cat_cols, num_cols, cat_but_car = grab_col_names(df, cat_th=20, car_th=40)

            df = label_encoder(df, cat_cols)

            X_scaled = StandardScaler().fit_transform(df[num_cols])
            df[num_cols] = pd.DataFrame(X_scaled, columns=df[num_cols].columns)

            X = df.drop(["satisfaction"], axis=1)


            bigDataPred = X.iloc[-bh:]

            return bigDataPred

# Toplu Veri Yükleme Fonksiyonu
def bigdats(uploaded):

    if uploaded is None:
        return None

    all_data = []

    for uploaded_file in uploaded:
        file_bytes = uploaded_file.read()

        if not file_bytes:
            continue


        bigData = pd.read_csv(io.BytesIO(file_bytes))
        all_data.append(bigData)

        if all_data:
            bigData = pd.concat(all_data, ignore_index=True)
            return bigData




# Kullanıcıdan alınan datanın excel formatında indirilmesi için buton oluştuma
def download_excel(lastdata):
    excel_buffer = io.BytesIO()

    lastdata.to_excel(excel_buffer, index=False)

    excel_buffer.seek(0)

    b64 = base64.b64encode(excel_buffer.read()).decode()
    href = f'data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{b64}'
    return href
