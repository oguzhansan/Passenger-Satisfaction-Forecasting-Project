"""
PROJECT : PREDICTION OF AIRLINES PASSENGER SATISFACTION (PIPELINE)

AuthorS  : Team Dataliners


    Business Problem
   In today's world, time and speed have become important factors, which increases the importance of air transportation
day by day. Consequently, competition in the airline industry is increasing. This competition plays a critical role
in the success of airlines because it is important to ensure customer satisfaction. However, the factors that influence
passenger satisfaction are quite complex and can be difficult to understand. In this project, we aim to predict the factors
affecting customer satisfaction using data analytics and machine learning.


    Dataset Story
   In this project, a dataset containing the results of a survey on Airline Passenger Satisfaction is used. The dataset is
divided into two parts: training and testing. The entire dataset (test and training total) consists of 129,880 records
and 25 columns. Analysis and modeling were performed on the training data. The training dataset consists of 103,904
observations and 25 variables. The variables "Id" and "Unnamed" are omitted because they do not make sense.

   Our target variable in the dataset is "satisfaction" and consists of two classes: Undecided or dissatisfied

    Features
- Gender                           : Passenger Sex (Male or female).
- Customer Type                    : Passenger type (Loyal or Disloyal).
- Age                              : The actual age of the passenger.
- Type of Travel                   : The purpose of the passenger's flight (Personal or Business)
- Class                            : Ticket Type (Business, economy, economy plus).
- Flight Distance                  : Distance of Flight.
- Inflight Wi-Fi Service           : Satisfaction level with Wi-Fi service on board (0: not rated; 1-5).
- Departure/Arrival time convenient: Departure/arrival time convenient level (0: not rated; 1-5).
- Ease of Online booking           : Easy online booking rate (0: not rated; 1-5).
- Gate location                    : Level of availible with the gate location (0: not rated; 1-5).
- Food and drink                   : Food and drink flavor level (0: not rated; 1-5).
- Online boarding                  : User friendly online boarding (0: not rated; 1-5).
- Seat comfort                     : Seat comfort level (0: not rated; 1-5).
- Inflight entertainment           : Quality of Inflight entertainment system (0: not rated; 1-5).
- On-board service                 : Flight satisfaction with on-board service (0: not rated; 1-5).
- Legroom service                  : Legroom suitability (0: not rated; 1-5).
- Baggage handling                 : Baggage handling (0: not rated; 1-5).
- Checkin service                  : User friendly checkin service (0: not rated; 1-5).
- Inflight service                 : Level of satisfaction with inflight service (0: not rated; 1-5).
- Cleanliness                      : Airplane cleanliness (0: not rated; 1-5).
- Departure delay in minutes       : Departure delay.
- Arrival delay in minutes         : Arrival delay.

"""

import joblib
import pandas as pd
import numpy as np
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import cross_validate, GridSearchCV, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier


pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 50)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.width', 500)

################################################
# Helper Functions
################################################


# Data Preprocessing & Feature Engineering
def grab_col_names(dataframe, cat_th=20, car_th=40):
    """

    Veri setindeki kategorik, numerik ve kategorik fakat kardinal değişkenlerin isimlerini verir.
    Not: Kategorik değişkenlerin içerisine numerik görünümlü kategorik değişkenler de dahildir.

    Parameters
    ------
        dataframe: dataframe
                Değişken isimleri alınmak istenilen dataframe
        cat_th: int, optional
                numerik fakat kategorik olan değişkenler için sınıf eşik değeri
        car_th: int, optinal
                kategorik fakat kardinal değişkenler için sınıf eşik değeri

    Returns
    ------
        cat_cols: list
                Kategorik değişken listesi
        num_cols: list
                Numerik değişken listesi
        cat_but_car: list
                Kategorik görünümlü kardinal değişken listesi

    Examples
    ------
        import seaborn as sns
        df = sns.load_dataset("iris")
        print(grab_col_names(df))


    Notes
    ------
        cat_cols + num_cols + cat_but_car = toplam değişken sayısı
        num_but_cat cat_cols'un içerisinde.
        Return olan 3 liste toplamı toplam değişken sayısına eşittir: cat_cols + num_cols + cat_but_car = değişken sayısı

    """

    # cat_cols, cat_but_car
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

def label_encoder(dataframe, cat_cols):
    le = LabelEncoder()
    for col in cat_cols:
        dataframe[col] = le.fit_transform(dataframe[col])
    return dataframe

def missing_values_table(dataframe, na_name=False):
    na_columns = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]
    n_miss = dataframe[na_columns].isnull().sum().sort_values(ascending=False)
    ratio = (dataframe[na_columns].isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending=False)
    missing_df = pd.concat([n_miss, np.round(ratio, 2)], axis=1, keys=["n_miss", "ratio"])
    print(missing_df, end="\n")
    if na_name:
        return na_columns

def airlines_data_prep(dataframe):

    missing_values_table(dataframe, True)
    dataframe['Arrival Delay in Minutes'].fillna(dataframe['Arrival Delay in Minutes'].median(), inplace=True)
    dataframe['satisfaction'].replace({'neutral or dissatisfied': 0, 'satisfied': 1}, inplace=True)
    dataframe = dataframe.drop(columns=["Unnamed: 0"])
    dataframe = dataframe.drop(columns=["id"])


    # Gender Loyality
    dataframe.loc[(dataframe['Gender'] == "Male") & (dataframe['Customer Type'] == "Loyal Customer"), 'NEW_CUSTOMER_GENDER'] = "Male Loyal"
    dataframe.loc[(dataframe['Gender'] == "Female") & (dataframe['Customer Type'] == "Loyal Customer"), 'NEW_CUSTOMER_GENDER'] = "Female Loyal"

    dataframe.loc[(dataframe['Gender'] == "Male") & (dataframe['Customer Type'] == "disloyal Customer"), 'NEW_CUSTOMER_GENDER'] = "Male Unloyal"
    dataframe.loc[(dataframe['Gender'] == "Female") & (dataframe['Customer Type'] == "disloyal Customer"), 'NEW_CUSTOMER_GENDER'] = "Female Unloyal"


    # Age Categorization
    dataframe.loc[(dataframe['Age'] >= 7) & (dataframe['Age'] < 25), 'NEW_AGE_CAT'] = "young"
    dataframe.loc[(dataframe['Age'] >= 25) & (dataframe['Age'] < 40), 'NEW_AGE_CAT'] = "mature"
    dataframe.loc[(dataframe['Age'] >= 40) & (dataframe['Age'] < 65), 'NEW_AGE_CAT'] = "middle_age"
    dataframe.loc[(dataframe['Age'] >= 65) & (dataframe['Age'] < 95), 'NEW_AGE_CAT'] = "old_age"


    # Age x Gender
    dataframe.loc[(dataframe['NEW_AGE_CAT'] == "young") & (dataframe['Gender'] == "Male"), 'NEW_AGE_Gender'] = "young Male"
    dataframe.loc[(dataframe['NEW_AGE_CAT'] == "young") & (dataframe['Gender'] == "Female"), 'NEW_AGE_Gender'] = "young Female"

    dataframe.loc[(dataframe['NEW_AGE_CAT'] == "mature") & (dataframe['Gender'] == "Male"), 'NEW_AGE_Gender'] = "mature Male"
    dataframe.loc[(dataframe['NEW_AGE_CAT'] == "mature") & (dataframe['Gender'] == "Female"), 'NEW_AGE_Gender'] = "mature Female"

    dataframe.loc[(dataframe['NEW_AGE_CAT'] == "middle_age") & (dataframe['Gender'] == "Male"), 'NEW_AGE_Gender'] = "middle_age Male"
    dataframe.loc[(dataframe['NEW_AGE_CAT'] == "middle_age") & (dataframe['Gender'] == "Female"), 'NEW_AGE_Gender'] = "middle_age Female"

    dataframe.loc[(dataframe['NEW_AGE_CAT'] == "old_age") & (dataframe['Gender'] == "Male"), 'NEW_AGE_Gender'] = "old_age Male"
    dataframe.loc[(dataframe['NEW_AGE_CAT'] == "old_age") & (dataframe['Gender'] == "Female"), 'NEW_AGE_Gender'] = "old_age Female"


    # Travel Type x Lotality
    dataframe.loc[(dataframe['Type of Travel'] == "Personal Travel") & (dataframe['Customer Type'] == "Loyal Customer"), 'NEW_CUSTOMER_TRAVEL_TYPE'] = "Personal Loyal"
    dataframe.loc[(dataframe['Type of Travel'] == "Business travel") & (dataframe['Customer Type'] == "Loyal Customer"), 'NEW_CUSTOMER_TRAVEL_TYPE'] = "Business Loyal"

    dataframe.loc[(dataframe['Type of Travel'] == "Personal Travel") & (dataframe['Customer Type'] == "disloyal Customer"), 'NEW_CUSTOMER_TRAVEL_TYPE'] = "Personal Unloyal"
    dataframe.loc[(dataframe['Type of Travel'] == "Business travel") & (dataframe['Customer Type'] == "disloyal Customer"), 'NEW_CUSTOMER_TRAVEL_TYPE'] = "Business Unloyal"


    # New Customer Travel Type x Gender
    dataframe.loc[(dataframe['NEW_CUSTOMER_TRAVEL_TYPE'] == "Personal Loyal") & (dataframe['Gender'] == "Male"), 'NEW_CUSTOMER-TRAVEL_GENDER'] = "Personal Loyal Male"
    dataframe.loc[(dataframe['NEW_CUSTOMER_TRAVEL_TYPE'] == "Personal Loyal") & (dataframe['Gender'] == "Female"), 'NEW_CUSTOMER-TRAVEL_GENDER'] = "Personal Loyal Female"

    dataframe.loc[(dataframe['NEW_CUSTOMER_TRAVEL_TYPE'] == "Business Loyal") & (dataframe['Gender'] == "Male"), 'NEW_CUSTOMER-TRAVEL_GENDER'] = "Business Loyal Male"
    dataframe.loc[(dataframe['NEW_CUSTOMER_TRAVEL_TYPE'] == "Business Loyal") & (dataframe['Gender'] == "Female"), 'NEW_CUSTOMER-TRAVEL_GENDER'] = "Business Loyal Female"

    dataframe.loc[(dataframe['NEW_CUSTOMER_TRAVEL_TYPE'] == "Personal Unloyal") & (dataframe['Gender'] == "Male"), 'NEW_CUSTOMER-TRAVEL_GENDER'] = "Personal Unloyal Male"
    dataframe.loc[(dataframe['NEW_CUSTOMER_TRAVEL_TYPE'] == "Personal Unloyal") & (dataframe['Gender'] == "Female"), 'NEW_CUSTOMER-TRAVEL_GENDER'] = "Personal Unloyal Female"

    dataframe.loc[(dataframe['NEW_CUSTOMER_TRAVEL_TYPE'] == "Business Unloyal") & (dataframe['Gender'] == "Male"), 'NEW_CUSTOMER-TRAVEL_GENDER'] = "Business Unloyal Male"
    dataframe.loc[(dataframe['NEW_CUSTOMER_TRAVEL_TYPE'] == "Business Unloyal") & (dataframe['Gender'] == "Female"), 'NEW_CUSTOMER-TRAVEL_GENDER'] = "Business Unloyal Female"


    # New Customer Travel Type x Class
    dataframe.loc[(dataframe['NEW_CUSTOMER_TRAVEL_TYPE'] == "Personal Loyal") & (dataframe['Class'] == "Eco"), 'NEW_TRAVEL_TYPE_CLASS'] = "Personal Loyal Eco"
    dataframe.loc[(dataframe['NEW_CUSTOMER_TRAVEL_TYPE'] == "Personal Loyal") & (dataframe['Class'] == "Eco Plus"), 'NEW_TRAVEL_TYPE_CLASS'] = "Personal Loyal Eco Plus"
    dataframe.loc[(dataframe['NEW_CUSTOMER_TRAVEL_TYPE'] == "Personal Loyal") & (dataframe['Class'] == "Business"), 'NEW_TRAVEL_TYPE_CLASS'] = "Personal Loyal Business"

    dataframe.loc[(dataframe['NEW_CUSTOMER_TRAVEL_TYPE'] == "Business Loyal") & (dataframe['Class'] == "Eco"), 'NEW_TRAVEL_TYPE_CLASS'] = "Business Loyal Eco"
    dataframe.loc[(dataframe['NEW_CUSTOMER_TRAVEL_TYPE'] == "Business Loyal") & (dataframe['Class'] == "Eco Plus"), 'NEW_TRAVEL_TYPE_CLASS'] = "Business Loyal Eco Plus"
    dataframe.loc[(dataframe['NEW_CUSTOMER_TRAVEL_TYPE'] == "Business Loyal") & (dataframe['Class'] == "Business"), 'NEW_TRAVEL_TYPE_CLASS'] = "Business Loyal Business"

    dataframe.loc[(dataframe['NEW_CUSTOMER_TRAVEL_TYPE'] == "Personal Unloyal") & (dataframe['Class'] == "Eco"), 'NEW_TRAVEL_TYPE_CLASS'] = "Personal Unloyal Eco"
    dataframe.loc[(dataframe['NEW_CUSTOMER_TRAVEL_TYPE'] == "Personal Unloyal") & (dataframe['Class'] == "Eco Plus"), 'NEW_TRAVEL_TYPE_CLASS'] = "Personal Unloyal Eco Plus"
    dataframe.loc[(dataframe['NEW_CUSTOMER_TRAVEL_TYPE'] == "Personal Unloyal") & (dataframe['Class'] == "Business"), 'NEW_TRAVEL_TYPE_CLASS'] = "Personal Unloyal Business"

    dataframe.loc[(dataframe['NEW_CUSTOMER_TRAVEL_TYPE'] == "Business Unloyal") & (dataframe['Class'] == "Eco"), 'NEW_TRAVEL_TYPE_CLASS'] = "Business Unloyal Eco"
    dataframe.loc[(dataframe['NEW_CUSTOMER_TRAVEL_TYPE'] == "Business Unloyal") & (dataframe['Class'] == "Eco Plus"), 'NEW_TRAVEL_TYPE_CLASS'] = "Business Unloyal Eco Plus"
    dataframe.loc[(dataframe['NEW_CUSTOMER_TRAVEL_TYPE'] == "Business Unloyal") & (dataframe['Class'] == "Business"), 'NEW_TRAVEL_TYPE_CLASS'] = "Business Unloyal Business"


    # New Delay Absolute
    dataframe["NEW_DELAY_GAP"] = abs(dataframe["Departure Delay in Minutes"] - dataframe["Arrival Delay in Minutes"])


    # Flight Distance Segmentation
    dataframe.loc[(dataframe['Flight Distance'] <= 1500), 'NEW_DISTANCE_SEGMENTATION'] = "kısa mesafe"
    dataframe.loc[(dataframe['Flight Distance'] > 1500), 'NEW_DISTANCE_SEGMENTATION'] = "uzun mesafe"


    # Based Service Score
    dataframe["NEW_FLIGHT_SITUATION"] = (dataframe["Inflight wifi service"] + dataframe["Food and drink"] + dataframe["Seat comfort"] + dataframe["Inflight entertainment"] + dataframe["Leg room service"]) / 25
    dataframe["NEW_OPERATIONAL"] = (dataframe["Departure/Arrival time convenient"] + dataframe["Cleanliness"] + dataframe["Baggage handling"] + dataframe["Gate location"]) / 20
    dataframe["NEW_ONLINE"] = (dataframe["Ease of Online booking"] + dataframe["Online boarding"] + dataframe["Checkin service"]) / 15
    dataframe["NEW_PERSONAL_BEHAVIOR"] = (dataframe["On-board service"] + dataframe["Inflight service"]) / 10


    cat_cols, num_cols, cat_but_car = grab_col_names(dataframe, cat_th=20, car_th=40)


    df = label_encoder(dataframe, cat_cols)

    X_scaled = StandardScaler().fit_transform(df[num_cols])
    df[num_cols] = pd.DataFrame(X_scaled, columns=df[num_cols].columns)

    y = df["satisfaction"]
    X = df.drop(["satisfaction"], axis=1)


    return X, y

# Base Models
def base_models(X, y, scoring="roc_auc"):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=17)
    print("Base Models....")
    classifiers = [('LR', LogisticRegression()),
                   ('KNN', KNeighborsClassifier()),
                   ("SVC", SVC()),
                   ("CART", DecisionTreeClassifier()),
                   ("RF", RandomForestClassifier()),
                   ('Adaboost', AdaBoostClassifier()),
                   ('GBM', GradientBoostingClassifier()),
                   ('XGBoost', XGBClassifier(use_label_encoder=False, eval_metric='logloss')),
                   ('LightGBM', LGBMClassifier()),
                   # ('CatBoost', CatBoostClassifier(verbose=False))
                   ]

    for name, classifier in classifiers:
            model = classifier.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            roc_auc = roc_auc_score(y_test, y_pred)
            print(f"{classifier} ROC AUC: {roc_auc}")


# Hyperparameter Optimization
# config.py
knn_params = {"n_neighbors": range(2, 50)}

cart_params = {'max_depth': range(1, 20),
               "min_samples_split": range(2, 30)}

rf_params = {"max_depth": [8, 15, None],
             "max_features": [5, 7, "auto"],
             "min_samples_split": [15, 20],
             "n_estimators": [200, 300]}

xgboost_params = {"learning_rate": [0.1, 0.01],
                  "max_depth": [5, 8],
                  "n_estimators": [100, 200],
                  "colsample_bytree": [0.5, 1]}

lightgbm_params = {"learning_rate": [0.01, 0.1],
                   "n_estimators": [300, 500],
                   "colsample_bytree": [0.7, 1]}

classifiers = [('KNN', KNeighborsClassifier(), knn_params),
               ("CART", DecisionTreeClassifier(), cart_params),
               ("RF", RandomForestClassifier(), rf_params),
               ('XGBoost', XGBClassifier(use_label_encoder=False, eval_metric='logloss'), xgboost_params),
               ('LightGBM', LGBMClassifier(), lightgbm_params)]


def hyperparameter_optimization(X, y, scoring="roc_auc"):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=17)
    print("Hyperparameter Optimization....")
    best_models = {}
    for name, classifier, params in classifiers:
        print(f"########## {name} ##########")
        model = classifier.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        roc_auc = roc_auc_score(y_test, y_pred)
        print(f"{classifier} (Before): ROC AUC: {roc_auc}")

        gs_best = GridSearchCV(classifier, params, n_jobs=-1, verbose=False).fit(X, y)
        final_model = gs_best.best_estimator_

        model = final_model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        roc_auc = roc_auc_score(y_test, y_pred)
        print(f"{classifier} (After): ROC AUC: {roc_auc}")
        print(f"{name} best params: {gs_best.best_params_}", end="\n\n")
        best_models[name] = final_model
    return best_models




def lgbm_model_save(best_models):
    lgbm_model = best_models["LightGBM"]
    return lgbm_model


################################################
# Pipeline Main Function
################################################


def main():
    df = pd.read_csv("notebook/data/data.csv")
    X, y = airlines_data_prep(df)
    base_models(X, y)
    best_models = hyperparameter_optimization(X, y)
    model = lgbm_model_save(best_models)
    joblib.dump(model, "lgbm.pkl")
    return model

if __name__ == "__main__":
    print("İşlem başladı")
    main()

