"""
PROJECT : PREDICTION OF AIRLINES PASSENGER SATISFACTION (WORKSTATION)

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

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.svm import SVR, SVC

from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler

import warnings
warnings.filterwarnings("ignore")

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 50)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.width', 500)


# Read the train and test CSV files
df = pd.read_csv("notebook/data/data.csv")


################################################
# 1. Exploratory Data Analysis
################################################


def check_df(dataframe, head=5):
    print(10 * "#-#-" + "Shape".center(20) + 10 * "#-#-")
    print(dataframe.shape)
    print(10 * "#-#-" + "Info".center(20) + 10 * "#-#-")
    print(dataframe.info())
    print(10 * "#-#-" + "Head".center(20) + 10 * "#-#-")
    print(dataframe.head(head))
    print(10 * "#-#-" + "Tail".center(20) + 10 * "#-#-")
    print(dataframe.tail(head))
    print(10 * "#-#-" + "NA Values".center(20) + 10 * "#-#-")
    print(dataframe.isnull().sum())
    print(10 * "#-#-" + "Zero Values".center(20) + 10 * "#-#-")
    print((dataframe == 0).sum())
    print(10 * "#-#-" + "Nunique Values".center(20) + 10 * "#-#-")
    print(dataframe.nunique())
    print(10 * "#-#-" + "Describe".center(20) + 10 * "#-#-")
    print(dataframe.describe([0, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99, 1]).T)

def grab_col_names(dataframe, cat_th=10, car_th=20):
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

    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and
                   dataframe[col].dtypes != "O"]
    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and
                   dataframe[col].dtypes == "O"]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f'cat_cols: {len(cat_cols)}')
    print(f'num_cols: {len(num_cols)}')
    print(f'cat_but_car: {len(cat_but_car)}')
    print(f'num_but_cat: {len(num_but_cat)}')
    return cat_cols, num_cols, cat_but_car

def cat_summary(dataframe, col_name, plot=False):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
          "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print("################")

    if plot:
        sns.countplot(x=dataframe[col_name], data=dataframe, palette='viridis')
        plt.show(block=True)

def num_summary(dataframe, numerical_col, plot=False):
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
    print(dataframe[numerical_col].describe(quantiles).T)

    if plot:
        dataframe[numerical_col].hist(bins=20, color='#2C73FF', edgecolor='blue')
        plt.xlabel(numerical_col)
        plt.title(numerical_col)
        plt.show(block=True)

def target_summary_with_cat(dataframe, target, col_name):
    print(pd.DataFrame({"Target Mean": dataframe.groupby(col_name)[target].mean()}), end="\n\n\n")

def plot_countplots(data, target, cat_cols, rows, cols, figsize):
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    axes = axes.flatten()

    for i, col in enumerate(cat_cols):
        sns.countplot(x=col, data=data, hue=target, palette='viridis', ax=axes[i])
        axes[i].set_title(f'Count Plot for {col} by {target}')

    plt.tight_layout()
    plt.show()

def target_summary_with_num(dataframe, target, numeric_col):
    print(dataframe.groupby(target).agg({numeric_col: "mean"}), end="\n\n\n")


df.head()

df = df.drop(columns=["Unnamed: 0"])
df = df.drop(columns=["id"])
# id and Unnamed dropped, because they have no business information.

df['satisfaction'].replace({'neutral or dissatisfied': 0, 'satisfied': 1}, inplace=True)
# Actually, target values can be numerical. We need to replace it.


check_df(df)
# Brief summarize.


cat_cols, num_cols, cat_but_car = grab_col_names(df)
# We wanna capture numerical and categorical variables.

# Target Visualization
plt.pie(df.satisfaction.value_counts(), labels=["Neutral or dissatisfied", "Satisfied"],
        colors=sns.color_palette("YlOrBr"), autopct='%1.1f%%')

# Calculate the counts of each gender
gender_counts = df['Gender'].value_counts()

# Calculate the percentage of each gender category
gender_percentage = (gender_counts / len(df)) * 100


# Analyze for Categorical&Numerical Values
for col in cat_cols:
    cat_summary(df, col, True)

for col in num_cols:
    num_summary(df, col,True)


# Examining numerical variables with target
for col in cat_cols:
    target_summary_with_cat(df, "satisfaction", col)

for col in num_cols:
    target_summary_with_num(df, "satisfaction", col)

plot_countplots(df, 'satisfaction', cat_cols, 10, 3, (20, 20))


# Correlation of all variables with each other
df[num_cols].corr()
f, ax = plt.subplots(figsize=[10, 10])
sns.heatmap(df[num_cols].corr(), annot=True, fmt=".2f", ax=ax, cmap="magma")
ax.set_title("Correlation Matrix", fontsize=20)
plt.show()


################################################
# 2. Data Preprocessing & Feature Engineering
################################################


def missing_values_table(dataframe, na_name=False):
    na_columns = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]
    n_miss = dataframe[na_columns].isnull().sum().sort_values(ascending=False)
    ratio = (dataframe[na_columns].isnull().sum() / df.shape[0] * 100).sort_values(ascending=False)
    missing_df = pd.concat([n_miss, np.round(ratio, 2)], axis=1, keys=["n_miss", "ratio"])
    print(missing_df, end="\n")

    if na_name:
        return na_columns

def outlier_thresholds(dataframe, col_name, q1=0.01, q3=0.99):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    iqr = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * iqr
    low_limit = quartile1 - 1.5 * iqr
    return low_limit, up_limit

def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit

def check_outlier(dataframe, col_name, q1=0.15, q3=0.91):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name, q1, q3)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
        return True
    else:
        return False

def label_encoder(dataframe, binary_col):
    labelencoder = LabelEncoder()
    dataframe[binary_col] = labelencoder.fit_transform(dataframe[binary_col])
    return dataframe


missing_values_table(df, True)

#   n_miss  ratio
# Arrival Delay in Minutes     310  0.300
# ['Arrival Delay in Minutes']


# This values can replace to median or drop. I want to replace it.
df['Arrival Delay in Minutes'].fillna(df['Arrival Delay in Minutes'].median(), inplace=True)

# I wanna check.
df['Arrival Delay in Minutes'].isnull().sum()


# Outlier values table
for col in df:
    plt.figure()
    sns.boxplot(x=df[col])

for col in num_cols:
    print(col, outlier_thresholds(df, col))

for col in num_cols:
    replace_with_thresholds(df, col)

for col in num_cols:
    print(col, check_outlier(df, col))

# Columns have not outlier values.


# New Features

# Gender x Customer Type

df.loc[(df['Gender'] == "Male") & (df['Customer Type'] == "Loyal Customer"), 'NEW_CUSTOMER_GENDER'] = "Male Loyal"
df.loc[(df['Gender'] == "Female") & (df['Customer Type'] == "Loyal Customer"), 'NEW_CUSTOMER_GENDER'] = "Female Loyal"

df.loc[(df['Gender'] == "Male") & (df['Customer Type'] == "disloyal Customer"), 'NEW_CUSTOMER_GENDER'] = "Male Unloyal"
df.loc[(df['Gender'] == "Female") & (df['Customer Type'] == "disloyal Customer"), 'NEW_CUSTOMER_GENDER'] = "Female Unloyal"

df.groupby("NEW_CUTOMER_GENDER").agg({"satisfaction": ["mean", "count"]})


# Age categorization

df.loc[(df['Age'] >= 7) & (df['Age'] < 25), 'NEW_AGE_CAT'] = "young"
df.loc[(df['Age'] >= 25) & (df['Age'] < 40), 'NEW_AGE_CAT'] = "mature"
df.loc[(df['Age'] >= 40) & (df['Age'] < 65), 'NEW_AGE_CAT'] = "middle_age"
df.loc[(df['Age'] >= 65) & (df['Age'] < 95), 'NEW_AGE_CAT'] = "old_age"

df.groupby("NEW_AGE_CAT").agg({"satisfaction": ["mean", "count"]})


# Age X gender

df.loc[(df['NEW_AGE_CAT'] == "young") & (df['Gender'] == "Male"), 'NEW_AGE_Gender'] = "young Male"
df.loc[(df['NEW_AGE_CAT'] == "young") & (df['Gender'] == "Female"), 'NEW_AGE_Gender'] = "young Female"

df.loc[(df['NEW_AGE_CAT'] == "mature") & (df['Gender'] == "Male"), 'NEW_AGE_Gender'] = "mature Male"
df.loc[(df['NEW_AGE_CAT'] == "mature") & (df['Gender'] == "Female"), 'NEW_AGE_Gender'] = "mature Female"

df.loc[(df['NEW_AGE_CAT'] == "middle_age") & (df['Gender'] == "Male"), 'NEW_AGE_Gender'] = "middle_age Male"
df.loc[(df['NEW_AGE_CAT'] == "middle_age") & (df['Gender'] == "Female"), 'NEW_AGE_Gender'] = "middle_age Female"

df.loc[(df['NEW_AGE_CAT'] == "old_age") & (df['Gender'] == "Male"), 'NEW_AGE_Gender'] = "old_age Male"
df.loc[(df['NEW_AGE_CAT'] == "old_age") & (df['Gender'] == "Female"), 'NEW_AGE_Gender'] = "old_age Female"

df.groupby("NEW_AGE_Gender").agg({"satisfaction": ["mean", "count"]})


# Travel type X lotality

df.loc[(df['Type of Travel'] == "Personal Travel") & (df['Customer Type'] == "Loyal Customer"), 'NEW_CUSTOMER_TRAVEL_TYPE'] = "Personal Loyal"
df.loc[(df['Type of Travel'] == "Business travel") & (df['Customer Type'] == "Loyal Customer"), 'NEW_CUSTOMER_TRAVEL_TYPE'] = "Business Loyal"

df.loc[(df['Type of Travel'] == "Personal Travel") & (df['Customer Type'] == "disloyal Customer"), 'NEW_CUSTOMER_TRAVEL_TYPE'] = "Personal Unloyal"
df.loc[(df['Type of Travel'] == "Business travel") & (df['Customer Type'] == "disloyal Customer"), 'NEW_CUSTOMER_TRAVEL_TYPE'] = "Business Unloyal"

df.groupby("NEW_CUSTOMER_TRAVEL_TYPE").agg({"satisfaction": ["mean", "count"]})


# New customer travel type x gender

df.loc[(df['NEW_CUSTOMER_TRAVEL_TYPE'] == "Personal Loyal") & (df['Gender'] == "Male"), 'NEW_CUSTOMER-TRAVEL_GENDER'] = "Personal Loyal Male"
df.loc[(df['NEW_CUSTOMER_TRAVEL_TYPE'] == "Personal Loyal") & (df['Gender'] == "Female"), 'NEW_CUSTOMER-TRAVEL_GENDER'] = "Personal Loyal Female"

df.loc[(df['NEW_CUSTOMER_TRAVEL_TYPE'] == "Business Loyal") & (df['Gender'] == "Male"), 'NEW_CUSTOMER-TRAVEL_GENDER'] = "Business Loyal Male"
df.loc[(df['NEW_CUSTOMER_TRAVEL_TYPE'] == "Business Loyal") & (df['Gender'] == "Female"), 'NEW_CUSTOMER-TRAVEL_GENDER'] = "Business Loyal Female"

df.loc[(df['NEW_CUSTOMER_TRAVEL_TYPE'] == "Personal Unloyal") & (df['Gender'] == "Male"), 'NEW_CUSTOMER-TRAVEL_GENDER'] = "Personal Unloyal Male"
df.loc[(df['NEW_CUSTOMER_TRAVEL_TYPE'] == "Personal Unloyal") & (df['Gender'] == "Female"), 'NEW_CUSTOMER-TRAVEL_GENDER'] = "Personal Unloyal Female"

df.loc[(df['NEW_CUSTOMER_TRAVEL_TYPE'] == "Business Unloyal") & (df['Gender'] == "Male"), 'NEW_CUSTOMER-TRAVEL_GENDER'] = "Business Unloyal Male"
df.loc[(df['NEW_CUSTOMER_TRAVEL_TYPE'] == "Business Unloyal") & (df['Gender'] == "Female"), 'NEW_CUSTOMER-TRAVEL_GENDER'] = "Business Unloyal Female"

df.groupby("NEW_CUSTOMER-TRAVEL_GENDER").agg({"satisfaction": ["mean", "count"]})


# New customer travel type x class

df.loc[(df['NEW_CUSTOMER_TRAVEL_TYPE'] == "Personal Loyal") & (df['Class'] == "Eco"), 'NEW_TRAVEL_TYPE_CLASS'] = "Personal Loyal Eco"
df.loc[(df['NEW_CUSTOMER_TRAVEL_TYPE'] == "Personal Loyal") & (df['Class'] == "Eco Plus"), 'NEW_TRAVEL_TYPE_CLASS'] = "Personal Loyal Eco Plus"
df.loc[(df['NEW_CUSTOMER_TRAVEL_TYPE'] == "Business Loyal") & (df['Class'] == "Eco Plus"), 'NEW_TRAVEL_TYPE_CLASS'] = "Business Loyal Eco Plus"
df.loc[(df['NEW_CUSTOMER_TRAVEL_TYPE'] == "Business Loyal") & (df['Class'] == "Business"), 'NEW_TRAVEL_TYPE_CLASS'] = "Business Loyal Business"

df.loc[(df['NEW_CUSTOMER_TRAVEL_TYPE'] == "Personal Unloyal") & (df['Class'] == "Eco"), 'NEW_TRAVEL_TYPE_CLASS'] = "Personal Unloyal Eco"
df.loc[(df['NEW_CUSTOMER_TRAVEL_TYPE'] == "Personal Unloyal") & (df['Class'] == "Eco Plus"), 'NEW_TRAVEL_TYPE_CLASS'] = "Personal Unloyal Eco Plus"
df.loc[(df['NEW_CUSTOMER_TRAVEL_TYPE'] == "Personal Unloyal") & (df['Class'] == "Business"), 'NEW_TRAVEL_TYPE_CLASS'] = "Personal Unloyal Business"

df.loc[(df['NEW_CUSTOMER_TRAVEL_TYPE'] == "Business Unloyal") & (df['Class'] == "Eco"), 'NEW_TRAVEL_TYPE_CLASS'] = "Business Unloyal Eco"
df.loc[(df['NEW_CUSTOMER_TRAVEL_TYPE'] == "Business Unloyal") & (df['Class'] == "Eco Plus"), 'NEW_TRAVEL_TYPE_CLASS'] = "Business Unloyal Eco Plus"
df.loc[(df['NEW_CUSTOMER_TRAVEL_TYPE'] == "Business Unloyal") & (df['Class'] == "Business"), 'NEW_TRAVEL_TYPE_CLASS'] = "Business Unloyal Business"

df.groupby("NEW_TRAVEL_TYPE_CLASS").agg({"satisfaction": ["mean", "count"]})


# New delay gap
df["NEW_DELAY_GAP"] = abs(df["Departure Delay in Minutes"] - df["Arrival Delay in Minutes"])

df.groupby("NEW_DELAY_GAP").agg({"satisfaction": ["mean", "count"]})


# Flight distance segmentation
df.loc[(df['Flight Distance'] <= 1500), 'NEW_DISTANCE_SEGMENTATION'] = "kısa mesafe"
df.loc[(df['Flight Distance'] > 1500), 'NEW_DISTANCE_SEGMENTATION'] = "uzun mesafe"

df.groupby("NEW_DISTANCE_SEGMENTATION").agg({"satisfaction": ["mean", "count"]})


# Based Service Score
df["NEW_FLIGHT_SITUATION"] = (df["Inflight wifi service"] + df["Food and drink"] + df["Seat comfort"] + df["Inflight entertainment"] + df["Leg room service"]) / 25
df["NEW_OPERATIONAL"] = (df["Departure/Arrival time convenient"] + df["Cleanliness"] + df["Baggage handling"] + df["Gate location"]) / 20
df["NEW_ONLINE"] = (df["Ease of Online booking"] + df["Online boarding"] + df["Checkin service"]) / 15
df["NEW_PERSONAL_BEHAVIOR"] = (df["On-board service"] + df["Inflight service"]) / 10

df.groupby("NEW_FLIGHT_SITUATION").agg({"satisfaction": ["mean", "count"]})
df.groupby("NEW_OPERATIONAL").agg({"satisfaction": ["mean", "count"]})
df.groupby("NEW_ONLINE").agg({"satisfaction": ["mean", "count"]})
df.groupby("NEW_PERSONAL_BEHAVIOR").agg({"satisfaction": ["mean", "count"]})


cat_cols, num_cols, cat_but_car = grab_col_names(df)
# Observations: 103904
# Variables: 35
# cat_cols: 26
# num_cols: 9
# cat_but_car: 0
# num_but_cat: 15

cat_cols = [col for col in cat_cols if "satisfaction" not in col]

binary_cols = [col for col in df.columns if df[col].dtypes == "O" and df[col].nunique() == 2]

# ['Gender', 'Customer Type', 'Type of Travel', 'NEW_DISTANCE_SEGMENTATION']

for col in binary_cols:
    df = label_encoder(df, col)

non_binary_cols = [col for col in df[cat_cols] if col not in binary_cols]

for col in non_binary_cols:
    df = label_encoder(df, col)


######################################################
# 3. Base Models
######################################################


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


X_scaled = StandardScaler().fit_transform(df[num_cols])
df[num_cols] = pd.DataFrame(X_scaled, columns=df[num_cols].columns)

y = df["satisfaction"]
X = df.drop(["satisfaction"], axis=1)

base_models(X, y)


# LogisticRegression() ROC AUC: 0.8723421881639924
# KNeighborsClassifier() ROC AUC: 0.9175566871613726
# SVC() ROC AUC: 0.9364587352724648
# DecisionTreeClassifier() ROC AUC: 0.9424947623004417
# RandomForestClassifier() ROC AUC: 0.9595615857049661
# AdaBoostClassifier() ROC AUC: 0.9247561159434128
# GradientBoostingClassifier() ROC AUC: 0.943063306670105
# XGBClassifier() ROC AUC: 0.9607174506598084
# LGBMClassifier() ROC AUC: 0.9624792244652398


######################################################
# 4. Automated Hyperparameter Optimization
######################################################


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

best_models = hyperparameter_optimization(X, y)


# Hyperparameter Optimization....
# ########## KNN ##########
# KNeighborsClassifier() (Before): ROC AUC: 0.9175566871613726
# KNeighborsClassifier() (After) : ROC AUC: 0.9196429803173012
# KNN best params: {'n_neighbors': 9}
#
# ########## CART ##########
# DecisionTreeClassifier() (Before): ROC AUC: 0.9429791379142162
# DecisionTreeClassifier() (After) : ROC AUC: 0.9493476862684218
# CART best params: {'max_depth': 15, 'min_samples_split': 28}
#
# ########## RF ##########
# RandomForestClassifier() (Before): ROC AUC: 0.9598739211898896
# RandomForestClassifier() (After): ROC AUC: 0.960006679746741
# RF best params: {'max_depth': None, 'max_features': 7, 'min_samples_split': 15, 'n_estimators': 200}
#
# ########## XGBClassifier ##########
# XGBClassifier() (Before): ROC AUC: 0.9607174506598084
# XGBClassifier() (After): ROC AUC: 0.9617641260009988
# XGBoost best params: {'colsample_bytree': 1, 'learning_rate': 0.1, 'max_depth': 8, 'n_estimators': 200}
#
# ########## LightGBM ##########
# LGBMClassifier() (Before): ROC AUC: 0.9624792244652398
# LGBMClassifier() (After): ROC AUC: 0.9627017714861271
# LightGBM best params: {'colsample_bytree': 1, 'learning_rate': 0.1, 'n_estimators': 300}




