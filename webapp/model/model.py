'''
This script does the following:
1) Loads in the dataset
2) Preprocesses the dataset according to findings from EDA
3) Builds several machine learning models
4) Evaluates and performs predictions on the models to find the one with best performance

Usage: python util/model.py
'''

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from imblearn.over_sampling import SMOTE
import pandas as pd
import numpy as np
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning) 

# Read engineered dataset into dataframe
df = pd.read_csv("data/Application_Data_Final.csv")

# Target df to be Status column
target = df['Status']
# Features df drops Status column
features = df.drop(['Status'], axis=1)

X_train, X_test, y_train, y_test = train_test_split(features, target, random_state=123)

# print("X_train: \n" , X_train.head())
# print("X_test: \n" , X_test.head())
# print("y_train: \n" , y_train.head())
# print("y_test: \n" , y_test.head())

y_train = y_train.astype('int')
# Use SMOTE due to sample disparity in good/bad status
X_balance, Y_balance = SMOTE().fit_resample(X_train, y_train)
X_balance = pd.DataFrame(X_balance, columns = X_train.columns)
Y_balance = pd.DataFrame(Y_balance, columns=["Status"])
# print(X_balance.shape)
# print(Y_balance.shape)

def display_results(model, X_balance, X_test, Y_balance, y_test, training=True):
    if training:
        pred = model.predict(X_balance)
        model_report = pd.DataFrame(classification_report(Y_balance, pred, output_dict=True))
        print(model_report)
        print("Accuracy Score: ", accuracy_score(Y_balance, pred))
    else:
        pred = model.predict(X_test)
        model_report = pd.DataFrame(classification_report(y_test, pred, output_dict=True))
        print(model_report)
        print("Accuracy Score: ", accuracy_score(y_test, pred))

# TESTING DIFFERENT ML MODELS
# LOGISTIC REGRESSION
lr_model = LogisticRegression(class_weight='balanced', solver='liblinear')
lr_model.fit(X_balance, Y_balance.values.ravel())
print("Results for Logistic Regression:\n")
print(display_results(lr_model, X_balance, X_test, Y_balance, y_test, training=True))
print(display_results(lr_model, X_balance, X_test, Y_balance, y_test, training=False))

# K-NEAREST NEIGHBORS
knn_model = KNeighborsClassifier()
knn_model.fit(X_balance, Y_balance.values.ravel())

print(display_results(knn_model, X_balance, X_test, Y_balance, y_test, training=True))
print(display_results(knn_model, X_balance, X_test, Y_balance, y_test, training=False))

# DECISION TREE
dt_model = DecisionTreeClassifier()
dt_model.fit(X_balance, Y_balance.values.ravel())

print(display_results(dt_model, X_balance, X_test, Y_balance, y_test, training=True))
print(display_results(dt_model, X_balance, X_test, Y_balance, y_test, training=False))

# # RANDOM FOREST
rf_model = RandomForestClassifier()
rf_model.fit(X_balance, Y_balance.values.ravel())

print(display_results(rf_model, X_balance, X_test, Y_balance, y_test, training=True))
print(display_results(rf_model, X_balance, X_test, Y_balance, y_test, training=False))

# # Male, 1, 1, 31111, 32, 5, True, False, False, False
# # Applicant_Gender,Owned_Car,Owned_Realty,Total_Income,Income_Type,Education_Type,Family_Status,Housing_Type,Job_Title,Applicant_Age,Years_of_Working,Total_Bad_Debt,Total_Good_Debt,Status
# #F,0,1,126000,Commercial associate,Higher education,Single / not married,House / apartment,Sales staff,52,7,9,5,0
# # inputs = np.array([[0, 0, 0, 0, 18, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
# # # inputs = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
# # pred = lr_model.predict(inputs)
# # print(pred)