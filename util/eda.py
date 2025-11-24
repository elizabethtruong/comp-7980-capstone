'''
This script performs exploratory data analysis on the dataset.

Usage: python util/eda.py
'''

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning) 

# Read cleaned dataset into dataframe
df = pd.read_csv("data/Application_Data_Cleaned.csv")

# Prints basic dataset info prior to feature engineering
def print_dataset_info():
    print("First 5 rows of dataset:\n", df.head())
    print("Number of observations, features in dataset:\n", df.shape)
    print(df.info())
    print("Number of unique values for each feature:\n", df.nunique())
    print("Number of missing records for each feature:\n", df.isnull().sum())

# FEATURE ENGINEERING
# Feature reduction/data cleaning
def feature_engineering():
    # Drop Applicant_ID column
    df.drop(['Applicant_ID'], axis=1, inplace=True)
    # CONVERTING YUAN TO USD FOR TOTAL_INCOME
    # Print range of values from Total_Income (Yuan)
    print("Total income range (Yuan):", df['Total_Income'].min(), "-", df['Total_Income'].max())
    # Need to convert Yuan to USD since our model will take inputs in USD
    yuan_to_usd_exchange_rate = 0.14
    df['Total_Income'] = df['Total_Income'] * yuan_to_usd_exchange_rate
    df['Total_Income'] = df['Total_Income'].round().astype(int)
    # Print range of values from Total_Income (USD)
    print("Total income range (USD):", df['Total_Income'].min(), "-", df['Total_Income'].max())

    # ADJUSTING INCOME_TYPE VALUES
    # Print unique values for Income_Type
    print(df['Income_Type'].unique())
    # Rename Income_Type column to Employment_Status
    df.rename(columns={"Income_Type": "Employment_Status"}, inplace=True)
    # Rename values in Employment_Status
    df['Employment_Status'].replace({"Working": "Salaried employee", "Commercial associate": "Hourly/commission employee", "State servant": "Government employee"}, inplace=True)
    print(df['Employment_Status'].unique())

    # ADJUSTING EDUCATION_TYPE VALUES
    # Print unique values for Education_Type
    print(df['Education_Type'].unique())
    # Rename Education_Type column to Education_Completed
    df.rename(columns={"Education_Type": "Education_Completed"}, inplace=True)
    # Rename values in Education_Completed
    df['Education_Completed'].replace({"Secondary / secondary special": "High school", "Lower secondary": "Some high school", "Higher education": "Bachelor's degree", "Incomplete higher": "Some college", "Academic degree": "Master's degree or higher"}, inplace=True)
    # Print unique values for Education_Completed
    print(df['Education_Completed'].unique())

    # ADJUSTING FAMILY_STATUS VALUES
    # Print unique values for Family_Status
    print(df['Family_Status'].unique())
    # Rename values in Family_Status
    df['Family_Status'].replace({"Single / not married": "Single", "Civil marriage": "Married"}, inplace=True)
    # Print unique values for Family_Status
    print(df['Family_Status'].unique())

    # ADJUSTING HOUSING_TYPE VALUES
    # Print unique values for Housing_Type
    print(df['Housing_Type'].unique())
    # Rename values in Housing_Type
    df['Housing_Type'].replace({"House / apartment": "Homeowner", "Rented apartment": "Renting", "Municipal apartment": "Public housing", "Co-op apartment": "Homeowner", "Office apartment": "Renting"}, inplace=True)
    # Print unique values for Housing_Type
    print(df['Housing_Type'].unique())

    # DROPPING OWNED_X_PHONE COLUMNS
    # Print unique values for Owned_Mobile_Phone
    print(df['Owned_Mobile_Phone'].unique())
    # Drop Owned_Mobile_Phone column since only one unique value
    df.drop(['Owned_Mobile_Phone'], axis=1, inplace=True)
    # If everyone owns a mobile phone, we can drop the other phone columns
    df.drop(['Owned_Work_Phone', 'Owned_Phone'], axis=1, inplace=True)

    # DROPPING OWNED_EMAIL COLUMN
    # Count rows where 'Owned_Email' is 0
    target_value_int = 0
    target_column_int = 'Owned_Email'
    count_int = (df[target_column_int] == target_value_int).sum()
    print(f"Number of rows where '{target_column_int}' is {target_value_int}: {count_int}")
    # Drop Owned_Email column since having an email or not won't affect credit approval
    df.drop(['Owned_Email'], axis=1, inplace=True)

    # DROPPING JOB_TITLE COLUMN
    # Print unique values for Job_Title
    print(df['Job_Title'].unique())
    # Drop Job_Title column since job role won't affect credit approval
    df.drop(['Job_Title'], axis=1, inplace=True)

    # DROPPING FAMILY INFO COLUMNS
    # Print unique values for Total_Family_Members
    print(df['Total_Family_Members'].unique())
    # Drop Total_Family_Members column since number of family members won't affect credit approval
    df.drop(['Total_Family_Members'], axis=1, inplace=True)
    # Print unique values for Total_Children
    print(df['Total_Children'].unique())
    # Drop Total_Children column since number of children won't affect credit approval
    df.drop(['Total_Children'], axis=1, inplace=True)

    # DROPPING DEBT COLUMNS
    # Print unique values for Total_Bad_Debt
    print(df['Total_Bad_Debt'].unique())
    # Drop Total_Bad_Debt column since target variable is Status which already factors in Total_Bad_Debt
    df.drop(['Total_Bad_Debt'], axis=1, inplace=True)
    # Print unique values for Total_Good_Debt
    print(df['Total_Good_Debt'].unique())
    # Drop Total_Good_Debt column since target variable is Status which already factors in Total_Good_Debt
    df.drop(['Total_Good_Debt'], axis=1, inplace=True)

    # Print information about data in dataframe after adjusting/dropping columns
    print(df.info())

def visualization():
    # VISUALIZATION
    sns_palette = ['blue', 'orange']
    sns.countplot(data=df, x="Applicant_Gender", palette=sns_palette)
    plt.title("Number of Applicants per Gender")
    plt.ylabel("Number of Applicants")
    plt.show()

    sns.countplot(data=df, x="Owned_Car", palette=sns_palette)
    plt.title("Number of Applicants that Own a Car")
    plt.ylabel("Number of Applicants")
    plt.show()

    sns.countplot(data=df, x="Owned_Realty", palette=sns_palette)
    plt.title("Number of Applicants that Own Realty")
    plt.ylabel("Number of Applicants")
    plt.show()

    sns.countplot(data=df, x="Employment_Status", palette=sns_palette)
    plt.title("Number of Applicants per Employment Status")
    plt.ylabel("Number of Applicants")
    plt.show()

    sns.countplot(data=df, x="Education_Completed", palette=sns_palette)
    plt.title("Number of Applicants per Highest Education Completed")
    plt.ylabel("Number of Applicants")
    plt.show()

    sns.countplot(data=df, x="Family_Status", palette=sns_palette)
    plt.title("Number of Applicants per Family Status")
    plt.ylabel("Number of Applicants")
    plt.show()

    sns.countplot(data=df, x="Housing_Type", palette=sns_palette)
    plt.title("Number of Applicants per Housing Type")
    plt.ylabel("Number of Applicants")
    plt.show()

    sns.countplot(data=df, x="Status", palette=sns_palette)
    plt.title("Number of Applicants per Approval Status")
    plt.ylabel("Number of Applicants")
    plt.show()

    plt.figure(figsize=[8,5])
    sns.histplot(data=df,x="Total_Income",bins=50).set(title="Distribution of Total Income",ylabel="Number of Applicants")
    plt.show()

    plt.figure(figsize=[8,5])
    sns.histplot(data=df,x="Applicant_Age",bins=20).set(title="Distribution of Applicant Age",ylabel="Number of Applicants")
    plt.show()

    plt.figure(figsize=[8,5])
    sns.histplot(data=df,x="Years_of_Working",bins=20).set(title="Distribution of Years Worked",ylabel="Number of Applicants")
    plt.show()

def convert_to_dummy(df):
    # CONVERT NON-NULL OBJECTS INTO DUMMY VARS
    # Change Applicant_Gender from M/F to 0/1 respectively
    df['Applicant_Gender'] = df['Applicant_Gender'].replace({'M': 0, 'F': 1})
    df['Applicant_Gender'] = df['Applicant_Gender'].astype(int)
    # Convert Employment_Status into dummy vars
    Employment_Status = pd.get_dummies(df['Employment_Status'], drop_first=False, dtype=int)
    # Convert Family_Status into dummy vars
    Family_Status = pd.get_dummies(df['Family_Status'], drop_first=False, dtype=int)
    # Convert Education_Completed into dummy vars
    Education_Completed = pd.get_dummies(df['Education_Completed'], drop_first=False, dtype=int)
    # Convert Housing_Type into dummy vars
    Housing_Type = pd.get_dummies(df['Housing_Type'], drop_first=False, dtype=int)

    df.drop(['Employment_Status', 'Family_Status', 'Education_Completed', 'Housing_Type'], axis=1, inplace=True)
    df = pd.concat([df, Employment_Status, Family_Status, Education_Completed, Housing_Type], axis=1)

    print(df.info())
    print(df.head())
    # Convert final df to csv for model
    df.to_csv('data/Application_Data_Final.csv', index=False)

if __name__ == "__main__":
    print_dataset_info()
    feature_engineering()
    convert_to_dummy(df)