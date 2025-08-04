import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np 
df = pd.read_csv('Titanic-Dataset.csv')
df.head(2)
# Name and ticket do not provide useful information for analysis, so we can drop them.
df.drop(['Name', 'Ticket'], axis=1, inplace=True)
for column in df.columns:
    if df[column].dtype == 'object':
        df[column] = df[column].astype('category')
        print(df[column].value_counts())
#No errors found in the dataset other than cabin.
#To make changes in columnn Cabin, we will only keep the values with one letter and two digits attached to it.
df_filtered = df[df['Cabin'].str.match(r'^[A-Za-z]\d{2}$', na=False)]
df_filtered['Cabin_letter'] = df_filtered['Cabin'].str[0]
df_filtered['Cabin_number'] = (df_filtered['Cabin'].str[1:3]).astype(int)
df_filtered.drop('Cabin', axis=1, inplace=True)
for column in df_filtered.columns:
    if df_filtered[column].dtype == 'category':
        print(df_filtered[column].value_counts())
#Finding the missing values in each column of the dataset
for column in df_filtered.columns:
    print(df_filtered[column].isnull().sum(), "missing values in", column)
#We shall fill the missing values in the 'Age' column with the mean of age and delete the missing rows in 'Embarked' column
#2 is negligible compared to the total number of rows
df_filtered['Age'].fillna(df_filtered['Age'].mean(), inplace=True)
df_filtered.dropna(subset=['Embarked'], inplace=True)
df_filtered.isnull().sum()
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
for column in df_filtered.columns:
    if df_filtered[column].dtype == 'category' or df_filtered[column].dtype == 'object':
        df_filtered[column] = le.fit_transform(df_filtered[column]).astype(int)
df_filtered.head(10)
        
#Special characters that are encoded or represent an ID need not be encoded. We shall scale down the age and fare columns
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
df_filtered[['Age', 'Fare']] = scaler.fit_transform(df_filtered[['Age', 'Fare']])
#Visualizing the data
plt.figure(figsize=(10, 6))
sns.countplot(x='Survived', data=df_filtered)
plt.title('Count of Survived vs Not Survived')
plt.xlabel('Survived')
plt.ylabel('Count')
plt.show()
#We don't know which data is more important so I will blindly clear the outliers in general
#I have considered the outliers for age also as someone with a very high age is not likely to have any survival chances (nor) likely to be an ship
sns.boxplot(x='Fare', data=df_filtered)
sns.boxplot(x='Age', data=df_filtered)
plt.show()
IQR = df_filtered['Fare'].quantile(0.75) - df_filtered['Fare'].quantile(0.25)
#Only upper bound outliers are present
upper_bound = df_filtered['Fare'].quantile(0.75) + 1.5 * IQR
df_filtered = df_filtered[df_filtered['Fare'] <= upper_bound]
IQR = df_filtered['Age'].quantile(0.75) - df_filtered['Age'].quantile(0.25)
#Only upper bound outliers are present
upper_bound = df_filtered['Age'].quantile(0.75) + 1.5 * IQR
df_filtered = df_filtered[df_filtered['Age'] <= upper_bound]
#check no. of rows after removing outliers
df_filtered.shape
df_filtered.to_csv('titanic_cleaned.csv', index=False)
print("Data cleaned and saved to 'titanic_cleaned.csv'.")