#Taking the cleaned data from the previous task as input
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
df = pd.read_csv('titanic_cleaned.csv')
#Making smaller plots and defining two plots side by side
df.head(10)
df_plotting = df[['Survived', 'Pclass', 'Sex', 'Age', 'SibSp', 'Parch',
       'Fare', 'Embarked', 'Cabin_letter']]
plt.figure(figsize=(6, 4))
plt.subplot(1, 2, 1)
for col in df_plotting.columns:
    plt.hist(df_plotting[col])
    plt.title(f'Count of {col}')
    plt.xticks(rotation=45)
    plt.show()
sns.boxplot(df_plotting)
plt.show()
"""
Categorical columns: 'Survived', 'Pclass', 'Sex', 'SibSp', 'Parch', 'Embarked', 'Cabin_letter', 'Cabin_number'
Age: Gaussian distribution (slightly right-skewed)
Fare: Non-Gaussian distribution (random)

"""
