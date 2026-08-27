 

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler


df = pd.read_csv("Titanic-Dataset.csv")


print("First 5 rows:")
print(df.head())


print("\nDataset Information:")
print(df.info())


print("\nMissing Values:")
print(df.isnull().sum())





if 'Age' in df.columns:
    df['Age'] = df['Age'].fillna(df['Age'].median())


if 'Fare' in df.columns:
    df['Fare'] = df['Fare'].fillna(df['Fare'].median())


if 'Embarked' in df.columns:
    df['Embarked'] = df['Embarked'].fillna(
        df['Embarked'].mode()[0]
    )


if 'Cabin' in df.columns:
    df = df.drop(columns=['Cabin'])

print("\nMissing Values After Cleaning:")
print(df.isnull().sum())




if 'Sex' in df.columns:
    df['Sex'] = df['Sex'].map({
        'male': 0,
        'female': 1
    })


if 'Embarked' in df.columns:
    df = pd.get_dummies(
        df,
        columns=['Embarked'],
        drop_first=True
    )

print("\nData After Encoding:")
print(df.head())




scaler = StandardScaler()

numerical_columns = []


for column in ['Age', 'Fare']:
    if column in df.columns:
        numerical_columns.append(column)


if numerical_columns:
    df[numerical_columns] = scaler.fit_transform(
        df[numerical_columns]
    )

print("\nData After Standardization:")
print(df.head())




plt.figure(figsize=(10, 5))

if 'Fare' in df.columns:
    sns.boxplot(x=df['Fare'])

plt.title("Boxplot for Fare")
plt.show()




if 'Fare' in df.columns:

    Q1 = df['Fare'].quantile(0.25)
    Q3 = df['Fare'].quantile(0.75)

    IQR = Q3 - Q1

    lower_limit = Q1 - 1.5 * IQR
    upper_limit = Q3 + 1.5 * IQR

    df = df[
        (df['Fare'] >= lower_limit) &
        (df['Fare'] <= upper_limit)
    ]

print("\nDataset Shape After Removing Outliers:")
print(df.shape)




print("\nFinal Dataset:")
print(df.head())

print("\nFinal Missing Values:")
print(df.isnull().sum())




df.to_csv(
    "cleaned_titanic_dataset.csv",
    index=False
)

print("\nPreprocessing Completed Successfully!")