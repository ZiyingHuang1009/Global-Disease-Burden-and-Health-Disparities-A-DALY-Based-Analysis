import pandas as pd
import numpy as np
import os

def load_data(filepath):
    df = pd.read_csv(filepath)
    print(" Initial Data Overview:")
    print(df.info())
    return df

def preprocess_data(df):
    print("\n Missing Values Before Cleaning:")
    print(df.isnull().sum())

    # Handling Missing Values: Impute or Drop
    df['Education Index'] = df['Education Index'].fillna(df['Education Index'].median())
    df['Urbanization Rate (%)'] = df['Urbanization Rate (%)'].fillna(df['Urbanization Rate (%)'].median())
    df = df.dropna(subset=['DALYs', 'Per Capita Income (USD)'])

    print("\n Missing Values After Imputation:")
    print(df.isnull().sum())

    # Data Cleaning
    df = df.drop_duplicates()
    df['Year'] = df['Year'].astype(int)

    # Outlier Detection: Using IQR for DALYs
    Q1 = df['DALYs'].quantile(0.25)
    Q3 = df['DALYs'].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    outliers = df[(df['DALYs'] < lower) | (df['DALYs'] > upper)]
    print(f"\n⚠️ Outliers in DALYs: {len(outliers)} rows")


    # Data Transformation: Normalization (Z-score)
    df['Income_Norm'] = (df['Per Capita Income (USD)'] - df['Per Capita Income (USD)'].mean()) / df['Per Capita Income (USD)'].std()
    df['Education_Norm'] = (df['Education Index'] - df['Education Index'].mean()) / df['Education Index'].std()
    df['Urbanization_Norm'] = (df['Urbanization Rate (%)'] - df['Urbanization Rate (%)'].mean()) / df['Urbanization Rate (%)'].std()

    # Feature Engineering
    communicable = ['Parasitic', 'Viral', 'Bacterial', 'Infectious']
    df['Disease Type'] = df['Disease Category'].apply(lambda x: 'Infectious' if x in communicable else 'Non-Communicable')
    df['Income Group'] = pd.qcut(df['Per Capita Income (USD)'], 3, labels=['Low', 'Medium', 'High'])

    return df

def save_cleaned_data(df, path='data/cleaned_data.csv'):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    print(f"\nCleaned data saved to: {path}")
