import seaborn as sns
import matplotlib.pyplot as plt
import os
from typing import Optional
import pandas as pd
from matplotlib.axes import Axes

def plot_income_vs_dalys(df):
    sample_df = df.sample(5000, random_state=1)
    sns.regplot(data=sample_df, x='Per Capita Income (USD)', y='DALYs', scatter_kws={'alpha':0.2})
    plt.title("Income vs DALYs with Regression Line")
    plt.legend(["Regression Line"])
    plt.savefig("assets/income_regression.png")
    plt.clf()

def plot_education_vs_dalys(df):
    sample_df = df.sample(5000, random_state=42)
    sns.scatterplot(data=sample_df, x='Education Index', y='DALYs', hue='Disease Category')
    plt.title("DALYs vs Education Index")
    plt.savefig("assets/education_vs_dalys.png")
    plt.clf()

def plot_urbanization_vs_dalys(df):
    sample_df = df.sample(5000, random_state=42)
    sns.scatterplot(data=sample_df, x='Urbanization Rate (%)', y='DALYs', hue='Disease Category')
    plt.title("DALYs vs Urbanization Rate")
    plt.savefig("assets/urbanization_vs_dalys.png")
    plt.clf()

def plot_correlation_matrix(df):
    selected = df[['DALYs', 'Per Capita Income (USD)', 'Education Index', 'Urbanization Rate (%)']]
    corr = selected.corr()
    plt.figure(figsize=(8, 6))  
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".5f")
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.title("Correlation Matrix")
    plt.tight_layout() 
    plt.savefig("assets/correlation_matrix.png")
    plt.clf()

def plot_treatment_vs_dalys(df):
    sns.boxplot(data=df, x='Treatment Type', y='DALYs')
    plt.xticks(rotation=45)
    plt.title("DALYs by Treatment Type")
    plt.tight_layout()
    plt.savefig("assets/dalys_by_treatment.png")
    plt.clf()

def plot_country_vs_dalys(df):
    top = df.groupby('Country')['DALYs'].mean().sort_values(ascending=False).head(10)
    plt.figure(figsize=(8, 6))
    sns.barplot(x=top.values, y=top.index)
    plt.title("Top 10 Countries by Avg DALYs")
    plt.xlabel("Average DALYs")
    plt.tight_layout()
    plt.savefig("assets/top_countries_dalys.png")
    plt.clf()

from sklearn.linear_model import LinearRegression
import numpy as np

def run_regression(df,save_path):
    X = df[['Per Capita Income (USD)', 'Education Index', 'Urbanization Rate (%)']]
    y = df['DALYs']
    model = LinearRegression().fit(X, y)
    print("Regression Coefficients:", model.coef_)
    print("Intercept:", model.intercept_)
    print("R² Score:", model.score(X, y))
 
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w") as f:
        f.write("========== MISSING DATA REPORT ==========\n")
        if df is not None:
            f.write("\n\nAfter Cleaning:\n")
            f.write(str(df.isnull().sum()))
        else:
            f.write("Original Data Not Provided\n")
        f.write("\n\n")

        # Outlier section
        Q1 = df['DALYs'].quantile(0.25)
        Q3 = df['DALYs'].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        outlier_count = df[(df['DALYs'] < lower) | (df['DALYs'] > upper)].shape[0]

        f.write("========== OUTLIER REPORT ==========\n")
        f.write(f"Lower Bound: {lower:.2f}, Upper Bound: {upper:.2f}\n")
        f.write(f"Outliers Detected: {outlier_count} rows\n\n")

        # Regression section
        r2_score = model.score(X, y)
        f.write("========== REGRESSION RESULTS ==========\n")
        f.write(f"Coefficients: {model.coef_}\n")
        f.write(f"Intercept: {model.intercept_:.2f}\n")
        f.write(f"R² Score: {r2_score:.10f}  (scientific: {r2_score:.2e})\n")

    print(f"\nFull summary saved to: {save_path}")
    return model

from scipy.stats import pearsonr

def healthcare_correlation_summary(df, save_path):
    access = df['Healthcare Access (%)']
    beds = df['Hospital Beds per 1000']
    dalys = df['DALYs']

    # Pearson correlation
    corr_access, p_access = pearsonr(access, dalys)
    corr_beds, p_beds = pearsonr(beds, dalys)

    # Append to summary file
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "a") as f:
        f.write("\n========== HEALTHCARE INFRASTRUCTURE CORRELATIONS ==========\n")
        f.write(f"Healthcare Access vs DALYs:\n")
        f.write(f"  Correlation: {corr_access:.4f}, p-value: {p_access:.4e}\n")
        f.write(f"Hospital Beds per 1000 vs DALYs:\n")
        f.write(f"  Correlation: {corr_beds:.4f}, p-value: {p_beds:.4e}\n")

    print("Healthcare correlations added to:", save_path)


def plot_healthcare_vs_dalys(df):
    sample_df = df.sample(5000, random_state=42) 
    sns.scatterplot(data=sample_df, x='Doctors per 1000', y='DALYs', hue='Disease Category')
    plt.title("DALYs vs Doctor Availability")
    plt.savefig("assets/dalys_vs_doctors.png")
    plt.clf()

def plot_dalys_over_time(df):
    yearly = df.groupby('Year')['DALYs'].mean().reset_index()
    sns.lineplot(data=yearly, x='Year', y='DALYs')
    plt.title("Average DALYs Over Time")
    plt.savefig("assets/dalys_over_time.png")
    plt.clf()

def plot_dalys_over_time_by_income(df):
    df_grouped = df.groupby(['Year', 'Income Group'])['DALYs'].mean().reset_index()
    sns.lineplot(data=df_grouped, x='Year', y='DALYs', hue='Income Group')
    plt.title("DALYs Over Time by Income Group")
    plt.savefig("assets/dalys_time_income.png")
    plt.clf()

def plot_dalys_vs_hospital_beds(df):
    sample = df.sample(5000)
    sns.scatterplot(data=sample, x='Hospital Beds per 1000', y='DALYs')
    plt.title("DALYs vs Hospital Beds per 1000")
    plt.savefig("assets/dalys_vs_beds.png")
    plt.clf()

def plot_dalys_vs_access(df):
    sample = df.sample(5000)
    sns.scatterplot(data=sample, x='Healthcare Access (%)', y='DALYs')
    plt.title("DALYs vs Healthcare Access (%)")
    plt.savefig("assets/dalys_vs_access.png")
    plt.clf()
