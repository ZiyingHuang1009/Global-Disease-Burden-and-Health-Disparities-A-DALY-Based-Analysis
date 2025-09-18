import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

def plot_dalys_histogram(df):
    sns.histplot(df['DALYs'], bins=30, kde=True)
    plt.title("Distribution of DALYs")
    plt.xlabel("DALYs")
    plt.savefig("assets/dalys_histogram.png")
    plt.clf()

def plot_dalys_by_gender(df):
    sns.barplot(data=df, x='Gender', y='DALYs', estimator='mean')
    plt.title("Average DALYs by Gender")
    plt.savefig("assets/dalys_by_gender.png")
    plt.clf()

def plot_dalys_by_age_group(df):
    sns.boxplot(data=df, x='Age Group', y='DALYs')
    plt.title("DALYs by Age Group")
    plt.savefig("assets/dalys_by_age_group.png")
    plt.clf()

def plot_dalys_by_category(df):
    sns.barplot(data=df, x='Disease Category', y='DALYs', estimator='mean')
    plt.xticks(rotation=45)
    plt.title("Average DALYs by Disease Category")
    plt.tight_layout()
    plt.savefig("assets/dalys_by_category.png")
    plt.clf()

def plot_dalys_by_disease_type(df):
    communicable = ['Parasitic', 'Viral', 'Bacterial', 'Infectious']
    df['Disease Type'] = df['Disease Category'].apply(
        lambda x: 'Infectious' if x in communicable else 'Non-Communicable'
    )
    sns.boxplot(data=df, x='Disease Type', y='DALYs')
    plt.title("DALYs by Disease Type")
    plt.savefig("assets/dalys_by_disease_type.png")
    plt.clf()
