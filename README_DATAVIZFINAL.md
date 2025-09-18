# Global Disease Burden Analyzer

A comprehensive, interactive Python application designed to explore the relationship between disease burden (measured in DALYs) and socioeconomic indicators such as income, education, urbanization, and healthcare infrastructure. This project offers both a CLI and GUI for data visualization and statistical analysis.

## Project Objective

To investigate how various socioeconomic factors (e.g., per capita income, education index, urbanization rate) correlate with disease burden across countries and demographics. This tool helps visualize disparities, perform statistical analysis, and support informed decision-making in public health policy.

## Research Questions

- How strongly do socioeconomic factors correlate with disease burden?
- Are certain demographics (age, gender) more affected?
- Do these correlations vary across disease categories (infectious vs. non-communicable)?
- How has disease burden changed over time across income groups?

## Features

- Interactive GUI with PyQt5 and Matplotlib
- Load and preprocess custom or default datasets
- 15+ visualization types including:
  - DALYs by gender, age group, disease category, and treatment
  - Correlation matrix of socioeconomic features
  - DALYs vs. income, education, urbanization, healthcare access
  - Trends over time and top-affected countries
- CLI mode with automated analysis and export of:
  - Cleaned datasets
  - Statistical regression and correlation reports
  - All visualizations to the `assets/` folder

## Project Structure

Global Disease Burden Analyzer
├── main.py                 # CLI/GUI entry point
├── gui.py                  # PyQt5 GUI logic
├── eda.py                  # Exploratory Data Analysis visuals
├── visualization.py        # Advanced statistical plots and regressions
├── preprocessing.py        # Data cleaning, transformation, and engineering
├── data/
│   ├── Global Health Statistics.csv   # Raw dataset
│   └── cleaned_data.csv               # Output after preprocessing
├── assets/                # Folder for saving generated plots
└── README.md              # Project documentation (this file)


## Usage

### Run CLI Analysis:

python main.py

This runs the full pipeline: preprocessing → regression → correlation → visualization.

### Launch GUI:

python main.py gui

This opens an interactive application for data exploration and plot export.

## Sample Visualizations

- DALYs Histogram
- Income vs DALYs Regression
- Education Index vs DALYs
- DALYs by Treatment Type
- DALYs Over Time by Income Group
- Correlation Matrix

## Statistical Analysis

- Regression Model: Analyzes linear relationships between DALYs and socioeconomic indicators.
- Pearson Correlation: Quantifies strength of association for healthcare variables.
- Outlier Detection: Uses IQR method to flag anomalies in DALY values.

## Data Preprocessing

- Imputation for missing values
- Z-score normalization for features
- Encoding of categorical disease types
- Feature engineering (Income Group, Disease Type)

## Dataset

- Source: Kaggle - Global Health Statistics
- Features include:
  - Country, Year, Age, Gender
  - Disease Category & Treatment Info
  - DALYs, Income, Education, Urbanization
  - Healthcare Access & Infrastructure

## Credits

- Developed by Sharmi Das and Jenny Huang as part of the DS8007 Final Project  
- Tools used: Python, Pandas, Seaborn, Matplotlib, Scikit-learn, PyQt5

## Future Work

- Add ML-based prediction for DALYs
- Integrate interactive dashboards using Plotly/Dash
- Deploy web version for public access
