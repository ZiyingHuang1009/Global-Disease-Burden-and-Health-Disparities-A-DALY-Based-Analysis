import os
import sys
from pathlib import Path
from PyQt5.QtWidgets import QApplication

def run_cli_analysis(data_path=None):
    try:
        from preprocessing import load_data, preprocess_data, save_cleaned_data
        from visualization import (
            plot_income_vs_dalys,
            plot_education_vs_dalys,
            plot_urbanization_vs_dalys,
            plot_correlation_matrix,
            plot_treatment_vs_dalys,
            plot_country_vs_dalys,
            run_regression,
            plot_healthcare_vs_dalys,
            plot_dalys_over_time,
            plot_dalys_over_time_by_income,
            plot_dalys_vs_hospital_beds,
            plot_dalys_vs_access,
            healthcare_correlation_summary
        )
        from eda import (
            plot_dalys_histogram,
            plot_dalys_by_gender,
            plot_dalys_by_age_group,
            plot_dalys_by_category,
            plot_dalys_by_disease_type
        )

        print("\n=== Starting Analysis ===")
        
        # Handle data path
        if data_path is None:
            data_path = Path(__file__).parent / "data" / "Global Health Statistics.csv"
        data_path = Path(data_path).absolute()
        
        if not data_path.exists():
            raise FileNotFoundError(f"Data file not found at: {data_path}")
        
        # Create assets path
        assets_dir = Path(__file__).parent / "assets"

        # Data pipeline
        print(f"\nLoading data from: {data_path}")
        df = load_data(data_path)
        df_cleaned = preprocess_data(df)
        save_cleaned_data(df_cleaned, 'data/cleaned_data.csv')

        # Analysis
        print("\nRunning statistical analysis...")
        run_regression(df_cleaned, save_path='data/analysis_results.txt')
        healthcare_correlation_summary(df_cleaned, save_path='data/analysis_results.txt')

        # Visualizations
        print("\nGenerating visualizations...")
        plot_income_vs_dalys(df_cleaned)
        plot_education_vs_dalys(df_cleaned)
        plot_urbanization_vs_dalys(df_cleaned)
        plot_correlation_matrix(df_cleaned)
        plot_treatment_vs_dalys(df_cleaned)
        plot_country_vs_dalys(df_cleaned)
        plot_healthcare_vs_dalys(df_cleaned) # type: ignore
        plot_dalys_over_time(df_cleaned) # type: ignore
        plot_dalys_over_time_by_income(df_cleaned)
        plot_dalys_vs_hospital_beds(df_cleaned)
        plot_dalys_vs_access(df_cleaned)

        print("\n=== Analysis Complete ===")
        print("Results saved to:")
        print(f"- Cleaned data: data/cleaned_data.csv")
        print(f"- Analysis results: data/analysis_results.txt")
        print(f"- Visualizations: assets/")
        return True

    except Exception as e:
        print(f"\n!!! Analysis Failed !!!\nError: {str(e)}", file=sys.stderr)
        return False

def run_gui():
    try:
        from gui import MainWindow 
        app = QApplication(sys.argv)
        window = MainWindow()
        window.show()
        sys.exit(app.exec_())
    except Exception as e:
        print(f"Failed to launch GUI: {str(e)}", file=sys.stderr)
        return False

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == 'gui':
        run_gui()
    else:
        if run_cli_analysis():
            # Auto-launch GUI after successful analysis
            run_gui()