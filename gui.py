import sys
import os
from pathlib import Path
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                            QHBoxLayout, QPushButton, QComboBox, QLabel, 
                            QFileDialog, QMessageBox, QSizePolicy, QStyle)
from PyQt5.QtCore import Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

project_root = Path(__file__).parent.parent.resolve()
sys.path.append(str(project_root))

try:
    from visualization import (
        plot_income_vs_dalys,
        plot_education_vs_dalys,
        plot_urbanization_vs_dalys,
        plot_correlation_matrix,
        plot_treatment_vs_dalys,
        plot_country_vs_dalys,
        plot_healthcare_vs_dalys,
        plot_dalys_over_time,
        plot_dalys_over_time_by_income,
        plot_dalys_vs_hospital_beds,
        plot_dalys_vs_access
    )
    from eda import (
        plot_dalys_histogram,
        plot_dalys_by_gender,
        plot_dalys_by_age_group,
        plot_dalys_by_category,
        plot_dalys_by_disease_type
    )
    from preprocessing import (
        load_data,
        preprocess_data,
        save_cleaned_data
    )
except ImportError as e:
    print(f"Import error: {e}")
    print("Current Python path:", sys.path)
    raise

class TitleBar(QWidget):
    def __init__(self, parent):
        super().__init__(parent)
        self.parent = parent
        self.setup_ui()

    def setup_ui(self):
        layout = QHBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)
        layout.setSpacing(5)

        # Title label
        self.title = QLabel("Global Disease Burden Analyzer")
        self.title.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.title.setAlignment(Qt.AlignCenter)
        
        # Window control buttons
        self.minimize_btn = QPushButton()
        self.minimize_btn.setIcon(self.style().standardIcon(QStyle.SP_TitleBarMinButton))
        self.minimize_btn.clicked.connect(self.parent.showMinimized)
        
        self.maximize_btn = QPushButton()
        self.maximize_btn.setIcon(self.style().standardIcon(QStyle.SP_TitleBarMaxButton))
        self.maximize_btn.clicked.connect(self.toggle_maximize)
        
        self.close_btn = QPushButton()
        self.close_btn.setIcon(self.style().standardIcon(QStyle.SP_TitleBarCloseButton))
        self.close_btn.clicked.connect(self.parent.close)

        # Add widgets to layout
        layout.addWidget(self.title)
        layout.addWidget(self.minimize_btn)
        layout.addWidget(self.maximize_btn)
        layout.addWidget(self.close_btn)

        # Styling
        self.setStyleSheet("""
            TitleBar {
                background-color: #2c3e50;
                padding: 3px;
                height: 30px;
            }
            QLabel {
                color: white;
                font-weight: bold;
                font-size: 12px;
            }
            QPushButton {
                background: transparent;
                border: none;
                padding: 0px;
                min-width: 20px;
                max-width: 20px;
                min-height: 20px;
                max-height: 20px;
            }
            QPushButton:hover {
                background: rgba(255, 255, 255, 0.2);
                border-radius: 4px;
            }
            QPushButton#close_btn:hover {
                background: #e74c3c;
            }
        """)
        self.close_btn.setObjectName("close_btn")

    def toggle_maximize(self):
        if self.parent.isMaximized():
            self.parent.showNormal()
            self.maximize_btn.setIcon(self.style().standardIcon(QStyle.SP_TitleBarMaxButton))
        else:
            self.parent.showMaximized()
            self.maximize_btn.setIcon(self.style().standardIcon(QStyle.SP_TitleBarNormalButton))

class MplCanvas(FigureCanvas):
    def __init__(self, parent=None, width=10, height=8, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        super().__init__(self.fig)
        self.axes = self.fig.add_subplot(111)
        
    def clear(self):
        self.fig.clf()
        self.axes = self.fig.add_subplot(111)
        self.draw()

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Global Disease Burden Analyzer")
        self.setGeometry(100, 100, 1200, 900)
        
        # Data storage
        self.data = None
        self.current_plot = None
        
        # Create main widget and layout
        self.main_widget = QWidget()
        self.setCentralWidget(self.main_widget)
        self.layout = QVBoxLayout()
        self.main_widget.setLayout(self.layout)
        
        # Initialize UI components
        self.init_ui()
        self.add_stats_panel()
        
    def init_ui(self):
        # Control panel
        control_panel = QWidget()
        control_layout = QHBoxLayout()
        control_panel.setLayout(control_layout)
        
        # Data upload button
        self.upload_btn = QPushButton("Upload Data")
        self.upload_btn.clicked.connect(self.upload_data)
        control_layout.addWidget(self.upload_btn)

        # Data load button
        self.load_btn = QPushButton("Load Data")
        self.load_btn.clicked.connect(self.load_data)
        control_layout.addWidget(self.load_btn)
        
        self.load_btn.setToolTip("Load pre-cleaned data (from data/cleaned_data.csv)")
        self.upload_btn.setToolTip("Upload and preprocess a new CSV file")
        
        # Visualization selector
        self.plot_selector = QComboBox()
        self.plot_selector.addItems([
            "Select Visualization",
            "DALYs Histogram",
            "DALYs by Gender",
            "DALYs by Age Group",
            "DALYs by Disease Category",
            "DALYs by Disease Type",
            "Income vs DALYs",
            "Education vs DALYs",
            "Urbanization vs DALYs",
            "Correlation Matrix",
            "DALYs by Treatment",
            "Top Countries by DALYs",
            "DALYs vs Doctors",
            "DALYs Over Time",
            "DALYs Over Time by Income",
            "DALYs vs Hospital Beds",
            "DALYs vs Healthcare Access"
        ])
        control_layout.addWidget(QLabel("Choose Plot:"))
        control_layout.addWidget(self.plot_selector)
        
        # Plot button
        self.plot_btn = QPushButton("Generate Plot")
        self.plot_btn.clicked.connect(self.generate_plot)
        self.plot_btn.setEnabled(False)
        control_layout.addWidget(self.plot_btn)
        
        # Export button
        self.export_btn = QPushButton("Export Plot")
        self.export_btn.clicked.connect(self.export_plot)
        self.export_btn.setEnabled(False)
        control_layout.addWidget(self.export_btn)
        
        self.layout.addWidget(control_panel)
        
        # Matplotlib canvas
        self.canvas = MplCanvas(self)
        self.layout.addWidget(self.canvas)
        
        # Status bar
        self.statusBar().showMessage("Ready")
    
    def add_stats_panel(self):
        self.stats_panel = QWidget()
        stats_layout = QVBoxLayout()
        self.stats_panel.setLayout(stats_layout)
        
        self.stats_label = QLabel("Statistical summary will appear here")
        self.stats_label.setWordWrap(True)
        stats_layout.addWidget(self.stats_label)
        
        self.layout.addWidget(self.stats_panel)
    
    def upload_data(self):
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Data File",
            "",
            "CSV Files (*.csv);;All Files (*)",
            options=options
        )
        
        if file_path:
            try:
                # Use existing preprocessing pipeline
                raw_df = load_data(file_path)
                self.data = preprocess_data(raw_df)
                
                # Auto-save cleaned data
                cleaned_path = os.path.join(project_root, "data", "cleaned_data.csv")
                save_cleaned_data(self.data, cleaned_path)
                
                self.statusBar().showMessage(f"Uploaded and processed: {os.path.basename(file_path)}")
                self.plot_btn.setEnabled(True)
                
            except Exception as e:
                QMessageBox.critical(self, "Upload Failed", f"Error processing file:\n{str(e)}")

    def load_data(self):
        try:
            
            cleaned_path = os.path.join(project_root, "data", "cleaned_data.csv")
            if os.path.exists(cleaned_path):
                self.data = pd.read_csv(cleaned_path)
                self.statusBar().showMessage("Loaded preprocessed data")
            else:
                # Fallback to sample data
                self.data = self.load_sample_data()
                
            self.plot_btn.setEnabled(True)
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Data loading failed: {str(e)}")


    def generate_plot(self):
        if self.data is None:
            QMessageBox.warning(self, "Warning", "Please load data first")
            return
            
        plot_type = self.plot_selector.currentText()
        
        try:
            # Clear previous plot
            self.canvas.clear()
            
            # Generate the selected plot
            if plot_type == "DALYs Histogram":
                sns.histplot(self.data['DALYs'], bins=30, kde=True, ax=self.canvas.axes)
                self.canvas.axes.set_title("Distribution of DALYs")
                
            elif plot_type == "DALYs by Gender":
                sns.barplot(data=self.data, x='Gender', y='DALYs', 
                           estimator='mean', ax=self.canvas.axes)
                self.canvas.axes.set_title("Average DALYs by Gender")
                
            elif plot_type == "DALYs by Age Group":
                sns.boxplot(data=self.data, x='Age Group', y='DALYs', ax=self.canvas.axes)
                self.canvas.axes.set_title("DALYs by Age Group")
                
            elif plot_type == "DALYs by Disease Category":
                sns.barplot(data=self.data, x='Disease Category', y='DALYs', 
                           estimator='mean', ax=self.canvas.axes)
                self.canvas.axes.set_title("Average DALYs by Disease Category")
                self.canvas.axes.tick_params(axis='x', rotation=45)
                
            elif plot_type == "DALYs by Disease Type":
                communicable = ['Parasitic', 'Viral', 'Bacterial', 'Infectious']
                self.data['Disease Type'] = self.data['Disease Category'].apply(
                    lambda x: 'Infectious' if x in communicable else 'Non-Communicable'
                )
                sns.boxplot(data=self.data, x='Disease Type', y='DALYs', ax=self.canvas.axes)
                self.canvas.axes.set_title("DALYs by Disease Type")
                
            elif plot_type == "Income vs DALYs":
                plot_income_vs_dalys(self.data, ax=self.canvas.axes)
                
            elif plot_type == "Education vs DALYs":
                plot_education_vs_dalys(self.data, ax=self.canvas.axes)
                
            elif plot_type == "Urbanization vs DALYs":
                plot_urbanization_vs_dalys(self.data, ax=self.canvas.axes)
                
            elif plot_type == "Correlation Matrix":
                plot_correlation_matrix(self.data, ax=self.canvas.axes)
                
            elif plot_type == "DALYs by Treatment":
                plot_treatment_vs_dalys(self.data, ax=self.canvas.axes)
                
            elif plot_type == "Top Countries by DALYs":
                plot_country_vs_dalys(self.data, ax=self.canvas.axes)
                
            elif plot_type == "DALYs vs Doctors":
                plot_healthcare_vs_dalys(self.data, ax=self.canvas.axes)
                
            elif plot_type == "DALYs Over Time":
                plot_dalys_over_time(self.data, ax=self.canvas.axes)
            
            elif plot_type == "DALYs Over Time by Income":
                plot_dalys_over_time_by_income(self.data, ax=self.canvas.axes)
                
            elif plot_type == "DALYs vs Hospital Beds":
                plot_dalys_vs_hospital_beds(self.data, ax=self.canvas.axes)
                
            elif plot_type == "DALYs vs Healthcare Access":
                plot_dalys_vs_access(self.data, ax=self.canvas.axes)
                
            else:
                QMessageBox.warning(self, "Warning", "Please select a valid plot type")
                return
                
            # Adjust layout and draw
            self.canvas.fig.tight_layout()
            self.canvas.draw()
            self.current_plot = plot_type
            self.export_btn.setEnabled(True)
            self.statusBar().showMessage(f"Generated: {plot_type}")
            
            # Update statistics panel
            self.update_stats_panel(plot_type)
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Plot failed: {str(e)}")
    
    def update_stats_panel(self, plot_type):
        stats_text = f"<b>Analysis of {plot_type}:</b><br>"
        
        try:
            if plot_type in ["Income vs DALYs", "Education vs DALYs", "Urbanization vs DALYs"]:
                x_col = {
                    "Income vs DALYs": "Per Capita Income (USD)",
                    "Education vs DALYs": "Education Index",
                    "Urbanization vs DALYs": "Urbanization Rate (%)"
                }[plot_type]
                
                corr, p = pearsonr(self.data[x_col], self.data['DALYs'])
                stats_text += f"Pearson correlation: {corr:.5f} (p={p:.3e})<br>"
            
            
            stats_text += (
                f"<br><b>Global Statistics:</b><br>"
                f"Mean DALYs: {self.data['DALYs'].mean():.1f}<br>"
                f"Median DALYs: {self.data['DALYs'].median():.1f}<br>"
                f"Std Dev: {self.data['DALYs'].std():.1f}"
            )
            
        except Exception as e:
            stats_text += f"<br>Statistical analysis failed: {str(e)}"
        
        self.stats_label.setText(stats_text)
    
    def export_plot(self):
        if not self.current_plot:
            return
            
        # Create assets directory 
        assets_dir = os.path.join(project_root, "assets")
        os.makedirs(assets_dir, exist_ok=True)
        
        # Suggest filename
        default_name = os.path.join(
            assets_dir,
            f"{self.current_plot.lower().replace(' ', '_')}.png"
        )
        
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Plot",
            default_name,
            "PNG Files (*.png);;PDF Files (*.pdf);;All Files (*)",
            options=options
        )
        
        if file_path:
            try:
                self.canvas.fig.savefig(file_path, bbox_inches='tight')
                self.statusBar().showMessage(f"Plot saved to {file_path}")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to save: {str(e)}")

def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()