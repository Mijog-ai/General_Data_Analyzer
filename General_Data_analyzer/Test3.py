import sys
import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from huggingface_hub import login
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QPushButton, QComboBox, QFileDialog, QLabel,
                             QGroupBox, QGridLayout, QCheckBox, QMessageBox, QTextEdit,
                             QListWidget, QAbstractItemView, QSpinBox, QDoubleSpinBox, QTableWidgetItem, QTableWidget,
                             QLineEdit, QTabWidget)
from PyQt5.QtCore import Qt
from scipy.ndimage import gaussian_filter1d
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
from PyQt5.QtWidgets import QComboBox, QTextEdit, QPushButton, QLabel



class DataAnalyzer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Data Analyzer")
        self.setGeometry(100, 100, 1200,1000)

        # Initialize variables
        self.df = None
        self.filename = None
        self.last_modified_time = None
        self.colors = ['b', 'r', 'g', 'c', 'm', 'y', 'k']
        self.cursor_x = None  # Stores X position of cursor

        self.setup_ui()

    def setup_ui(self):
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QVBoxLayout(main_widget)

        self.tabs = QTabWidget()
        layout.addWidget(self.tabs)

        self.tab1 = QWidget()
        self.tab2 = QWidget()
        self.tabs.addTab(self.tab1, "Data Analysis")
        self.tabs.addTab(self.tab2, "AI-Anomaly_Detector")

        self.setup_tab1()
        self.setup_tab2()

    def setup_tab2(self):
        layout = QVBoxLayout(self.tab2)

        self.description_label = QLabel("Pump Model Selection & Anomaly Detection")
        layout.addWidget(self.description_label)

        # Dropdown menu for pump models
        self.pump_dropdown = QComboBox()
        self.pump_dropdown.addItem("Fetching pump models...")  # Placeholder
        layout.addWidget(self.pump_dropdown)

        # Button to analyze selected pump model
        self.analyze_button = QPushButton("Analyze Pump Data")
        self.analyze_button.clicked.connect(self.analyze_pump_model)
        layout.addWidget(self.analyze_button)

        # Output area for anomaly results
        self.anomaly_results = QTextEdit()
        self.anomaly_results.setReadOnly(True)
        self.anomaly_results.setPlaceholderText("Anomaly detection results will appear here...")
        layout.addWidget(self.anomaly_results)

        # Fetch pump models in the background
        self.fetch_pump_models()

    def setup_tab1(self):
        layout = QVBoxLayout(self.tab1)
        h_layout = QHBoxLayout()

        # Left panel
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)

        # File selection
        file_group = QGroupBox("File Selection")
        file_layout = QVBoxLayout()



        self.btn_file = QPushButton("Select File")
        self.btn_file.clicked.connect(self.load_file)
        file_layout.addWidget(self.btn_file)

        self.file_label = QLabel("No file selected")
        self.file_label.setStyleSheet("color: gray; font-style: italic;")
        self.file_label.setWordWrap(True)
        file_layout.addWidget(self.file_label)

        # Add data preview
        self.data_preview = QTextEdit()
        self.data_preview.setReadOnly(True)
        self.data_preview.setMaximumHeight(150)
        file_layout.addWidget(self.data_preview)

        file_group.setLayout(file_layout)
        left_layout.addWidget(file_group)

        # Plot controls
        plot_group = QGroupBox("Plot Controls")
        plot_layout = QGridLayout()

        plot_layout.addWidget(QLabel("X-Axis:"), 0, 0)
        self.x_combo = QComboBox()
        plot_layout.addWidget(self.x_combo, 0, 1)

        plot_layout.addWidget(QLabel("Y-Axes:"), 1, 0)
        self.y_list = QListWidget()
        self.y_list.setSelectionMode(QAbstractItemView.MultiSelection)
        plot_layout.addWidget(self.y_list, 1, 1)



        # Add Export Button
        self.export_btn = QPushButton("Export Data and Plot")
        self.export_btn.clicked.connect(self.export_plot_and_data)
        left_layout.addWidget(self.export_btn)

        # Range filters
        # X-Range
        plot_layout.addWidget(QLabel("X-Range:"), 2, 0)
        x_range_layout = QGridLayout()

        self.x_min_label = QLabel("Min:")
        x_range_layout.addWidget(self.x_min_label, 0, 0)
        self.x_min = QDoubleSpinBox()
        self.x_min.setDecimals(2)
        self.x_min.setRange(-1e6, 1e6)
        self.x_min.setValue(-1e6)
        x_range_layout.addWidget(self.x_min, 0, 1)

        self.x_max_label = QLabel("Max:")
        x_range_layout.addWidget(self.x_max_label, 0, 2)
        self.x_max = QDoubleSpinBox()
        self.x_max.setDecimals(2)
        self.x_max.setRange(-1e6, 1e6)
        self.x_max.setValue(1e6)
        x_range_layout.addWidget(self.x_max, 0, 3)

        plot_layout.addLayout(x_range_layout, 2, 1)

        # Y-Range
        plot_layout.addWidget(QLabel("Y-Range:"), 3, 0)
        y_range_layout = QGridLayout()

        self.y_min_label = QLabel("Min:")
        y_range_layout.addWidget(self.y_min_label, 0, 0)
        self.y_min = QDoubleSpinBox()
        self.y_min.setDecimals(2)
        self.y_min.setRange(-1e6, 1e6)
        self.y_min.setValue(-1e6)
        y_range_layout.addWidget(self.y_min, 0, 1)

        self.y_max_label = QLabel("Max:")
        y_range_layout.addWidget(self.y_max_label, 0, 2)
        self.y_max = QDoubleSpinBox()
        self.y_max.setDecimals(2)
        self.y_max.setRange(-1e6, 1e6)
        self.y_max.setValue(1e6)
        y_range_layout.addWidget(self.y_max, 0, 3)

        plot_layout.addLayout(y_range_layout, 3, 1)

        # Add Title Input
        title_group = QGroupBox("Plot Title")
        title_layout = QHBoxLayout()
        self.title_input = QLineEdit()
        self.title_input.setPlaceholderText("Enter plot title here")
        self.title_input.textChanged.connect(self.update_plot_title)  # Connect signal
        title_layout.addWidget(self.title_input)
        title_group.setLayout(title_layout)
        plot_layout.addWidget(title_group, 5, 0, 1, 2)

        # Add Reset Button
        self.reset_btn = QPushButton("Reset Ranges")
        self.reset_btn.clicked.connect(self.reset_ranges)
        plot_layout.addWidget(self.reset_btn, 6, 0, 1, 2)  # Move Reset button to its own row

        # Add grid and legend options
        options_layout = QHBoxLayout()
        self.grid_cb = QCheckBox("Show Grid")
        self.grid_cb.setChecked(True)
        options_layout.addWidget(self.grid_cb)

        self.legend_cb = QCheckBox("Show Legend")
        self.legend_cb.setChecked(True)
        options_layout.addWidget(self.legend_cb)

        plot_layout.addLayout(options_layout, 4, 0, 1, 2)

        plot_group.setLayout(plot_layout)
        left_layout.addWidget(plot_group)

        # Smoothing options
        plot_layout.addWidget(QLabel("Smoothing Filter:"), 7, 0)
        self.smoothing_combo = QComboBox()
        self.smoothing_combo.addItems(["None", "Gaussian"])  # Add more filters if needed
        plot_layout.addWidget(self.smoothing_combo, 7, 1)

        # Gaussian smoothing options
        gaussian_group = QGroupBox("Gaussian Filter Settings")
        gaussian_layout = QGridLayout()
        gaussian_layout.addWidget(QLabel("Sigma:"), 0, 0)
        self.sigma_spin = QDoubleSpinBox()
        self.sigma_spin.setRange(0.1, 50)
        self.sigma_spin.setValue(1.0)
        gaussian_layout.addWidget(self.sigma_spin, 0, 1)

        gaussian_group.setLayout(gaussian_layout)
        plot_layout.addWidget(gaussian_group, 8, 0, 1, 2)

        # Statistics section
        stats_group = QGroupBox("Statistics")
        stats_layout = QVBoxLayout()
        self.stats_table = QTableWidget()
        stats_layout.addWidget(self.stats_table)
        stats_group.setLayout(stats_layout)
        left_layout.addWidget(stats_group)

        # Plot button
        self.plot_btn = QPushButton("Create Plot")
        self.plot_btn.clicked.connect(self.create_plot)
        left_layout.addWidget(self.plot_btn)

        left_layout.addStretch()
        h_layout.addWidget(left_panel, stretch=1)

        # Right panel for plot
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)

        self.figure, self.ax = plt.subplots(figsize=(10, 8))
        self.canvas = FigureCanvas(self.figure)
        right_layout.addWidget(self.canvas)

        self.toolbar = NavigationToolbar(self.canvas, self)
        right_layout.addWidget(self.toolbar)

        # Initialize vertical cursor (Hidden initially)
        self.cursor_line = self.ax.axvline(x=0, color='r', linestyle='--', linewidth=1.5)
        self.cursor_line.set_visible(False)  # Hide initially

        # Initialize text for cursor (Hidden initially)
        self.cursor_text = self.ax.text(0, 0, "", fontsize=10, color="black",
                                        bbox=dict(facecolor="white", alpha=0.7))
        self.cursor_text.set_visible(False)

        self.canvas.mpl_connect("button_press_event", self.on_click)
        self.canvas.mpl_connect("motion_notify_event", self.on_mouse_drag)

        h_layout.addWidget(right_panel, stretch=2)
        layout.addLayout(h_layout)

    def update_plot_title(self):
        """Update the plot title dynamically based on the input."""
        plot_title = self.title_input.text().strip()
        self.ax.set_title(plot_title)
        self.canvas.draw()

    def load_file(self):
        filename, _ = QFileDialog.getOpenFileName(
            self, "Select File", "", "CSV Files (*.csv);;Text Files (*.txt);;All Files (*.*)"
        )
        if filename:
            try:
                # Read the file using semicolon as the delimiter
                self.df = pd.read_csv(filename, delimiter=';', decimal=',')
                self.filename = filename
                self.preprocess_headers()
                self.update_ui()

                base_name = os.path.basename(filename)
                self.file_label.setText(f"Loaded File: {base_name}")
                self.file_label.setStyleSheet("color: black; font-weight: bold;")

                self.fetch_pump_models()

            except Exception as e:
                QMessageBox.warning(self, "Error", f"Failed to load file: {e}")


    def preprocess_headers(self):
        if self.df is not None:
            # Simplify column names if they include descriptions
            self.df.columns = [col.split(';')[0].strip() for col in self.df.columns]



    def reset_ranges(self):
        if self.df is not None:
            # Reset X and Y ranges to dataset's min and max values
            x_col = self.x_combo.currentText()
            if x_col:
                self.x_min.setValue(self.df[x_col].min())
                self.x_max.setValue(self.df[x_col].max())

            y_cols = [item.text() for item in self.y_list.selectedItems()]
            if y_cols:
                y_min_global = min(self.df[y_col].min() for y_col in y_cols)
                y_max_global = max(self.df[y_col].max() for y_col in y_cols)
                self.y_min.setValue(y_min_global)
                self.y_max.setValue(y_max_global)

            QMessageBox.information(self, "Ranges Reset", "Ranges have been reset to the default values.")
        else:
            QMessageBox.warning(self, "No Data", "Please load a file first.")

    def export_plot_and_data(self):
        if self.df is None:
            QMessageBox.warning(self, "Error", "Please load a file first.")
            return

        # Ask the user for a filename and storage location
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save File", "", "CSV Files (*.csv);;All Files (*.*)"
        )

        if not file_path:  # User canceled
            return

        # Ensure the file has a .csv extension
        if not file_path.lower().endswith(".csv"):
            file_path += ".csv"

        # Extract the directory and base name from user input
        save_directory = os.path.dirname(file_path)
        custom_base_name = os.path.splitext(os.path.basename(file_path))[0]

        # Prepare file names
        csv_filename = os.path.join(save_directory, f"{custom_base_name}_Analyzed_data.csv")
        plot_filename = os.path.join(save_directory, f"{custom_base_name}_Plot.png")

        # Ensure X and Y axes are selected
        x_col = self.x_combo.currentText()
        y_cols = [item.text() for item in self.y_list.selectedItems()]

        if not x_col or not y_cols:
            QMessageBox.warning(self, "Error", "Please select X and Y axes before exporting.")
            return

        # Filter the data within the specified range
        x_min, x_max = self.x_min.value(), self.x_max.value()
        filtered_df = self.df[(self.df[x_col] >= x_min) & (self.df[x_col] <= x_max)].copy()

        # Create a new DataFrame for exporting
        export_df = filtered_df[[x_col]].copy()

        for y_col in y_cols:
            export_df[y_col] = filtered_df[y_col]

            # Apply smoothing if enabled
            if self.smoothing_combo.currentText() == "Gaussian":
                smoothed_data = self.apply_smoothing(filtered_df[y_col])
                export_df[f"{y_col}_smoothed"] = smoothed_data  # Store smoothed values in a new column

        try:
            # Save the filtered data with proper formatting (semicolon separator for European CSV)
            export_df.to_csv(csv_filename, index=False, sep=";", decimal=",")

            QMessageBox.information(self, "Success", f"Filtered data saved to:\n{csv_filename}")

            # Save the plot as an image
            self.figure.savefig(plot_filename)
            QMessageBox.information(self, "Success", f"Plot saved to:\n{plot_filename}")
        except Exception as e:
            QMessageBox.warning(self, "Error", f"An error occurred while exporting: {e}")

    def update_statistics(self, filtered_df):
        if filtered_df is not None and not filtered_df.empty:
            # Create a QTableWidget for statistics
            self.stats_table.setRowCount(0)  # Clear the table
            self.stats_table.setColumnCount(4)  # Columns: Metric, X-Axis, Y-Axis Name, Value
            self.stats_table.setHorizontalHeaderLabels(["Metric", "X-Axis", "Y-Axis", "Value"])

            x_col = self.x_combo.currentText()
            y_cols = [item.text() for item in self.y_list.selectedItems()]

            # X-axis statistics
            x_stats = {
                "Mean": filtered_df[x_col].mean(),
                "Min": filtered_df[x_col].min(),
                "Max": filtered_df[x_col].max()
            }
            for metric, value in x_stats.items():
                row = self.stats_table.rowCount()
                self.stats_table.insertRow(row)
                self.stats_table.setItem(row, 0, QTableWidgetItem(metric))
                self.stats_table.setItem(row, 1, QTableWidgetItem(x_col))
                self.stats_table.setItem(row, 2, QTableWidgetItem("-"))  # No specific Y-axis
                self.stats_table.setItem(row, 3, QTableWidgetItem(f"{value:.2f}"))

            # Y-axis statistics
            for y_col in y_cols:
                y_stats = {
                    "Mean": filtered_df[y_col].mean(),
                    "Min": filtered_df[y_col].min(),
                    "Max": filtered_df[y_col].max()
                }
                for metric, value in y_stats.items():
                    row = self.stats_table.rowCount()
                    self.stats_table.insertRow(row)
                    self.stats_table.setItem(row, 0, QTableWidgetItem(metric))
                    self.stats_table.setItem(row, 1, QTableWidgetItem("-"))  # No specific X-axis
                    self.stats_table.setItem(row, 2, QTableWidgetItem(y_col))
                    self.stats_table.setItem(row, 3, QTableWidgetItem(f"{value:.2f}"))
        else:
            self.stats_table.setRowCount(0)  # Clear the table
            QMessageBox.warning(self, "No Data", "No data available for statistics.")


    def update_ui(self):
        if self.df is not None:
            self.x_combo.clear()
            self.y_list.clear()
            self.x_combo.addItems(self.df.columns)
            self.y_list.addItems(self.df.columns)

            # Update data preview
            self.data_preview.setText(self.df.head().to_string(index=False))

    def apply_smoothing(self, data):
        filter_type = self.smoothing_combo.currentText()
        if filter_type == "Gaussian":
            sigma = self.sigma_spin.value()
            return gaussian_filter1d(data, sigma)
        return data

    def create_plot(self):
        try:
            if self.df is None:
                QMessageBox.warning(self, "Error", "Please load a file first.")
                return

            x_col = self.x_combo.currentText()
            y_cols = [item.text() for item in self.y_list.selectedItems()]

            if not x_col or not y_cols:
                QMessageBox.warning(self, "Error", "Please select X and Y axes.")
                return

            # Ensure X-axis column is numeric
            self.df[x_col] = pd.to_numeric(self.df[x_col], errors='coerce')

            # Filter data within specified X-axis range
            x_min, x_max = self.x_min.value(), self.x_max.value()
            filtered_df = self.df[(self.df[x_col] >= x_min) & (self.df[x_col] <= x_max)]

            # Update statistics with the filtered data
            self.update_statistics(filtered_df)

            # Clear the axes for the new plot
            self.ax.clear()

            for y_col in y_cols:

                if self.smoothing_combo.currentText() == "None":
                    # Ensure Y-axis column is numeric
                    self.df[y_col] = pd.to_numeric(self.df[y_col], errors='coerce')

                    # Plot original data
                    self.ax.plot(
                        filtered_df[x_col],
                        filtered_df[y_col],
                        label=f"{y_col}",
                        alpha=1
                    )

                # Apply smoothing if smoothing is enabled
                if self.smoothing_combo.currentText() != "None":
                    # Ensure Y-axis column is numeric
                    self.df[y_col] = pd.to_numeric(self.df[y_col], errors='coerce')

                    # Plot original data
                    self.ax.plot(
                        filtered_df[x_col],
                        filtered_df[y_col],
                        label=f"{y_col} (Original)",
                        alpha=0.4
                    )
                    smoothed_data = self.apply_smoothing(filtered_df[y_col])
                    self.ax.plot(
                        filtered_df[x_col],
                        smoothed_data,
                        label=f"{y_col} (Smoothed)",
                        alpha=1.0
                    )

            # Set plot options
            if self.grid_cb.isChecked():
                self.ax.grid(True)

            if self.legend_cb.isChecked():
                self.ax.legend()

            self.ax.set_xlabel(x_col)

            # Set custom plot title if provided
            plot_title = self.title_input.text().strip()
            if plot_title:
                self.ax.set_title(plot_title)

            # Adjust layout and redraw canvas
            self.figure.tight_layout()
            self.canvas.draw()

        except Exception as e:
            QMessageBox.warning(self, "Error", f"An error occurred: {e}")

    def on_click(self, event):
        """Handles mouse click to create a new vertical cursor line at the nearest data point."""
        if event.inaxes is None or self.df is None or event.xdata is None:
            return  # Ignore clicks outside the plot or invalid data

        self.cursor_x = event.xdata  # Store clicked X-coordinate

        # Remove previous cursor line before adding a new one
        for line in self.ax.lines:
            if getattr(line, "_cursor_line", False):  # Check if it is a cursor
                line.remove()

        # Find the closest X-value (npoints)
        x_col = self.x_combo.currentText()
        if x_col not in self.df:
            return

        x_data = self.df[x_col].dropna().values  # Drop NaN values
        if x_data.size == 0:
            return

        # Find the index of the nearest X value
        nearest_index = (np.abs(x_data - self.cursor_x)).argmin()
        nearest_x = x_data[nearest_index]

        # Draw a new vertical cursor line at the nearest X value
        cursor_line = self.ax.axvline(x=nearest_x, color='r', linestyle='--', linewidth=1.5)
        cursor_line._cursor_line = True  # Mark as cursor line

        # Update cursor text with nearest points
        self.update_cursor(nearest_x, nearest_index)

        # Refresh the canvas
        self.canvas.draw()

    def on_mouse_drag(self, event):
        """Handles cursor movement when dragging."""
        if event.inaxes is None or self.df is None or self.cursor_x is None or event.xdata is None:
            return  # Ignore if outside the axes or no cursor initialized

        self.cursor_x = event.xdata  # Update X-position of cursor
        self.update_cursor(self.cursor_x)  # Move cursor dynamically
        self.canvas.draw()  # Ensure updates are drawn

    def update_cursor(self, nearest_x, nearest_index):
        """Updates the vertical cursor text to display values at the nearest X data point."""
        x_col = self.x_combo.currentText()
        y_cols = [item.text() for item in self.y_list.selectedItems()]

        if not x_col or not y_cols:
            return  # No selected axes

        # Get the Y values at the nearest X
        y_values = {y_col: self.df[y_col].iloc[nearest_index] for y_col in y_cols if y_col in self.df}

        # Format text for display
        text_str = f"X: {nearest_x:.2f}\n" + "\n".join([f"{col}: {y_values[col]:.2f}" for col in y_values])

        # Position cursor text at the highest Y value
        y_pos = max(y_values.values()) if y_values else self.ax.get_ylim()[0]

        # Remove previous text before adding a new one
        for txt in self.ax.texts:
            if getattr(txt, "_cursor_text", False):  # Check if it is cursor text
                txt.remove()

        # Create new cursor text
        cursor_text = self.ax.text(nearest_x, y_pos, text_str, fontsize=10, color="red",
                                   bbox=dict(facecolor="white", alpha=0.7))
        cursor_text._cursor_text = True  # Mark as cursor text

        self.canvas.draw()

    def fetch_pump_models(self):
        """Generate pump models dynamically based on the dataset in Tab1."""
        if not hasattr(self, 'df') or self.df is None or self.df.empty:
            self.pump_dropdown.clear()
            self.pump_dropdown.addItem("Error: No data loaded in Tab1")
            return

        try:
            # ✅ Extract relevant column names for context
            columns = ", ".join(self.df.columns[:2]) if len(self.df.columns) > 0 else "sensor data"

            # ✅ Use GPT-2 to generate pump model names
            pipe = pipeline("text-generation", model="gpt2")
            prompt = f"Based on the data: {columns}, Types of anomalies"
            results = pipe(prompt, max_length=20, do_sample=True, temperature=0.7, num_return_sequences=1)

            # ✅ Extract pump names
            pump_models = results[0]['generated_text'].split(", ")
            self.pump_dropdown.clear()
            self.pump_dropdown.addItems(pump_models)

            # ✅ Force UI update
            self.pump_dropdown.repaint()

        except Exception as e:
            self.pump_dropdown.clear()
            self.pump_dropdown.addItem(f"Error fetching models: {e}")

    def analyze_pump_model(self):
        """Compares the selected pump model with the loaded file data (Tab1) to detect anomalies."""
        if not hasattr(self, 'df') or self.df is None or self.df.empty:
            self.anomaly_results.setText("Error: No data file loaded in Tab1.")
            return

        selected_pump = self.pump_dropdown.currentText()
        if selected_pump in ["Waiting for data from Tab1...", "Error: No data loaded in Tab1"]:
            self.anomaly_results.setText("Error: No valid pump model selected.")
            return

        # ✅ Simple anomaly detection based on mean thresholds
        mean_values = self.df.mean()
        anomaly_report = f"Analyzing pump model: {selected_pump}\n\n"

        for col in self.df.columns:
            if self.df[col].dtype in [np.float64, np.int64]:  # Numeric columns only
                mean_val = mean_values[col]
                threshold = mean_val * 1.2  # Example threshold: 20% deviation
                anomalies = self.df[self.df[col] > threshold]

                if not anomalies.empty:
                    anomaly_report += f"⚠️ {col}: {len(anomalies)} anomalies detected (values > {threshold:.2f})\n"

        self.anomaly_results.setText(anomaly_report if anomaly_report.strip() else "No anomalies detected.")

def main():
    app = QApplication(sys.argv)
    window = DataAnalyzer()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
