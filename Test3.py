import sys
import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QPushButton, QComboBox, QFileDialog, QLabel,
                             QGroupBox, QGridLayout, QCheckBox, QMessageBox, QTextEdit,
                             QListWidget, QAbstractItemView, QSpinBox, QDoubleSpinBox, QTableWidgetItem, QTableWidget,
                             QLineEdit)
from PyQt5.QtCore import Qt
from scipy.ndimage import gaussian_filter1d


class DataAnalyzer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Data Analyzer")
        self.setGeometry(100, 100, 1200, 800)

        # Initialize variables
        self.df = None
        self.filename = None
        self.last_modified_time = None
        self.colors = ['b', 'r', 'g', 'c', 'm', 'y', 'k']
        self.setup_ui()

    def setup_ui(self):
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QVBoxLayout(main_widget)

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

        # Range filters
        plot_layout.addWidget(QLabel("X-Range:"), 2, 0)
        x_range_layout = QHBoxLayout()
        self.x_min = QDoubleSpinBox()
        self.x_min.setDecimals(2)
        self.x_min.setRange(-1e6, 1e6)
        self.x_min.setValue(-1e6)
        self.x_min.setPrefix("Min: ")
        x_range_layout.addWidget(self.x_min)
        self.x_max = QDoubleSpinBox()
        self.x_max.setDecimals(2)
        self.x_max.setRange(-1e6, 1e6)
        self.x_max.setValue(1e6)
        self.x_max.setPrefix("Max: ")
        x_range_layout.addWidget(self.x_max)
        plot_layout.addLayout(x_range_layout, 2, 1)

        plot_layout.addWidget(QLabel("Y-Range:"), 3, 0)
        y_range_layout = QHBoxLayout()
        self.y_min = QDoubleSpinBox()
        self.y_min.setDecimals(2)
        self.y_min.setRange(-1e6, 1e6)
        self.y_min.setValue(-1e6)
        self.y_min.setPrefix("Min: ")
        y_range_layout.addWidget(self.y_min)
        self.y_max = QDoubleSpinBox()
        self.y_max.setDecimals(2)
        self.y_max.setRange(-1e6, 1e6)
        self.y_max.setValue(1e6)
        self.y_max.setPrefix("Max: ")
        y_range_layout.addWidget(self.y_max)
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


def main():
    app = QApplication(sys.argv)
    window = DataAnalyzer()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
