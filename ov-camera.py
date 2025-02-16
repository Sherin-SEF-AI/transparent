import sys
import random
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                           QHBoxLayout, QGridLayout, QLabel, QPushButton, 
                           QComboBox, QTableWidget, QTableWidgetItem, QTabWidget,
                           QProgressBar, QFrame, QFileDialog, QSpinBox)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal
from PyQt5.QtGui import QPalette, QColor
import pyqtgraph as pg

class SafetyMetric(QFrame):
    def __init__(self, title, parent=None):
        super().__init__(parent)
        self.setFrameStyle(QFrame.Box | QFrame.Raised)
        self.layout = QVBoxLayout(self)
        
        self.title = QLabel(title)
        self.title.setStyleSheet("font-weight: bold; font-size: 14px;")
        
        self.value_label = QLabel("0.0")
        self.value_label.setStyleSheet("font-size: 24px;")
        
        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        
        self.layout.addWidget(self.title)
        self.layout.addWidget(self.value_label)
        self.layout.addWidget(self.progress)
        
    def update_value(self, value):
        self.value_label.setText(f"{value:.1f}")
        self.progress.setValue(int(value))
        
        # Update color based on value
        if value >= 90:
            color = "green"
        elif value >= 80:
            color = "orange"
        else:
            color = "red"
        
        self.progress.setStyleSheet(f"QProgressBar::chunk {{ background-color: {color}; }}")

class HistoricalDataTable(QTableWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setColumnCount(6)
        self.setHorizontalHeaderLabels([
            "Timestamp", "Overall", "Temperature", 
            "Humidity", "Hygiene", "Cross-Contamination"
        ])
        self.verticalHeader().setVisible(False)
        self.setAlternatingRowColors(True)
        
    def add_data_point(self, timestamp, metrics):
        row = self.rowCount()
        self.insertRow(row)
        self.setItem(row, 0, QTableWidgetItem(timestamp))
        for col, value in enumerate(metrics, 1):
            self.setItem(row, col, QTableWidgetItem(f"{value:.1f}"))
        
        # Keep only last 100 rows
        if self.rowCount() > 100:
            self.removeRow(0)

class SafetyDashboard(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Food Safety Monitoring System")
        self.resize(1200, 800)
        
        # Initialize data storage
        self.historical_data = []
        self.alert_thresholds = {
            'temperature': (35, 75),  # Min, Max
            'humidity': (30, 60),
            'hygiene': 85,
            'cross_contamination': 85
        }
        
        self.setup_ui()
        
        # Setup update timer
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_metrics)
        self.timer.start(2000)  # Update every 2 seconds
        
    def setup_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        
        # Create tab widget
        tabs = QTabWidget()
        main_layout.addWidget(tabs)
        
        # Dashboard tab
        dashboard_widget = QWidget()
        dashboard_layout = QVBoxLayout(dashboard_widget)
        
        # Metrics grid
        metrics_widget = QWidget()
        metrics_layout = QGridLayout(metrics_widget)
        
        self.metrics = {
            'overall': SafetyMetric("Overall Safety Index"),
            'temperature': SafetyMetric("Temperature"),
            'humidity': SafetyMetric("Humidity"),
            'hygiene': SafetyMetric("Hygiene"),
            'cross_contamination': SafetyMetric("Cross-Contamination")
        }
        
        # Add metrics to grid
        positions = [(i, j) for i in range(2) for j in range(3)]
        for (metric, widget), (row, col) in zip(self.metrics.items(), positions):
            metrics_layout.addWidget(widget, row, col)
        
        dashboard_layout.addWidget(metrics_widget)
        
        # Graph
        self.graph = pg.PlotWidget()
        self.graph.setBackground('w')
        self.graph.setTitle("Safety Index History")
        self.graph.setLabel('left', 'Safety Index')
        self.graph.setLabel('bottom', 'Time')
        self.graph_line = self.graph.plot(pen=pg.mkPen(color='b', width=2))
        dashboard_layout.addWidget(self.graph)
        
        # Add dashboard tab
        tabs.addTab(dashboard_widget, "Dashboard")
        
        # History tab
        history_widget = QWidget()
        history_layout = QVBoxLayout(history_widget)
        
        # Add export button
        export_btn = QPushButton("Export Data")
        export_btn.clicked.connect(self.export_data)
        history_layout.addWidget(export_btn)
        
        # Add table
        self.history_table = HistoricalDataTable()
        history_layout.addWidget(self.history_table)
        
        tabs.addTab(history_widget, "History")
        
        # Settings tab
        settings_widget = QWidget()
        settings_layout = QVBoxLayout(settings_widget)
        
        # Add threshold settings
        threshold_group = QFrame()
        threshold_layout = QGridLayout(threshold_group)
        threshold_layout.addWidget(QLabel("Alert Thresholds"), 0, 0)
        
        row = 1
        for metric, threshold in self.alert_thresholds.items():
            threshold_layout.addWidget(QLabel(metric.replace('_', ' ').title()), row, 0)
            if isinstance(threshold, tuple):
                min_spin = QSpinBox()
                max_spin = QSpinBox()
                min_spin.setRange(0, 100)
                max_spin.setRange(0, 100)
                min_spin.setValue(threshold[0])
                max_spin.setValue(threshold[1])
                threshold_layout.addWidget(min_spin, row, 1)
                threshold_layout.addWidget(max_spin, row, 2)
            else:
                spin = QSpinBox()
                spin.setRange(0, 100)
                spin.setValue(threshold)
                threshold_layout.addWidget(spin, row, 1)
            row += 1
            
        settings_layout.addWidget(threshold_group)
        settings_layout.addStretch()
        
        tabs.addTab(settings_widget, "Settings")
        
    def update_metrics(self):
        # Simulate real data updates
        metrics = {
            'overall': random.uniform(85, 100),
            'temperature': random.uniform(85, 100),
            'humidity': random.uniform(85, 100),
            'hygiene': random.uniform(85, 100),
            'cross_contamination': random.uniform(85, 100)
        }
        
        # Update metric widgets
        for metric, value in metrics.items():
            self.metrics[metric].update_value(value)
        
        # Update historical data
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.historical_data.append((timestamp, list(metrics.values())))
        
        # Update graph
        if len(self.historical_data) > 50:
            self.historical_data.pop(0)
        
        x = list(range(len(self.historical_data)))
        y = [data[1][0] for data in self.historical_data]  # Overall safety index
        self.graph_line.setData(x, y)
        
        # Update history table
        self.history_table.add_data_point(timestamp, list(metrics.values()))
        
        # Check for alerts
        self.check_alerts(metrics)
    
    def check_alerts(self, metrics):
        alerts = []
        
        temp = metrics['temperature']
        if temp < self.alert_thresholds['temperature'][0]:
            alerts.append(f"Temperature too low: {temp:.1f}")
        elif temp > self.alert_thresholds['temperature'][1]:
            alerts.append(f"Temperature too high: {temp:.1f}")
            
        humidity = metrics['humidity']
        if humidity < self.alert_thresholds['humidity'][0]:
            alerts.append(f"Humidity too low: {humidity:.1f}")
        elif humidity > self.alert_thresholds['humidity'][1]:
            alerts.append(f"Humidity too high: {humidity:.1f}")
            
        if metrics['hygiene'] < self.alert_thresholds['hygiene']:
            alerts.append(f"Hygiene below threshold: {metrics['hygiene']:.1f}")
            
        if metrics['cross_contamination'] < self.alert_thresholds['cross_contamination']:
            alerts.append(f"Cross-contamination risk high: {metrics['cross_contamination']:.1f}")
            
        # TODO: Implement alert display system
    
    def export_data(self):
        filename, _ = QFileDialog.getSaveFileName(
            self, "Export Data", "", "CSV Files (*.csv);;All Files (*)"
        )
        if filename:
            df = pd.DataFrame(
                self.historical_data,
                columns=['Timestamp', 'Overall', 'Temperature', 'Humidity', 
                        'Hygiene', 'Cross-Contamination']
            )
            df.to_csv(filename, index=False)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    
    # Set application style
    app.setStyle('Fusion')
    
    # Create dark palette
    palette = QPalette()
    palette.setColor(QPalette.Window, QColor(53, 53, 53))
    palette.setColor(QPalette.WindowText, Qt.white)
    palette.setColor(QPalette.Base, QColor(25, 25, 25))
    palette.setColor(QPalette.AlternateBase, QColor(53, 53, 53))
    palette.setColor(QPalette.ToolTipBase, Qt.white)
    palette.setColor(QPalette.ToolTipText, Qt.white)
    palette.setColor(QPalette.Text, Qt.white)
    palette.setColor(QPalette.Button, QColor(53, 53, 53))
    palette.setColor(QPalette.ButtonText, Qt.white)
    palette.setColor(QPalette.BrightText, Qt.red)
    palette.setColor(QPalette.Link, QColor(42, 130, 218))
    palette.setColor(QPalette.Highlight, QColor(42, 130, 218))
    palette.setColor(QPalette.HighlightedText, Qt.black)
    app.setPalette(palette)
    
    window = SafetyDashboard()
    window.show()
    
    sys.exit(app.exec_())