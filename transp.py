import sys
import random
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                           QHBoxLayout, QGridLayout, QLabel, QPushButton, 
                           QComboBox, QTableWidget, QTableWidgetItem, QTabWidget,
                           QProgressBar, QFrame, QFileDialog, QSpinBox, QGroupBox,
                           QCheckBox, QSystemTrayIcon, QMenu, QCalendarWidget,
                           QDateTimeEdit, QScrollArea, QTextEdit)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QDateTime
from PyQt5.QtGui import QPalette, QColor, QIcon
import pyqtgraph as pg
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

class SafetyMetric(QFrame):
    def __init__(self, title, parent=None):
        super().__init__(parent)
        self.setFrameStyle(QFrame.Box | QFrame.Raised)
        layout = QVBoxLayout(self)
        
        self.title = QLabel(title)
        self.title.setStyleSheet("font-weight: bold; font-size: 14px;")
        
        self.value_label = QLabel("0.0")
        self.value_label.setStyleSheet("font-size: 24px;")
        
        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        
        layout.addWidget(self.title)
        layout.addWidget(self.value_label)
        layout.addWidget(self.progress)
        
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

import sys
import random
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                           QHBoxLayout, QGridLayout, QLabel, QPushButton, 
                           QComboBox, QTableWidget, QTableWidgetItem, QTabWidget,
                           QProgressBar, QFrame, QFileDialog, QSpinBox, QGroupBox,
                           QCheckBox, QSystemTrayIcon, QMenu, QCalendarWidget,
                           QDateTimeEdit, QScrollArea)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QDateTime
from PyQt5.QtGui import QPalette, QColor, QIcon
import pyqtgraph as pg
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure



class PredictionWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        
        # Prediction settings
        settings_group = QGroupBox("Prediction Settings")
        settings_layout = QHBoxLayout(settings_group)
        
        self.metric_selector = QComboBox()
        self.metric_selector.addItems(['Overall', 'Temperature', 'Humidity', 'Hygiene', 'Cross-Contamination'])
        
        self.horizon_spinner = QSpinBox()
        self.horizon_spinner.setRange(1, 24)
        self.horizon_spinner.setValue(6)
        
        settings_layout.addWidget(QLabel("Metric:"))
        settings_layout.addWidget(self.metric_selector)
        settings_layout.addWidget(QLabel("Prediction Horizon (hours):"))
        settings_layout.addWidget(self.horizon_spinner)
        
        layout.addWidget(settings_group)
        
        # Prediction plot
        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setBackground('w')
        self.plot_widget.setTitle("Prediction Analysis")
        self.plot_widget.setLabel('left', 'Value')
        self.plot_widget.setLabel('bottom', 'Time')
        
        self.historical_line = self.plot_widget.plot(
            pen=pg.mkPen(color='b', width=2),
            name='Historical'
        )
        self.prediction_line = self.plot_widget.plot(
            pen=pg.mkPen(color='r', width=2, style=Qt.DashLine),
            name='Prediction'
        )
        
        layout.addWidget(self.plot_widget)
        
        # Prediction metrics
        metrics_group = QGroupBox("Prediction Metrics")
        metrics_layout = QGridLayout(metrics_group)
        
        self.mse_label = QLabel("MSE: N/A")
        self.r2_label = QLabel("R²: N/A")
        self.confidence_label = QLabel("Confidence: N/A")
        
        metrics_layout.addWidget(self.mse_label, 0, 0)
        metrics_layout.addWidget(self.r2_label, 0, 1)
        metrics_layout.addWidget(self.confidence_label, 0, 2)
        
        layout.addWidget(metrics_group)
        
        self.model = RandomForestRegressor(n_estimators=100)
        self.scaler = StandardScaler()
        
    def update_prediction(self, historical_data):
        if len(historical_data) < 10:  # Need minimum data points
            return
            
        # Prepare data
        metric_idx = self.metric_selector.currentIndex() + 1
        values = np.array([data[1][metric_idx] for data in historical_data])
        
        # Create features (time series lags)
        X = np.array([values[i:i+5] for i in range(len(values)-5)])
        y = values[5:]
        
        if len(X) < 2:
            return
            
        # Train model
        self.model.fit(X, y)
        
        # Make prediction
        last_window = values[-5:].reshape(1, -1)
        predictions = []
        for _ in range(self.horizon_spinner.value()):
            pred = self.model.predict(last_window)
            predictions.append(pred[0])
            last_window = np.roll(last_window, -1)
            last_window[0, -1] = pred[0]
        
        # Update plot
        x_historical = list(range(len(values)))
        x_future = list(range(len(values)-1, len(values) + len(predictions)-1))
        
        self.historical_line.setData(x_historical, values)
        self.prediction_line.setData(x_future, [values[-1]] + predictions)
        
        # Update metrics
        y_pred = self.model.predict(X)
        mse = mean_squared_error(y, y_pred)
        r2 = r2_score(y, y_pred)
        
        self.mse_label.setText(f"MSE: {mse:.2f}")
        self.r2_label.setText(f"R²: {r2:.2f}")
        self.confidence_label.setText(f"Confidence: {r2 * 100:.1f}%")

class AnomalyDetectionWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        
        # Settings
        settings_group = QGroupBox("Anomaly Detection Settings")
        settings_layout = QHBoxLayout(settings_group)
        
        self.contamination = QSpinBox()
        self.contamination.setRange(1, 20)
        self.contamination.setValue(10)
        
        settings_layout.addWidget(QLabel("Contamination (%):"))
        settings_layout.addWidget(self.contamination)
        
        layout.addWidget(settings_group)
        
        # Anomaly plot
        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setBackground('w')
        self.plot_widget.setTitle("Anomaly Detection")
        self.plot_widget.setLabel('left', 'Value')
        self.plot_widget.setLabel('bottom', 'Time')
        
        self.normal_scatter = self.plot_widget.plot(
            symbolBrush=(0,0,255), symbolSize=10,
            symbol='o', name='Normal'
        )
        self.anomaly_scatter = self.plot_widget.plot(
            symbolBrush=(255,0,0), symbolSize=10,
            symbol='x', name='Anomaly'
        )
        
        layout.addWidget(self.plot_widget)
        
        # Anomaly list
        self.anomaly_list = QTableWidget()
        self.anomaly_list.setColumnCount(3)
        self.anomaly_list.setHorizontalHeaderLabels(['Timestamp', 'Metric', 'Value'])
        
        layout.addWidget(self.anomaly_list)
        
        self.model = IsolationForest(contamination=0.1)
        
    def update_anomalies(self, historical_data):
        if len(historical_data) < 10:
            return
            
        # Prepare data
        timestamps = [data[0] for data in historical_data]
        values = np.array([data[1] for data in historical_data])
        
        # Detect anomalies
        self.model.fit(values)
        predictions = self.model.predict(values)
        
        # Split normal and anomaly points
        normal_idx = predictions == 1
        anomaly_idx = predictions == -1
        
        x = list(range(len(values)))
        
        # Update plots
        normal_x = np.array(x)[normal_idx]
        normal_y = values[normal_idx, 0]  # Overall safety index
        
        anomaly_x = np.array(x)[anomaly_idx]
        anomaly_y = values[anomaly_idx, 0]
        
        self.normal_scatter.setData(normal_x, normal_y)
        self.anomaly_scatter.setData(anomaly_x, anomaly_y)
        
        # Update anomaly list
        self.anomaly_list.setRowCount(0)
        for i, is_anomaly in enumerate(anomaly_idx):
            if is_anomaly:
                row = self.anomaly_list.rowCount()
                self.anomaly_list.insertRow(row)
                self.anomaly_list.setItem(row, 0, QTableWidgetItem(timestamps[i]))
                self.anomaly_list.setItem(row, 1, QTableWidgetItem("Overall"))
                self.anomaly_list.setItem(row, 2, QTableWidgetItem(f"{values[i, 0]:.1f}"))

class CorrelationWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        
        self.figure = Figure(figsize=(8, 6))
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas)
        
        self.ax = self.figure.add_subplot(111)
        
    def update_correlation(self, historical_data):
        if len(historical_data) < 2:
            return
            
        # Prepare data
        values = np.array([data[1] for data in historical_data])
        metrics = ['Overall', 'Temperature', 'Humidity', 'Hygiene', 'Cross-Contamination']
        
        # Calculate correlation matrix
        corr_matrix = np.corrcoef(values.T)
        
        self.ax.clear()
        im = self.ax.imshow(corr_matrix, cmap='coolwarm', aspect='auto')
        
        # Add labels
        self.ax.set_xticks(range(len(metrics)))
        self.ax.set_yticks(range(len(metrics)))
        self.ax.set_xticklabels(metrics, rotation=45)
        self.ax.set_yticklabels(metrics)
        
        # Add colorbar
        self.figure.colorbar(im)
        
        self.ax.set_title('Metric Correlations')
        
        self.figure.tight_layout()
        self.canvas.draw()

class RadarChart(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        
        self.figure = Figure(figsize=(8, 6))
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas)
        
        self.ax = self.figure.add_subplot(111, projection='polar')
        
    def update_radar(self, metrics):
        self.ax.clear()
        
        # Prepare data for radar chart
        categories = list(metrics.keys())
        values = list(metrics.values())
        
        # Close the polygon by appending the first value
        values += values[:1]
        angles = np.linspace(0, 2*np.pi, len(categories), endpoint=False)
        angles = np.concatenate((angles, [angles[0]]))  # close the polygon
        
        self.ax.plot(angles, values)
        self.ax.fill(angles, values, alpha=0.25)
        self.ax.set_xticks(angles[:-1])
        self.ax.set_xticklabels(categories)
        self.ax.set_ylim(0, 100)
        
        self.canvas.draw()

class AlertWidget(QFrame):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFrameStyle(QFrame.Box | QFrame.Raised)
        layout = QVBoxLayout(self)
        
        self.alert_list = QTableWidget()
        self.alert_list.setColumnCount(3)
        self.alert_list.setHorizontalHeaderLabels(['Timestamp', 'Metric', 'Alert'])
        self.alert_list.verticalHeader().setVisible(False)
        
        layout.addWidget(self.alert_list)
        
    def add_alert(self, timestamp, metric, message):
        row = self.alert_list.rowCount()
        self.alert_list.insertRow(row)
        self.alert_list.setItem(row, 0, QTableWidgetItem(timestamp))
        self.alert_list.setItem(row, 1, QTableWidgetItem(metric))
        self.alert_list.setItem(row, 2, QTableWidgetItem(message))
        
        # Keep only last 100 alerts
        if self.alert_list.rowCount() > 100:
            self.alert_list.removeRow(0)

class StatisticsWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        
        self.stats_table = QTableWidget()
        self.stats_table.setColumnCount(4)
        self.stats_table.setHorizontalHeaderLabels(['Metric', 'Average', 'Min', 'Max'])
        self.stats_table.verticalHeader().setVisible(False)
        
        layout.addWidget(self.stats_table)
        
    def update_statistics(self, historical_data):
        metrics = ['Overall', 'Temperature', 'Humidity', 'Hygiene', 'Cross-Contamination']
        self.stats_table.setRowCount(len(metrics))
        
        for i, metric in enumerate(metrics):
            values = [data[1][i] for data in historical_data]
            
            self.stats_table.setItem(i, 0, QTableWidgetItem(metric))
            self.stats_table.setItem(i, 1, QTableWidgetItem(f"{np.mean(values):.1f}"))
            self.stats_table.setItem(i, 2, QTableWidgetItem(f"{np.min(values):.1f}"))
            self.stats_table.setItem(i, 3, QTableWidgetItem(f"{np.max(values):.1f}"))

class TrendAnalysisWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        
        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setBackground('w')
        self.plot_widget.setTitle("Trend Analysis")
        self.plot_widget.setLabel('left', 'Safety Index')
        self.plot_widget.setLabel('bottom', 'Time')
        
        self.plot_lines = {}
        colors = ['b', 'r', 'g', 'y', 'm']
        metrics = ['Overall', 'Temperature', 'Humidity', 'Hygiene', 'Cross-Contamination']
        
        for metric, color in zip(metrics, colors):
            self.plot_lines[metric] = self.plot_widget.plot(
                pen=pg.mkPen(color=color, width=2), name=metric
            )
        
        layout.addWidget(self.plot_widget)
        
        # Add legend
        self.plot_widget.addLegend()
        
    def update_trends(self, historical_data):
        x = list(range(len(historical_data)))
        
        for i, (metric, line) in enumerate(self.plot_lines.items()):
            y = [data[1][i] for data in historical_data]
            line.setData(x, y)

class SafetyDashboard(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Advanced Food Safety Monitoring System")
        self.resize(1400, 900)
        
        # Initialize data storage
        self.historical_data = []
        self.alert_thresholds = {
            'temperature': (35, 75),
            'humidity': (30, 60),
            'hygiene': 85,
            'cross_contamination': 85
        }
        
        # Setup system tray
        self.setup_system_tray()
        
        self.setup_ui()
        
        # Setup update timer
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_metrics)
        self.timer.start(2000)  # Update every 2 seconds
        
    def setup_system_tray(self):
        self.tray_icon = QSystemTrayIcon(self)
        self.tray_icon.setIcon(QIcon.fromTheme("dialog-information"))
        
        # Create tray menu
        tray_menu = QMenu()
        show_action = tray_menu.addAction("Show Dashboard")
        show_action.triggered.connect(self.show)
        quit_action = tray_menu.addAction("Quit")
        quit_action.triggered.connect(QApplication.quit)
        
        self.tray_icon.setContextMenu(tray_menu)
        self.tray_icon.show()
        
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
        
        # Add metrics grid
        metrics_widget = QWidget()
        metrics_layout = QGridLayout(metrics_widget)
        
        self.metrics = {
            'overall': SafetyMetric("Overall Safety Index"),
            'temperature': SafetyMetric("Temperature"),
            'humidity': SafetyMetric("Humidity"),
            'hygiene': SafetyMetric("Hygiene"),
            'cross_contamination': SafetyMetric("Cross-Contamination")
        }
        
        positions = [(i, j) for i in range(2) for j in range(3)]
        for (metric, widget), (row, col) in zip(self.metrics.items(), positions):
            metrics_layout.addWidget(widget, row, col)
        
        dashboard_layout.addWidget(metrics_widget)
        
        # Add visualizations
        viz_widget = QWidget()
        viz_layout = QHBoxLayout(viz_widget)
        
        # Add radar chart for current status
        self.radar_chart = RadarChart()
        viz_layout.addWidget(self.radar_chart)
        
        # Add correlation analysis
        self.correlation = CorrelationWidget()
        viz_layout.addWidget(self.correlation)
        
        dashboard_layout.addWidget(viz_widget)
        
        # Add ML analysis section
        ml_widget = QWidget()
        ml_layout = QHBoxLayout(ml_widget)
        
        # Add prediction widget
        self.prediction = PredictionWidget()
        ml_layout.addWidget(self.prediction)
        
        # Add anomaly detection
        self.anomaly = AnomalyDetectionWidget()
        ml_layout.addWidget(self.anomaly)
        
        dashboard_layout.addWidget(ml_widget)
        
        tabs.addTab(dashboard_widget, "Dashboard")
        
        # Analysis tab
        analysis_widget = QWidget()
        analysis_layout = QVBoxLayout(analysis_widget)
        
        # Add trend analysis
        self.trend_analysis = TrendAnalysisWidget()
        analysis_layout.addWidget(self.trend_analysis)
        
        # Add statistics
        self.statistics = StatisticsWidget()
        analysis_layout.addWidget(self.statistics)
        
        tabs.addTab(analysis_widget, "Analysis")
        
        # Alerts tab
        alerts_widget = QWidget()
        alerts_layout = QVBoxLayout(alerts_widget)
        
        self.alert_widget = AlertWidget()
        alerts_layout.addWidget(self.alert_widget)
        
        tabs.addTab(alerts_widget, "Alerts")
        
        # Reports tab
        reports_widget = QWidget()
        reports_layout = QVBoxLayout(reports_widget)
        
        # Add date range selection
        date_range = QGroupBox("Date Range")
        date_range_layout = QHBoxLayout(date_range)
        
        self.start_date = QDateTimeEdit()
        self.start_date.setDateTime(QDateTime.currentDateTime().addDays(-7))
        self.end_date = QDateTimeEdit()
        self.end_date.setDateTime(QDateTime.currentDateTime())
        
        date_range_layout.addWidget(QLabel("From:"))
        date_range_layout.addWidget(self.start_date)
        date_range_layout.addWidget(QLabel("To:"))
        date_range_layout.addWidget(self.end_date)
        
        reports_layout.addWidget(date_range)
        
        # Add export options
        export_group = QGroupBox("Export Options")
        export_layout = QVBoxLayout(export_group)
        
        self.export_metrics = QCheckBox("Include Metrics")
        self.export_alerts = QCheckBox("Include Alerts")
        self.export_stats = QCheckBox("Include Statistics")
        
        export_layout.addWidget(self.export_metrics)
        export_layout.addWidget(self.export_alerts)
        export_layout.addWidget(self.export_stats)
        
        export_btn = QPushButton("Generate Report")
        export_btn.clicked.connect(self.generate_report)
        
        export_layout.addWidget(export_btn)
        reports_layout.addWidget(export_group)
        
        tabs.addTab(reports_widget, "Reports")
        
        # Settings tab
        settings_widget = QScrollArea()
        settings_content = QWidget()
        settings_layout = QVBoxLayout(settings_content)
        
        # Alert settings
        alert_settings = QGroupBox("Alert Settings")
        alert_layout = QGridLayout(alert_settings)
        
        row = 0
        for metric, threshold in self.alert_thresholds.items():
            alert_layout.addWidget(QLabel(metric.replace('_', ' ').title()), row, 0)
            if isinstance(threshold, tuple):
                min_spin = QSpinBox()
                max_spin = QSpinBox()
                min_spin.setRange(0, 100)
                max_spin.setRange(0, 100)
                min_spin.setValue(threshold[0])
                max_spin.setValue(threshold[1])
                alert_layout.addWidget(min_spin, row, 1)
                alert_layout.addWidget(max_spin, row, 2)
            else:
                spin = QSpinBox()
                spin.setRange(0, 100)
                spin.setValue(threshold)
                alert_layout.addWidget(spin, row, 1)
            row += 1
            
        settings_layout.addWidget(alert_settings)
        
        # Display settings
        display_settings = QGroupBox("Display Settings")
        display_layout = QVBoxLayout(display_settings)
        
        self.dark_mode = QCheckBox("Dark Mode")
        self.dark_mode.setChecked(True)
        self.dark_mode.stateChanged.connect(self.toggle_theme)
        
        self.show_tooltips = QCheckBox("Show Tooltips")
        self.show_tooltips.setChecked(True)
        
        display_layout.addWidget(self.dark_mode)
        display_layout.addWidget(self.show_tooltips)
        
        settings_layout.addWidget(display_settings)
        
        # Update frequency
        update_settings = QGroupBox("Update Settings")
        update_layout = QHBoxLayout(update_settings)
        
        update_layout.addWidget(QLabel("Update Frequency (ms):"))
        self.update_frequency = QSpinBox()
        self.update_frequency.setRange(500, 10000)
        self.update_frequency.setValue(2000)
        self.update_frequency.valueChanged.connect(self.update_timer_frequency)
        update_layout.addWidget(self.update_frequency)
        
        settings_layout.addWidget(update_settings)
        
        # Notification settings
        notification_settings = QGroupBox("Notification Settings")
        notification_layout = QVBoxLayout(notification_settings)
        
        self.enable_notifications = QCheckBox("Enable System Notifications")
        self.enable_notifications.setChecked(True)
        
        self.notification_threshold = QSpinBox()
        self.notification_threshold.setRange(0, 100)
        self.notification_threshold.setValue(80)
        
        notification_layout.addWidget(self.enable_notifications)
        notification_layout.addWidget(QLabel("Critical Threshold:"))
        notification_layout.addWidget(self.notification_threshold)
        
        settings_layout.addWidget(notification_settings)
        
        settings_layout.addStretch()
        settings_widget.setWidget(settings_content)
        settings_widget.setWidgetResizable(True)
        
        tabs.addTab(settings_widget, "Settings")
        
    def update_timer_frequency(self, value):
        self.timer.setInterval(value)
        
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
        
        # Keep last 100 data points
        if len(self.historical_data) > 100:
            self.historical_data.pop(0)
        
        # Update basic visualizations
        self.radar_chart.update_radar(metrics)
        self.trend_analysis.update_trends(self.historical_data)
        self.statistics.update_statistics(self.historical_data)
        
        # Update ML components
        self.prediction.update_prediction(self.historical_data)
        self.anomaly.update_anomalies(self.historical_data)
        self.correlation.update_correlation(self.historical_data)
        
        # Check for alerts
        self.check_alerts(metrics)
        
        # Generate insights
        self.generate_insights(metrics)
        
    def check_alerts(self, metrics):
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        temp = metrics['temperature']
        if temp < self.alert_thresholds['temperature'][0]:
            self.add_alert(current_time, "Temperature", f"Temperature too low: {temp:.1f}")
        elif temp > self.alert_thresholds['temperature'][1]:
            self.add_alert(current_time, "Temperature", f"Temperature too high: {temp:.1f}")
            
        humidity = metrics['humidity']
        if humidity < self.alert_thresholds['humidity'][0]:
            self.add_alert(current_time, "Humidity", f"Humidity too low: {humidity:.1f}")
        elif humidity > self.alert_thresholds['humidity'][1]:
            self.add_alert(current_time, "Humidity", f"Humidity too high: {humidity:.1f}")
            
        if metrics['hygiene'] < self.alert_thresholds['hygiene']:
            self.add_alert(current_time, "Hygiene", f"Hygiene below threshold: {metrics['hygiene']:.1f}")
            
        if metrics['cross_contamination'] < self.alert_thresholds['cross_contamination']:
            self.add_alert(current_time, "Cross-contamination", 
                         f"Cross-contamination risk high: {metrics['cross_contamination']:.1f}")
    
    def add_alert(self, timestamp, metric, message):
        self.alert_widget.add_alert(timestamp, metric, message)
        
        if self.enable_notifications.isChecked():
            self.tray_icon.showMessage(
                f"Food Safety Alert - {metric}",
                message,
                QSystemTrayIcon.Warning,
                3000
            )
    
    def generate_insights(self, metrics):
        if len(self.historical_data) < 10:
            return
            
        # Create insights tab if it doesn't exist
        if not hasattr(self, 'insights_text'):
            insights_tab = QWidget()
            insights_layout = QVBoxLayout(insights_tab)
            
            self.insights_text = QTextEdit()
            self.insights_text.setReadOnly(True)
            insights_layout.addWidget(self.insights_text)
            
            # Find the tab widget
            for child in self.findChildren(QTabWidget):
                child.addTab(insights_tab, "AI Insights")
                break
        
        # Generate insights
        insights = []
        
        # Trend analysis
        for metric, value in metrics.items():
            if len(self.historical_data) >= 2:
                prev_value = self.historical_data[-2][1][list(metrics.keys()).index(metric)]
                change = value - prev_value
                if abs(change) > 2:
                    direction = "increased" if change > 0 else "decreased"
                    insights.append(f"⚠️ {metric.replace('_', ' ').title()} has {direction} by {abs(change):.1f} points")
        
        # Correlation insights
        if len(self.historical_data) >= 10:
            values = np.array([data[1] for data in self.historical_data])
            corr_matrix = np.corrcoef(values.T)
            metric_names = list(metrics.keys())
            
            for i in range(len(metric_names)):
                for j in range(i+1, len(metric_names)):
                    if abs(corr_matrix[i,j]) > 0.8:
                        insights.append(f"📊 Strong correlation detected between {metric_names[i]} and {metric_names[j]}")
        
        # Anomaly insights
        recent_anomalies = []
        for i in range(max(0, len(self.historical_data)-5), len(self.historical_data)):
            if i >= 0:
                for metric, value in zip(metrics.keys(), self.historical_data[i][1]):
                    if value < 70:
                        recent_anomalies.append(f"🚨 Critical low value detected for {metric}: {value:.1f}")
        
        # Prediction insights
        if hasattr(self, 'prediction') and hasattr(self.prediction, 'model'):
            try:
                confidence = self.prediction.model.score(
                    np.array([data[1] for data in self.historical_data[:-1]]),
                    np.array([data[1] for data in self.historical_data[1:]])
                )
                if confidence > 0.8:
                    insights.append(f"🎯 High prediction confidence: {confidence:.1%}")
            except:
                pass
        
        # Update insights text
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        insights_text = f"AI Insights (Generated at {current_time})\n\n"
        insights_text += "\n".join(insights) if insights else "No significant insights at this time."
        
        self.insights_text.setText(insights_text)
    
    def generate_report(self):
        filename, _ = QFileDialog.getSaveFileName(
            self, "Save Report", "", "Excel Files (*.xlsx);;CSV Files (*.csv);;All Files (*)"
        )
        
        if not filename:
            return
            
        report_data = {
            'Timestamp': [],
            'Overall': [],
            'Temperature': [],
            'Humidity': [],
            'Hygiene': [],
            'Cross-Contamination': []
        }
        
        for timestamp, values in self.historical_data:
            report_data['Timestamp'].append(timestamp)
            report_data['Overall'].append(values[0])
            report_data['Temperature'].append(values[1])
            report_data['Humidity'].append(values[2])
            report_data['Hygiene'].append(values[3])
            report_data['Cross-Contamination'].append(values[4])
            
        df = pd.DataFrame(report_data)
        
        if filename.endswith('.xlsx'):
            with pd.ExcelWriter(filename) as writer:
                df.to_excel(writer, sheet_name='Metrics', index=False)
                
                if self.export_stats.isChecked():
                    stats = df.describe()
                    stats.to_excel(writer, sheet_name='Statistics')
        else:
            df.to_csv(filename, index=False)
            
    def toggle_theme(self, checked):
        if checked:
            # Dark theme
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
        else:
            # Light theme
            palette = QPalette()
            palette.setColor(QPalette.Window, Qt.white)
            palette.setColor(QPalette.WindowText, Qt.black)
            palette.setColor(QPalette.Base, Qt.white)
            palette.setColor(QPalette.AlternateBase, QColor(245, 245, 245))
            palette.setColor(QPalette.ToolTipBase, Qt.white)
            palette.setColor(QPalette.ToolTipText, Qt.black)
            palette.setColor(QPalette.Text, Qt.black)
            palette.setColor(QPalette.Button, Qt.white)
            palette.setColor(QPalette.ButtonText, Qt.black)
            palette.setColor(QPalette.BrightText, Qt.red)
            palette.setColor(QPalette.Link, QColor(0, 0, 255))
            palette.setColor(QPalette.Highlight, QColor(0, 120, 215))
            palette.setColor(QPalette.HighlightedText, Qt.white)
            
        QApplication.setPalette(palette)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    
    # Set application style
    app.setStyle('Fusion')
    
    # Set dark theme by default
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