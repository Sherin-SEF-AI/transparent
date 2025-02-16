import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from dataclasses import dataclass
import random
import json
import sqlite3
from typing import Dict, List
import csv
from PIL import Image, ImageTk
import io
import threading
import queue
from scipy import stats
import plotly.graph_objects as go
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph
from reportlab.lib.styles import getSampleStyleSheet
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# Advanced Data Structures
@dataclass
class SensorReading:
    timestamp: datetime
    temperature: float
    humidity: float
    pressure: float
    co2_level: float
    light_level: float
    quality_score: float
    location: str
    batch_id: str
    equipment_id: str
    operator_id: str
    alert_status: str = "normal"

class AnalyticsEngine:
    def __init__(self):
        self.models = {}
        self.thresholds = self.load_thresholds()

    def load_thresholds(self):
        # Load from configuration file or use defaults
        return {
            'temperature': {'min': 2, 'max': 25, 'critical_min': 0, 'critical_max': 30},
            'humidity': {'min': 30, 'max': 70, 'critical_min': 20, 'critical_max': 80},
            'co2_level': {'min': 300, 'max': 500, 'critical_min': 200, 'critical_max': 600}
        }

    def analyze_trend(self, data: pd.DataFrame, metric: str) -> dict:
        if len(data) < 2:
            return {'trend': 'insufficient_data'}

        # Perform trend analysis
        slope, intercept, r_value, p_value, std_err = stats.linregress(
            range(len(data)), data[metric])

        return {
            'trend': 'increasing' if slope > 0 else 'decreasing',
            'slope': slope,
            'r_squared': r_value ** 2,
            'significance': p_value < 0.05
        }

    def detect_anomalies(self, data: pd.DataFrame, metric: str) -> pd.DataFrame:
        if len(data) < 3:
            return pd.DataFrame()

        # Calculate Z-scores
        z_scores = np.abs(stats.zscore(data[metric]))
        anomalies = data[z_scores > 3].copy()
        
        # Add anomaly classification
        thresholds = self.thresholds.get(metric, {})
        if thresholds:
            anomalies['severity'] = anomalies[metric].apply(
                lambda x: 'critical' if (x < thresholds['critical_min'] or 
                                       x > thresholds['critical_max'])
                else 'warning'
            )

        return anomalies

    def predict_next_value(self, data: pd.DataFrame, metric: str) -> float:
        if len(data) < 5:
            return None

        # Simple exponential smoothing
        alpha = 0.3
        last_smoothed = data[metric].ewm(alpha=alpha).mean().iloc[-1]
        return last_smoothed

    def generate_health_score(self, reading: SensorReading) -> float:
        scores = []
        
        # Temperature score
        temp_score = self.calculate_metric_score(
            reading.temperature,
            self.thresholds['temperature']['min'],
            self.thresholds['temperature']['max'],
            self.thresholds['temperature']['critical_min'],
            self.thresholds['temperature']['critical_max']
        )
        scores.append(temp_score)

        # Humidity score
        humidity_score = self.calculate_metric_score(
            reading.humidity,
            self.thresholds['humidity']['min'],
            self.thresholds['humidity']['max'],
            self.thresholds['humidity']['critical_min'],
            self.thresholds['humidity']['critical_max']
        )
        scores.append(humidity_score)

        # CO2 score
        co2_score = self.calculate_metric_score(
            reading.co2_level,
            self.thresholds['co2_level']['min'],
            self.thresholds['co2_level']['max'],
            self.thresholds['co2_level']['critical_min'],
            self.thresholds['co2_level']['critical_max']
        )
        scores.append(co2_score)

        return np.mean(scores)

    def calculate_metric_score(self, value, min_val, max_val, 
                             critical_min, critical_max) -> float:
        if value < critical_min or value > critical_max:
            return 0
        elif value < min_val:
            return 50 * (value - critical_min) / (min_val - critical_min)
        elif value > max_val:
            return 50 * (critical_max - value) / (critical_max - max_val)
        else:
            return 100

class DatabaseManager:
    def __init__(self):
        self.conn = sqlite3.connect('traceability.db')
        self.setup_database()

    def setup_database(self):
        cursor = self.conn.cursor()
        
        # Sensor readings table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS sensor_readings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                temperature REAL,
                humidity REAL,
                pressure REAL,
                co2_level REAL,
                light_level REAL,
                quality_score REAL,
                location TEXT,
                batch_id TEXT,
                equipment_id TEXT,
                operator_id TEXT,
                alert_status TEXT
            )
        ''')

        # Equipment table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS equipment (
                id TEXT PRIMARY KEY,
                name TEXT,
                location TEXT,
                last_maintenance TEXT,
                status TEXT
            )
        ''')

        # Operators table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS operators (
                id TEXT PRIMARY KEY,
                name TEXT,
                role TEXT,
                status TEXT
            )
        ''')

        # Batches table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS batches (
                id TEXT PRIMARY KEY,
                start_time TEXT,
                end_time TEXT,
                product_type TEXT,
                status TEXT
            )
        ''')

        self.conn.commit()

    def save_reading(self, reading: SensorReading):
        cursor = self.conn.cursor()
        cursor.execute('''
            INSERT INTO sensor_readings 
            (timestamp, temperature, humidity, pressure, co2_level, 
             light_level, quality_score, location, batch_id, 
             equipment_id, operator_id, alert_status)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            reading.timestamp.isoformat(),
            reading.temperature,
            reading.humidity,
            reading.pressure,
            reading.co2_level,
            reading.light_level,
            reading.quality_score,
            reading.location,
            reading.batch_id,
            reading.equipment_id,
            reading.operator_id,
            reading.alert_status
        ))
        self.conn.commit()

    def get_readings(self, location: str, start_time: datetime, 
                    end_time: datetime) -> pd.DataFrame:
        query = '''
            SELECT * FROM sensor_readings 
            WHERE location = ? 
            AND timestamp BETWEEN ? AND ?
        '''
        return pd.read_sql_query(
            query, 
            self.conn, 
            params=[location, start_time.isoformat(), end_time.isoformat()]
        )

    def export_data(self, format='csv', start_time=None, end_time=None):
        query = 'SELECT * FROM sensor_readings'
        if start_time and end_time:
            query += ' WHERE timestamp BETWEEN ? AND ?'
            params = [start_time.isoformat(), end_time.isoformat()]
        else:
            params = []

        df = pd.read_sql_query(query, self.conn, params=params)
        
        if format == 'csv':
            return df.to_csv(index=False)
        elif format == 'json':
            return df.to_json(orient='records')
        elif format == 'excel':
            buffer = io.BytesIO()
            df.to_excel(buffer, index=False)
            return buffer.getvalue()
        return None
    
class AdvancedPlotFrame(ttk.Frame):
    def __init__(self, parent):
        super().__init__(parent)
        self.setup_ui()
        
    def setup_ui(self):
        # Create figure with subplots
        self.figure = Figure(figsize=(10, 6), dpi=100)
        self.figure.set_facecolor('#f0f0f0')
        
        # Create subplots
        self.temp_ax = self.figure.add_subplot(221)
        self.humidity_ax = self.figure.add_subplot(222)
        self.co2_ax = self.figure.add_subplot(223)
        self.quality_ax = self.figure.add_subplot(224)
        
        self.canvas = FigureCanvasTkAgg(self.figure, self)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Add navigation toolbar
        self.toolbar = NavigationToolbar2Tk(self.canvas, self)
        self.toolbar.update()
        
    def update_plots(self, data: pd.DataFrame):
        # Clear all axes
        for ax in [self.temp_ax, self.humidity_ax, self.co2_ax, self.quality_ax]:
            ax.clear()
            
        # Plot temperature
        self.temp_ax.plot(data['timestamp'], data['temperature'], 'r-')
        self.temp_ax.set_title('Temperature History')
        self.temp_ax.set_ylabel('Temperature (°C)')
        
        # Plot humidity
        self.humidity_ax.plot(data['timestamp'], data['humidity'], 'b-')
        self.humidity_ax.set_title('Humidity History')
        self.humidity_ax.set_ylabel('Humidity (%)')
        
        # Plot CO2
        self.co2_ax.plot(data['timestamp'], data['co2_level'], 'g-')
        self.co2_ax.set_title('CO2 Level History')
        self.co2_ax.set_ylabel('CO2 (ppm)')
        
        # Plot quality score
        self.quality_ax.plot(data['timestamp'], data['quality_score'], 'y-')
        self.quality_ax.set_title('Quality Score History')
        self.quality_ax.set_ylabel('Score')
        
        self.figure.tight_layout()
        self.canvas.draw()

class LocationDashboard(ttk.Frame):
    def __init__(self, parent, location: str, db_manager: DatabaseManager, analytics: AnalyticsEngine):
        super().__init__(parent)
        self.location = location
        self.db_manager = db_manager
        self.analytics = analytics
        self.setup_ui()
        
    def setup_ui(self):
        # Main container with two columns
        left_panel = ttk.Frame(self)
        left_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        right_panel = ttk.Frame(self)
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        # Current readings panel
        readings_frame = ttk.LabelFrame(left_panel, text="Current Readings")
        readings_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.reading_labels = {}
        metrics = ['Temperature', 'Humidity', 'CO2 Level', 'Quality Score']
        for i, metric in enumerate(metrics):
            label = ttk.Label(readings_frame, text=f"{metric}: --")
            label.grid(row=i, column=0, padx=5, pady=2, sticky='w')
            self.reading_labels[metric] = label
            
        # Status indicators
        status_frame = ttk.LabelFrame(left_panel, text="Status")
        status_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.status_labels = {}
        indicators = ['System Status', 'Alert Level', 'Data Quality']
        for i, indicator in enumerate(indicators):
            label = ttk.Label(status_frame, text=f"{indicator}: Normal")
            label.grid(row=i, column=0, padx=5, pady=2, sticky='w')
            self.status_labels[indicator] = label
            
        # Advanced visualization
        self.plot_frame = AdvancedPlotFrame(right_panel)
        self.plot_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Controls panel
        controls_frame = ttk.LabelFrame(left_panel, text="Controls")
        controls_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Button(controls_frame, text="Export Data", 
                  command=self.export_data).pack(padx=5, pady=2)
        ttk.Button(controls_frame, text="Generate Report", 
                  command=self.generate_report).pack(padx=5, pady=2)
        ttk.Button(controls_frame, text="View Analytics", 
                  command=self.show_analytics).pack(padx=5, pady=2)
        
    
    def show_analytics(self):
        # Create analytics window
        analytics_window = tk.Toplevel(self)
        analytics_window.title(f"{self.location} Analytics")
        analytics_window.geometry("1000x800")
        
        # Create main container
        main_frame = ttk.Frame(analytics_window)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Statistics Panel
        stats_frame = ttk.LabelFrame(main_frame, text="Statistical Analysis")
        stats_frame.pack(fill=tk.X, pady=5)
        
        # Get last 24 hours of data
        end_time = datetime.now()
        start_time = end_time - timedelta(hours=24)
        data = self.db_manager.get_readings(self.location, start_time, end_time)
        
        # Create statistics display
        metrics = ['temperature', 'humidity', 'co2_level', 'quality_score']
        for i, metric in enumerate(metrics):
            metric_frame = ttk.Frame(stats_frame)
            metric_frame.pack(fill=tk.X, pady=2)
            
            stats = data[metric].describe()
            
            ttk.Label(metric_frame, 
                    text=f"{metric.replace('_', ' ').title()}:",
                    font=('Helvetica', 10, 'bold')).grid(row=0, column=0, padx=5)
            ttk.Label(metric_frame, 
                    text=f"Mean: {stats['mean']:.2f}").grid(row=0, column=1, padx=5)
            ttk.Label(metric_frame, 
                    text=f"Std: {stats['std']:.2f}").grid(row=0, column=2, padx=5)
            ttk.Label(metric_frame, 
                    text=f"Min: {stats['min']:.2f}").grid(row=0, column=3, padx=5)
            ttk.Label(metric_frame, 
                    text=f"Max: {stats['max']:.2f}").grid(row=0, column=4, padx=5)
        
        # Trend Analysis
        trend_frame = ttk.LabelFrame(main_frame, text="Trend Analysis")
        trend_frame.pack(fill=tk.X, pady=5)
        
        for metric in metrics:
            trend_info = self.analytics.analyze_trend(data, metric)
            trend_label = ttk.Label(trend_frame, 
                                text=f"{metric.replace('_', ' ').title()}: "
                                    f"Trend {trend_info['trend']}, "
                                    f"R² = {trend_info.get('r_squared', 0):.3f}")
            trend_label.pack(padx=5, pady=2)
        
        # Anomaly Detection
        anomaly_frame = ttk.LabelFrame(main_frame, text="Anomaly Detection")
        anomaly_frame.pack(fill=tk.X, pady=5)
        
        for metric in metrics:
            anomalies = self.analytics.detect_anomalies(data, metric)
            if not anomalies.empty:
                anomaly_text = f"{metric.replace('_', ' ').title()}: {len(anomalies)} anomalies detected"
                ttk.Label(anomaly_frame, text=anomaly_text).pack(padx=5, pady=2)
        
        # Visualization Frame
        viz_frame = ttk.LabelFrame(main_frame, text="Data Visualization")
        viz_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        figure = Figure(figsize=(10, 6), dpi=100)
        canvas = FigureCanvasTkAgg(figure, viz_frame)
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Create subplots for each metric
        for i, metric in enumerate(metrics, 1):
            ax = figure.add_subplot(2, 2, i)
            ax.plot(data['timestamp'], data[metric], 'b-')
            ax.set_title(f'{metric.replace("_", " ").title()} Trend')
            ax.tick_params(axis='x', rotation=45)
        
        figure.tight_layout()
        canvas.draw()
        
        # Prediction Panel
        pred_frame = ttk.LabelFrame(main_frame, text="Predictions")
        pred_frame.pack(fill=tk.X, pady=5)
        
        for metric in metrics:
            next_value = self.analytics.predict_next_value(data, metric)
            if next_value is not None:
                pred_text = f"Predicted next {metric.replace('_', ' ')}: {next_value:.2f}"
                ttk.Label(pred_frame, text=pred_text).pack(padx=5, pady=2)
        
        # Add export button
        ttk.Button(main_frame, 
                text="Export Analytics Report",
                command=lambda: self.export_analytics_report(data)).pack(pady=10)

    def export_analytics_report(self, data):
        try:
            file_path = filedialog.asksaveasfilename(
                defaultextension=".pdf",
                filetypes=[("PDF files", "*.pdf")]
            )
            if file_path:
                # Generate PDF report
                doc = SimpleDocTemplate(file_path, pagesize=letter)
                styles = getSampleStyleSheet()
                elements = []
                
                # Add title
                elements.append(Paragraph(f"Analytics Report - {self.location}", 
                                    styles['Title']))
                
                # Add statistics
                elements.append(Paragraph("Statistical Analysis", styles['Heading1']))
                for metric in ['temperature', 'humidity', 'co2_level', 'quality_score']:
                    stats = data[metric].describe()
                    stats_text = (f"{metric.replace('_', ' ').title()}:\n"
                                f"Mean: {stats['mean']:.2f}\n"
                                f"Std: {stats['std']:.2f}\n"
                                f"Min: {stats['min']:.2f}\n"
                                f"Max: {stats['max']:.2f}")
                    elements.append(Paragraph(stats_text, styles['Normal']))
                
                # Build PDF
                doc.build(elements)
                messagebox.showinfo("Success", "Analytics report exported successfully!")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to export report: {str(e)}")
        
    def update_display(self, reading: SensorReading):
        # Update current readings
        self.reading_labels['Temperature'].config(
            text=f"Temperature: {reading.temperature:.1f}°C")
        self.reading_labels['Humidity'].config(
            text=f"Humidity: {reading.humidity:.1f}%")
        self.reading_labels['CO2 Level'].config(
            text=f"CO2 Level: {reading.co2_level:.1f} ppm")
        self.reading_labels['Quality Score'].config(
            text=f"Quality Score: {reading.quality_score:.1f}")
            
        # Update status indicators based on analytics
        health_score = self.analytics.generate_health_score(reading)
        alert_level = "Normal" if health_score > 80 else "Warning" if health_score > 60 else "Critical"
        
        self.status_labels['System Status'].config(
            text=f"System Status: {'Online' if health_score > 60 else 'Degraded'}")
        self.status_labels['Alert Level'].config(
            text=f"Alert Level: {alert_level}")
        self.status_labels['Data Quality'].config(
            text=f"Data Quality: {'Good' if health_score > 80 else 'Fair'}")
            
    def export_data(self):
        try:
            file_path = filedialog.asksaveasfilename(
                defaultextension=".csv",
                filetypes=[
                    ("CSV files", "*.csv"),
                    ("Excel files", "*.xlsx"),
                    ("JSON files", "*.json")
                ]
            )
            if file_path:
                end_time = datetime.now()
                start_time = end_time - timedelta(hours=24)
                data = self.db_manager.export_data(
                    format=file_path.split('.')[-1],
                    start_time=start_time,
                    end_time=end_time
                )
                with open(file_path, 'wb') as f:
                    f.write(data)
                messagebox.showinfo("Success", "Data exported successfully!")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to export data: {str(e)}")
            
    def generate_report(self):
        try:
            file_path = filedialog.asksaveasfilename(
                defaultextension=".pdf",
                filetypes=[("PDF files", "*.pdf")]
            )
            if file_path:
                # Generate report logic here
                messagebox.showinfo("Success", "Report generated successfully!")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to generate report: {str(e)}")
            
    def show_analytics(self):
        # Create analytics window
        analytics_window = tk.Toplevel(self)
        analytics_window.title(f"{self.location} Analytics")
        analytics_window.geometry("800x600")
        
        # Add analytics content
        ttk.Label(analytics_window, 
                 text="Advanced Analytics Dashboard",
                 font=('Helvetica', 16, 'bold')).pack(pady=10)
                 
        # Add more analytics widgets here

class MainApplication(tk.Tk):
    def __init__(self):
        super().__init__()
        
        self.title("Advanced Food Traceability System")
        self.state('zoomed')
        
        # Initialize managers
        self.db_manager = DatabaseManager()
        self.analytics = AnalyticsEngine()
        
        self.setup_ui()
        self.start_monitoring()
        
    def setup_ui(self):
        # Configure styles
        style = ttk.Style()
        style.configure('Header.TLabel', font=('Helvetica', 14, 'bold'))
        
        # Main container
        main_container = ttk.Frame(self)
        main_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Header
        header_frame = ttk.Frame(main_container)
        header_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(header_frame, 
                 text="Food Traceability Dashboard",
                 style='Header.TLabel').pack(side=tk.LEFT)
                 
        # System status
        self.status_label = ttk.Label(header_frame, 
                                    text="System Status: Online",
                                    style='Header.TLabel')
        self.status_label.pack(side=tk.RIGHT)
        
        # Location tabs
        self.notebook = ttk.Notebook(main_container)
        self.notebook.pack(fill=tk.BOTH, expand=True)
        
        # Create location dashboards
        self.dashboards = {}
        locations = ['Farm', 'Transport', 'Storage', 'Kitchen']
        
        for location in locations:
            dashboard = LocationDashboard(self.notebook, location, 
                                        self.db_manager, self.analytics)
            self.dashboards[location] = dashboard
            self.notebook.add(dashboard, text=location)
            
    def start_monitoring(self):
        for location, dashboard in self.dashboards.items():
            # Simulate sensor reading
            reading = SensorReading(
                timestamp=datetime.now(),
                temperature=random.uniform(2, 25),
                humidity=random.uniform(30, 70),
                pressure=random.uniform(1010, 1015),
                co2_level=random.uniform(300, 500),
                light_level=random.uniform(100, 1000),
                quality_score=random.uniform(90, 100),
                location=location,
                batch_id=f"BATCH_{datetime.now().strftime('%Y%m%d')}",
                equipment_id=f"EQ_{location[:3].upper()}_001",
                operator_id=f"OP_{location[:3].upper()}_001"
            )
            
            # Update dashboard
            dashboard.update_display(reading)
            
            # Save to database
            self.db_manager.save_reading(reading)
        
        # Schedule next update
        self.after(1000, self.start_monitoring)

if __name__ == "__main__":
    app = MainApplication()
    app.mainloop()