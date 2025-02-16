import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from dataclasses import dataclass
import random
from typing import Dict, List
import sys

# Set matplotlib style to a built-in style
plt.style.use('ggplot')

# Data Structures
@dataclass
class SensorReading:
    timestamp: datetime
    temperature: float
    humidity: float
    pressure: float
    co2_level: float
    light_level: float
    quality_score: float

class SensorSimulator:
    def __init__(self, location: str):
        self.location = location
        self.base_values = {
            'farm': {'temp': 22, 'humidity': 60, 'pressure': 1013, 'co2': 400, 'light': 1000},
            'transport': {'temp': 4, 'humidity': 40, 'pressure': 1013, 'co2': 350, 'light': 200},
            'storage': {'temp': 3, 'humidity': 35, 'pressure': 1013, 'co2': 300, 'light': 100},
            'kitchen': {'temp': 18, 'humidity': 50, 'pressure': 1013, 'co2': 450, 'light': 800}
        }

    def get_reading(self) -> SensorReading:
        base = self.base_values[self.location]
        return SensorReading(
            timestamp=datetime.now(),
            temperature=base['temp'] + random.uniform(-1, 1),
            humidity=base['humidity'] + random.uniform(-5, 5),
            pressure=base['pressure'] + random.uniform(-2, 2),
            co2_level=base['co2'] + random.uniform(-20, 20),
            light_level=base['light'] + random.uniform(-50, 50),
            quality_score=random.uniform(90, 100)
        )

class LocationFrame(ttk.Frame):
    def __init__(self, parent, location):
        super().__init__(parent)
        self.location = location
        self.setup_ui()
        
    def setup_ui(self):
        # Sensor readings frame
        readings_frame = ttk.LabelFrame(self, text=f"{self.location.capitalize()} Sensor Readings")
        readings_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.reading_labels = {}
        metrics = ['Temperature (°C)', 'Humidity (%)', 'Pressure (hPa)', 
                  'CO2 (ppm)', 'Light (lux)', 'Quality Score']
        
        for i, metric in enumerate(metrics):
            label = ttk.Label(readings_frame, text=f"{metric}: --")
            label.grid(row=i//3, column=i%3, padx=10, pady=5, sticky='w')
            self.reading_labels[metric] = label
            
        # Graph frame
        graph_frame = ttk.LabelFrame(self, text="Temperature History")
        graph_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.figure = Figure(figsize=(6, 4), dpi=100)
        self.ax = self.figure.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.figure, graph_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

    def update_display(self, reading: SensorReading):
        self.reading_labels['Temperature (°C)'].config(
            text=f"Temperature (°C): {reading.temperature:.1f}")
        self.reading_labels['Humidity (%)'].config(
            text=f"Humidity (%): {reading.humidity:.1f}")
        self.reading_labels['Pressure (hPa)'].config(
            text=f"Pressure (hPa): {reading.pressure:.1f}")
        self.reading_labels['CO2 (ppm)'].config(
            text=f"CO2 (ppm): {reading.co2_level:.1f}")
        self.reading_labels['Light (lux)'].config(
            text=f"Light (lux): {reading.light_level:.1f}")
        self.reading_labels['Quality Score'].config(
            text=f"Quality Score: {reading.quality_score:.1f}")

    def update_plot(self, df: pd.DataFrame):
        self.ax.clear()
        self.ax.plot(df['timestamp'], df['temperature'], '-b', linewidth=2)
        self.ax.set_title(f'{self.location.capitalize()} Temperature History')
        self.ax.set_xlabel('Time')
        self.ax.set_ylabel('Temperature (°C)')
        self.ax.tick_params(axis='x', rotation=45)
        self.figure.tight_layout()
        self.canvas.draw()

class TraceabilityDashboard(tk.Tk):
    def __init__(self):
        super().__init__()

        self.title("Food Traceability Dashboard")
        self.geometry("1200x800")
        self.configure(bg='white')

        # Initialize data
        self.locations = ['farm', 'transport', 'storage', 'kitchen']
        self.sensors = {loc: SensorSimulator(loc) for loc in self.locations}
        self.data = {loc: [] for loc in self.locations}
        self.max_history = 50

        self.setup_ui()
        self.update_dashboard()

    def setup_ui(self):
        # Style configuration
        style = ttk.Style()
        style.configure('TLabel', font=('Helvetica', 10))
        style.configure('TLabelframe', font=('Helvetica', 11, 'bold'))
        style.configure('Header.TLabel', font=('Helvetica', 14, 'bold'))

        # Main container
        main_container = ttk.Frame(self)
        main_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Header with title and overall score
        header_frame = ttk.Frame(main_container)
        header_frame.pack(fill=tk.X, pady=(0, 10))
        
        title_label = ttk.Label(header_frame, 
                               text="Food Traceability Dashboard",
                               style='Header.TLabel')
        title_label.pack(side=tk.LEFT)
        
        self.overall_score = ttk.Label(header_frame, 
                                     text="Overall Score: 96.5",
                                     style='Header.TLabel')
        self.overall_score.pack(side=tk.RIGHT)

        # Notebook for location tabs
        self.notebook = ttk.Notebook(main_container)
        self.notebook.pack(fill=tk.BOTH, expand=True)

        # Create location tabs
        self.location_frames = {}
        for location in self.locations:
            frame = LocationFrame(self.notebook, location)
            self.location_frames[location] = frame
            self.notebook.add(frame, text=location.capitalize())

    def update_dashboard(self):
        # Update sensor data
        for location in self.locations:
            reading = self.sensors[location].get_reading()
            self.data[location].append(reading)
            if len(self.data[location]) > self.max_history:
                self.data[location].pop(0)

            # Update displays
            frame = self.location_frames[location]
            frame.update_display(reading)
            
            # Update plots
            df = pd.DataFrame([vars(r) for r in self.data[location]])
            frame.update_plot(df)

        # Calculate and update overall score
        scores = [reading.quality_score for loc in self.locations 
                 for reading in [self.data[loc][-1]]]
        overall = sum(scores) / len(scores)
        self.overall_score.config(
            text=f"Overall Score: {overall:.1f}")

        # Schedule next update
        self.after(1000, self.update_dashboard)

def main():
    app = TraceabilityDashboard()
    app.mainloop()

if __name__ == "__main__":
    main()