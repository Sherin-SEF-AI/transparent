import serial
import tkinter as tk
from tkinter import ttk
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import pandas as pd
from collections import deque
import threading
import time

class SensorDashboard:
    def __init__(self, root):
        self.root = root
        self.root.title("Sensor Dashboard")
        
        # Initialize data storage
        self.max_points = 100
        self.data = {
            'temperature': deque(maxlen=self.max_points),
            'humidity': deque(maxlen=self.max_points),
            'mq2': deque(maxlen=self.max_points),
            'mq135': deque(maxlen=self.max_points),
            'distance': deque(maxlen=self.max_points),
            'acceleration_x': deque(maxlen=self.max_points),
            'acceleration_y': deque(maxlen=self.max_points),
            'acceleration_z': deque(maxlen=self.max_points),
            'gyro_x': deque(maxlen=self.max_points),
            'gyro_y': deque(maxlen=self.max_points),
            'gyro_z': deque(maxlen=self.max_points)
        }
        
        # Create frames
        self.readings_frame = ttk.LabelFrame(root, text="Current Readings")
        self.readings_frame.grid(row=0, column=0, padx=10, pady=5, sticky="nsew")
        
        self.graph_frame = ttk.LabelFrame(root, text="Visualizations")
        self.graph_frame.grid(row=1, column=0, padx=10, pady=5, sticky="nsew")
        
        # Create labels for current readings
        self.labels = {}
        row = 0
        for sensor in self.data.keys():
            ttk.Label(self.readings_frame, text=f"{sensor}:").grid(row=row, column=0, padx=5, pady=2)
            self.labels[sensor] = ttk.Label(self.readings_frame, text="0")
            self.labels[sensor].grid(row=row, column=1, padx=5, pady=2)
            row += 1
        
        # Create plots
        self.fig, self.axes = plt.subplots(3, 2, figsize=(12, 8))
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.graph_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Start serial communication
        self.serial_port = serial.Serial('COM5', 9600)  # Change COM port as needed
        
        # Start data collection thread
        self.running = True
        self.thread = threading.Thread(target=self.update_data)
        self.thread.start()
        
        # Schedule plot updates
        self.root.after(1000, self.update_plots)
    
    def update_data(self):
        while self.running:
            if self.serial_port.in_waiting:
                try:
                    line = self.serial_port.readline().decode('utf-8').strip()
                    values = list(map(float, line.split(',')))
                    
                    # Update data queues
                    sensors = list(self.data.keys())
                    for i, value in enumerate(values):
                        self.data[sensors[i]].append(value)
                        
                        # Update labels
                        self.root.after(0, lambda s=sensors[i], v=value: 
                                      self.labels[s].config(text=f"{v:.2f}"))
                except:
                    pass
            time.sleep(0.1)
    
    def update_plots(self):
        self.fig.clear()
        
        # Create plots using seaborn
        ax1 = self.fig.add_subplot(321)
        sns.lineplot(data=list(self.data['temperature']), ax=ax1)
        ax1.set_title('Temperature')
        
        ax2 = self.fig.add_subplot(322)
        sns.lineplot(data=list(self.data['humidity']), ax=ax2)
        ax2.set_title('Humidity')
        
        ax3 = self.fig.add_subplot(323)
        sns.lineplot(data=list(self.data['mq2']), ax=ax3)
        ax3.set_title('MQ2 Gas')
        
        ax4 = self.fig.add_subplot(324)
        sns.lineplot(data=list(self.data['mq135']), ax=ax4)
        ax4.set_title('MQ135 Air Quality')
        
        ax5 = self.fig.add_subplot(325)
        sns.lineplot(data=list(self.data['distance']), ax=ax5)
        ax5.set_title('Distance')
        
        ax6 = self.fig.add_subplot(326)
        sns.lineplot(data=list(self.data['acceleration_x']), label='X')
        sns.lineplot(data=list(self.data['acceleration_y']), label='Y')
        sns.lineplot(data=list(self.data['acceleration_z']), label='Z')
        ax6.set_title('Acceleration')
        
        self.fig.tight_layout()
        self.canvas.draw()
        
        # Schedule next update
        self.root.after(1000, self.update_plots)
    
    def on_closing(self):
        self.running = False
        self.thread.join()
        self.serial_port.close()
        self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = SensorDashboard(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()