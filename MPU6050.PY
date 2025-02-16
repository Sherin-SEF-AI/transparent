import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import serial
import serial.tools.list_ports
import threading
import time
from datetime import datetime
import csv
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import json
import os

class EnvironmentalMonitor:
    def __init__(self, root):
        """Initialize the environmental monitoring system with all necessary components"""
        # Set up the main window
        self.root = root
        self.root.title("Environmental Monitoring System")
        self.root.geometry("1200x800")
        
        # Initialize all data storage variables
        self.data_buffer = []          # Stores sensor readings
        self.buffer_size = 100         # Maximum number of readings to keep
        self.threshold_vars = {}       # Stores alert threshold settings
        self.motion_labels = {}        # UI labels for motion sensor data
        self.env_labels = {}           # UI labels for environmental data
        self.gas_labels = {}           # UI labels for gas sensor data
        
        # Initialize communication variables
        self.serial = None
        self.running = False
        self.thread = None
        self.logging = False
        self.log_file = None
        self.csv_writer = None
        
        # Set up the user interface
        self.create_tabs()
        self.create_dashboard()
        self.setup_plots()
        self.setup_settings()
        self.setup_analysis()
        
        # Initialize default threshold values
        self.initialize_thresholds()
        
        # Set up plot updates
        self.plot_update_interval = 1000  # milliseconds
        self.root.after(self.plot_update_interval, self.update_plots)

    def create_tabs(self):
        """Create the tabbed interface for different sections of the application"""
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Create individual tabs
        self.main_tab = ttk.Frame(self.notebook)
        self.plots_tab = ttk.Frame(self.notebook)
        self.settings_tab = ttk.Frame(self.notebook)
        self.analysis_tab = ttk.Frame(self.notebook)
        
        # Add tabs to notebook
        self.notebook.add(self.main_tab, text="Dashboard")
        self.notebook.add(self.plots_tab, text="Visualizations")
        self.notebook.add(self.settings_tab, text="Settings")
        self.notebook.add(self.analysis_tab, text="Analysis")

    def initialize_thresholds(self):
        """Set up default threshold values for sensor alerts"""
        default_thresholds = {
            'temperature_high': 30,
            'temperature_low': 15,
            'humidity_high': 70,
            'mq2_threshold': 100,
            'mq135_threshold': 100
        }
        
        for key, value in default_thresholds.items():
            self.threshold_vars[key] = tk.StringVar(value=str(value))

    def create_dashboard(self):
        """Create the main dashboard interface with sensor readings and controls"""
        # Create connection frame
        conn_frame = ttk.LabelFrame(self.main_tab, text="Connection Settings")
        conn_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(conn_frame, text="Port:").pack(side=tk.LEFT, padx=5)
        self.port_var = tk.StringVar()
        self.port_combo = ttk.Combobox(conn_frame, textvariable=self.port_var)
        self.port_combo['values'] = self.get_available_ports()
        self.port_combo.pack(side=tk.LEFT, padx=5)
        
        self.connect_button = ttk.Button(conn_frame, text="Connect",
                                       command=self.toggle_connection)
        self.connect_button.pack(side=tk.LEFT, padx=5)
        
        # Create sensor frames
        self.create_sensor_displays()
        
        # Create logging and alert frames
        self.create_logging_frame()
        self.create_alerts_frame()

    def create_sensor_displays(self):
        """Create display areas for different sensor readings"""
        # Motion sensor frame
        motion_frame = ttk.LabelFrame(self.main_tab, text="Motion Sensor (MPU6050)")
        motion_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Create labels for motion sensor readings
        motion_params = ['Accel X', 'Accel Y', 'Accel Z',
                        'Gyro X', 'Gyro Y', 'Gyro Z']
        
        for i, param in enumerate(motion_params):
            ttk.Label(motion_frame, text=param + ":").grid(
                row=i//3, column=(i%3)*2, padx=5, pady=5)
            self.motion_labels[param] = ttk.Label(motion_frame, text="0.0")
            self.motion_labels[param].grid(
                row=i//3, column=(i%3)*2+1, padx=5, pady=5)

        # Environmental sensors frame
        env_frame = ttk.LabelFrame(self.main_tab, text="Environmental Sensors")
        env_frame.pack(fill=tk.X, padx=5, pady=5)
        
        env_params = ['Temperature', 'Humidity', 'Distance']
        for i, param in enumerate(env_params):
            ttk.Label(env_frame, text=param + ":").grid(
                row=0, column=i*2, padx=5, pady=5)
            self.env_labels[param] = ttk.Label(env_frame, text="0.0")
            self.env_labels[param].grid(
                row=0, column=i*2+1, padx=5, pady=5)

        # Gas sensors frame
        gas_frame = ttk.LabelFrame(self.main_tab, text="Gas Sensors")
        gas_frame.pack(fill=tk.X, padx=5, pady=5)
        
        gas_params = ['MQ2 Level', 'MQ135 Level']
        for i, param in enumerate(gas_params):
            ttk.Label(gas_frame, text=param + ":").grid(
                row=0, column=i*2, padx=5, pady=5)
            self.gas_labels[param] = ttk.Label(gas_frame, text="0.0")
            self.gas_labels[param].grid(
                row=0, column=i*2+1, padx=5, pady=5)

    def create_logging_frame(self):
        """Create controls for data logging"""
        log_frame = ttk.LabelFrame(self.main_tab, text="Data Logging")
        log_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.log_button = ttk.Button(log_frame, text="Start Logging",
                                   command=self.toggle_logging)
        self.log_button.pack(side=tk.LEFT, padx=5)
        
        self.log_status = ttk.Label(log_frame, text="Logging: Stopped")
        self.log_status.pack(side=tk.LEFT, padx=5)

    def create_alerts_frame(self):
        """Create area for displaying alerts"""
        self.alerts_frame = ttk.LabelFrame(self.main_tab, text="Alerts")
        self.alerts_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.alerts_text = tk.Text(self.alerts_frame, height=3)
        self.alerts_text.pack(fill=tk.X, padx=5, pady=5)

    def setup_plots(self):
        """Set up real-time data visualization using Matplotlib"""
        # Create three subplots for different sensor groups
        self.fig, (self.ax1, self.ax2, self.ax3) = plt.subplots(3, 1, figsize=(10, 8))
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.plots_tab)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Initialize empty line objects for real-time plotting
        self.lines = {}
        self.initialize_plot_lines()

    def initialize_plot_lines(self):
        """Initialize the plot lines for each sensor type with appropriate formatting"""
        # Clear existing plots
        for ax in [self.ax1, self.ax2, self.ax3]:
            ax.clear()
        
        # Temperature and Humidity plot (top)
        self.lines['temperature'], = self.ax1.plot([], [], 'r-', label='Temperature (°C)')
        self.lines['humidity'], = self.ax1.plot([], [], 'b-', label='Humidity (%)')
        self.ax1.set_title('Environmental Conditions')
        self.ax1.legend()
        self.ax1.grid(True)
        
        # Gas Sensors plot (middle)
        self.lines['mq2'], = self.ax2.plot([], [], 'g-', label='MQ2 Gas Level')
        self.lines['mq135'], = self.ax2.plot([], [], 'm-', label='MQ135 Gas Level')
        self.ax2.set_title('Gas Sensor Readings')
        self.ax2.legend()
        self.ax2.grid(True)
        
        # Acceleration plot (bottom)
        self.lines['accel_x'], = self.ax3.plot([], [], 'r-', label='X-axis')
        self.lines['accel_y'], = self.ax3.plot([], [], 'g-', label='Y-axis')
        self.lines['accel_z'], = self.ax3.plot([], [], 'b-', label='Z-axis')
        self.ax3.set_title('Acceleration Data')
        self.ax3.legend()
        self.ax3.grid(True)

    def update_plots(self):
        """Update all plots with the latest sensor data"""
        if len(self.data_buffer) > 0:
            # Convert data buffer to pandas DataFrame for easier handling
            df = pd.DataFrame(self.data_buffer)
            x_range = range(len(df))  # Time points for x-axis
            
            # Update environmental data plot
            self.lines['temperature'].set_data(x_range, df['temperature'])
            self.lines['humidity'].set_data(x_range, df['humidity'])
            
            # Update gas sensor plot
            self.lines['mq2'].set_data(x_range, df['mq2'])
            self.lines['mq135'].set_data(x_range, df['mq135'])
            
            # Update acceleration plot
            self.lines['accel_x'].set_data(x_range, df['accel_x'])
            self.lines['accel_y'].set_data(x_range, df['accel_y'])
            self.lines['accel_z'].set_data(x_range, df['accel_z'])
            
            # Adjust plot limits to show all data
            for ax in [self.ax1, self.ax2, self.ax3]:
                ax.relim()
                ax.autoscale_view()
            
            # Redraw the canvas
            self.canvas.draw()
        
        # Schedule next update
        self.root.after(self.plot_update_interval, self.update_plots)

    def get_available_ports(self):
        """Get list of available serial ports"""
        return [port.device for port in serial.tools.list_ports.comports()]

    def toggle_connection(self):
        """Handle connection/disconnection to Arduino"""
        if self.serial and self.serial.is_open:
            # Disconnect if currently connected
            self.running = False
            time.sleep(0.5)  # Allow time for thread to close
            self.serial.close()
            self.connect_button.config(text="Connect")
            messagebox.showinfo("Disconnected", "Serial connection closed")
        else:
            # Attempt to connect to selected port
            try:
                port = self.port_var.get()
                if not port:
                    messagebox.showerror("Error", "Please select a port")
                    return
                
                self.serial = serial.Serial(port, 9600, timeout=1)
                self.running = True
                self.thread = threading.Thread(target=self.read_sensor_data)
                self.thread.daemon = True  # Thread will close with main program
                self.thread.start()
                
                self.connect_button.config(text="Disconnect")
                messagebox.showinfo("Connected", f"Connected to {port}")
            except Exception as e:
                messagebox.showerror("Connection Error", f"Failed to connect: {str(e)}")

    def read_sensor_data(self):
        """Read and process incoming sensor data from Arduino"""
        while self.running:
            try:
                if self.serial and self.serial.in_waiting:
                    # Read line from serial port
                    line = self.serial.readline().decode('utf-8').strip()
                    values = line.split(',')
                    
                    # Process data if we have all expected values
                    if len(values) == 11:
                        data = {
                            'timestamp': datetime.now(),
                            'accel_x': float(values[0]),
                            'accel_y': float(values[1]),
                            'accel_z': float(values[2]),
                            'gyro_x': float(values[3]),
                            'gyro_y': float(values[4]),
                            'gyro_z': float(values[5]),
                            'temperature': float(values[6]),
                            'humidity': float(values[7]),
                            'mq2': float(values[8]),
                            'mq135': float(values[9]),
                            'distance': float(values[10])
                        }
                        
                        # Update display and process data
                        self.root.after(0, self.update_display, data)
                        
            except Exception as e:
                print(f"Error reading sensor data: {e}")
            
            time.sleep(0.1)  # Prevent CPU overload

    def update_display(self, data):
        """Update all display elements with new sensor readings"""
        try:
            # Update motion sensor labels
            for axis in ['X', 'Y', 'Z']:
                self.motion_labels[f'Accel {axis}'].config(
                    text=f"{data[f'accel_{axis.lower()}']:.2f}")
                self.motion_labels[f'Gyro {axis}'].config(
                    text=f"{data[f'gyro_{axis.lower()}']:.2f}")
            
            # Update environmental sensor labels
            self.env_labels['Temperature'].config(text=f"{data['temperature']:.1f}°C")
            self.env_labels['Humidity'].config(text=f"{data['humidity']:.1f}%")
            self.env_labels['Distance'].config(text=f"{data['distance']:.1f} cm")
            
            # Update gas sensor labels
            self.gas_labels['MQ2 Level'].config(text=f"{data['mq2']:.2f}")
            self.gas_labels['MQ135 Level'].config(text=f"{data['mq135']:.2f}")
            
            # Add data to buffer and maintain buffer size
            self.data_buffer.append(data)
            if len(self.data_buffer) > self.buffer_size:
                self.data_buffer.pop(0)
            
            # Check for alerts and log data if enabled
            self.check_alerts(data)
            if self.logging:
                self.log_data(data)
                
        except Exception as e:
            print(f"Error updating display: {e}")

    def check_alerts(self, data):
        """Monitor sensor readings and trigger alerts when thresholds are exceeded"""
        try:
            alerts = []
            
            # Check temperature thresholds
            temp_high = float(self.threshold_vars['temperature_high'].get())
            temp_low = float(self.threshold_vars['temperature_low'].get())
            
            if data['temperature'] > temp_high:
                alerts.append(f"High Temperature Warning: {data['temperature']:.1f}°C")
            elif data['temperature'] < temp_low:
                alerts.append(f"Low Temperature Warning: {data['temperature']:.1f}°C")
            
            # Check humidity threshold
            humidity_high = float(self.threshold_vars['humidity_high'].get())
            if data['humidity'] > humidity_high:
                alerts.append(f"High Humidity Warning: {data['humidity']:.1f}%")
            
            # Check gas sensor thresholds
            mq2_threshold = float(self.threshold_vars['mq2_threshold'].get())
            mq135_threshold = float(self.threshold_vars['mq135_threshold'].get())
            
            if data['mq2'] > mq2_threshold:
                alerts.append(f"MQ2 Gas Level Alert: {data['mq2']:.2f}")
            if data['mq135'] > mq135_threshold:
                alerts.append(f"MQ135 Gas Level Alert: {data['mq135']:.2f}")
            
            # Display any triggered alerts
            if alerts:
                self.display_alerts(alerts)
                
        except ValueError as e:
            print(f"Alert threshold error: {e}")

    def display_alerts(self, alerts):
        """Display alert messages with timestamps in the alerts text widget"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        alert_text = f"=== Alerts at {timestamp} ===\n"
        
        for alert in alerts:
            alert_text += f"• {alert}\n"
        
        self.alerts_text.delete('1.0', tk.END)
        self.alerts_text.insert(tk.END, alert_text)
        self.alerts_text.see(tk.END)

    def toggle_logging(self):
        """Start or stop data logging to CSV file"""
        if not self.logging:
            try:
                # Create filename with timestamp
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                filename = f"sensor_log_{timestamp}.csv"
                
                # Open file and initialize CSV writer
                self.log_file = open(filename, 'w', newline='')
                self.csv_writer = csv.writer(self.log_file)
                
                # Write header row
                self.csv_writer.writerow([
                    'Timestamp', 'Temperature', 'Humidity', 'Distance',
                    'MQ2', 'MQ135', 'Accel_X', 'Accel_Y', 'Accel_Z',
                    'Gyro_X', 'Gyro_Y', 'Gyro_Z'
                ])
                
                self.logging = True
                self.log_button.config(text="Stop Logging")
                self.log_status.config(text="Logging: Active")
                
            except Exception as e:
                messagebox.showerror("Logging Error", f"Failed to start logging: {str(e)}")
                
        else:
            # Stop logging and close file
            self.logging = False
            if self.log_file:
                self.log_file.close()
            self.log_button.config(text="Start Logging")
            self.log_status.config(text="Logging: Stopped")

    def log_data(self, data):
        """Write sensor data to CSV file"""
        if self.logging and self.csv_writer:
            try:
                row = [
                    data['timestamp'].strftime('%Y-%m-%d %H:%M:%S'),
                    data['temperature'],
                    data['humidity'],
                    data['distance'],
                    data['mq2'],
                    data['mq135'],
                    data['accel_x'],
                    data['accel_y'],
                    data['accel_z'],
                    data['gyro_x'],
                    data['gyro_y'],
                    data['gyro_z']
                ]
                self.csv_writer.writerow(row)
                self.log_file.flush()  # Ensure data is written immediately
                
            except Exception as e:
                print(f"Logging error: {e}")
                self.toggle_logging()  # Stop logging on error

    def setup_settings(self):
        """Create the settings interface for configuring system parameters"""
        settings_frame = ttk.LabelFrame(self.settings_tab, text="Alert Thresholds")
        settings_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Create threshold settings
        thresholds = [
            ('temperature_high', "High Temperature (°C):", 30),
            ('temperature_low', "Low Temperature (°C):", 15),
            ('humidity_high', "High Humidity (%):", 70),
            ('mq2_threshold', "MQ2 Gas Alert Level:", 100),
            ('mq135_threshold', "MQ135 Gas Alert Level:", 100)
        ]
        
        for key, label, default in thresholds:
            frame = ttk.Frame(settings_frame)
            frame.pack(fill=tk.X, pady=2)
            
            ttk.Label(frame, text=label).pack(side=tk.LEFT, padx=5)
            entry = ttk.Entry(frame, textvariable=self.threshold_vars[key], width=10)
            entry.pack(side=tk.LEFT, padx=5)
        
        # Add control buttons
        button_frame = ttk.Frame(settings_frame)
        button_frame.pack(fill=tk.X, pady=5)
        
        ttk.Button(button_frame, text="Save Settings", 
                  command=self.save_settings).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Load Settings", 
                  command=self.load_settings).pack(side=tk.LEFT, padx=5)

    def save_settings(self):
        """Save current settings to JSON file"""
        try:
            settings = {key: var.get() for key, var in self.threshold_vars.items()}
            
            with open('sensor_settings.json', 'w') as f:
                json.dump(settings, f, indent=4)
                
            messagebox.showinfo("Success", "Settings saved successfully")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save settings: {str(e)}")

    def load_settings(self):
        """Load settings from JSON file"""
        try:
            if os.path.exists('sensor_settings.json'):
                with open('sensor_settings.json', 'r') as f:
                    settings = json.load(f)
                    
                for key, value in settings.items():
                    if key in self.threshold_vars:
                        self.threshold_vars[key].set(value)
                        
                messagebox.showinfo("Success", "Settings loaded successfully")
                
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load settings: {str(e)}")

    def on_closing(self):
        """Clean up resources when closing the application"""
        try:
            # Stop data collection
            self.running = False
            time.sleep(0.5)  # Allow thread to close
            
            # Close serial connection
            if self.serial and self.serial.is_open:
                self.serial.close()
            
            # Close log file
            if self.logging and self.log_file:
                self.log_file.close()
            
            # Save current settings
            self.save_settings()
            
            # Destroy the window
            self.root.destroy()
            
        except Exception as e:
            print(f"Error during cleanup: {e}")
            self.root.destroy()

    def __init__(self, root):
        """Initialize the environmental monitoring system with all necessary components"""
        # Set up the main window
        self.root = root
        self.root.title("Environmental Monitoring System")
        self.root.geometry("1200x800")
        
        # Initialize data storage variables
        self.data_buffer = []          # Stores sensor readings
        self.buffer_size = 100         # Maximum number of readings to keep
        
        # Initialize threshold variables with default values
        self.threshold_vars = {
            'temperature_high': tk.StringVar(value="30"),
            'temperature_low': tk.StringVar(value="15"),
            'humidity_high': tk.StringVar(value="70"),
            'mq2_threshold': tk.StringVar(value="100"),
            'mq135_threshold': tk.StringVar(value="100")
        }
        
        # Initialize display label dictionaries
        self.motion_labels = {}        # UI labels for motion sensor data
        self.env_labels = {}           # UI labels for environmental data
        self.gas_labels = {}           # UI labels for gas sensor data
        
        # Initialize communication variables
        self.serial = None
        self.running = False
        self.thread = None
        self.logging = False
        self.log_file = None
        self.csv_writer = None
        
        # Set up the user interface
        self.create_tabs()
        self.create_dashboard()
        self.setup_plots()
        self.setup_settings()
        self.setup_analysis()
        
        # Set up plot updates
        self.plot_update_interval = 1000  # milliseconds
        self.root.after(self.plot_update_interval, self.update_plots)

    def setup_analysis(self):
        """
        Creates the analysis tab interface where users can analyze sensor data
        and generate reports. This includes options for data visualization,
        statistical analysis, and data export features.
        """
        # Create main frame for analysis tools
        analysis_frame = ttk.LabelFrame(self.analysis_tab, text="Data Analysis Tools")
        analysis_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        # Create a frame for data visualization controls
        viz_frame = ttk.Frame(analysis_frame)
        viz_frame.pack(fill=tk.X, pady=5)
        
        # Add visualization options
        ttk.Label(viz_frame, text="Select Time Range:").pack(side=tk.LEFT, padx=5)
        self.time_range = tk.StringVar(value="all")
        ttk.Radiobutton(viz_frame, text="All Data", 
                        variable=self.time_range, value="all").pack(side=tk.LEFT)
        ttk.Radiobutton(viz_frame, text="Last Hour", 
                        variable=self.time_range, value="hour").pack(side=tk.LEFT)
        ttk.Radiobutton(viz_frame, text="Last 10 Minutes", 
                        variable=self.time_range, value="10min").pack(side=tk.LEFT)

        # Create export controls frame
        export_frame = ttk.Frame(analysis_frame)
        export_frame.pack(fill=tk.X, pady=10)
        
        # Add export buttons
        ttk.Button(export_frame, text="Export to CSV",
                command=lambda: self.export_data("csv")).pack(side=tk.LEFT, padx=5)
        ttk.Button(export_frame, text="Export to Excel",
                command=lambda: self.export_data("excel")).pack(side=tk.LEFT, padx=5)
        
        # Create statistics frame
        stats_frame = ttk.LabelFrame(analysis_frame, text="Statistical Analysis")
        stats_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        # Add text widget for displaying statistics
        self.stats_text = tk.Text(stats_frame, height=10)
        self.stats_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Add update button for statistics
        ttk.Button(stats_frame, text="Update Statistics",
                command=self.update_statistics).pack(pady=5)

    def update_statistics(self):
        """
        Calculates and displays basic statistics for all sensor readings.
        This provides users with insights about their sensor data.
        """
        if not self.data_buffer:
            self.stats_text.delete('1.0', tk.END)
            self.stats_text.insert(tk.END, "No data available for analysis")
            return

        # Convert data buffer to pandas DataFrame for analysis
        df = pd.DataFrame(self.data_buffer)
        
        # Calculate basic statistics
        stats_text = "=== Sensor Data Statistics ===\n\n"
        
        # Temperature statistics
        stats_text += "Temperature:\n"
        stats_text += f"  Average: {df['temperature'].mean():.1f}°C\n"
        stats_text += f"  Maximum: {df['temperature'].max():.1f}°C\n"
        stats_text += f"  Minimum: {df['temperature'].min():.1f}°C\n\n"
        
        # Humidity statistics
        stats_text += "Humidity:\n"
        stats_text += f"  Average: {df['humidity'].mean():.1f}%\n"
        stats_text += f"  Maximum: {df['humidity'].max():.1f}%\n"
        stats_text += f"  Minimum: {df['humidity'].min():.1f}%\n\n"
        
        # Gas sensor statistics
        stats_text += "Gas Levels:\n"
        stats_text += f"  MQ2 Average: {df['mq2'].mean():.2f}\n"
        stats_text += f"  MQ2 Maximum: {df['mq2'].max():.2f}\n"
        stats_text += f"  MQ135 Average: {df['mq135'].mean():.2f}\n"
        stats_text += f"  MQ135 Maximum: {df['mq135'].max():.2f}\n"
        
        # Update the statistics display
        self.stats_text.delete('1.0', tk.END)
        self.stats_text.insert(tk.END, stats_text)

# Create and start the application
if __name__ == "__main__":
    try:
        root = tk.Tk()
        app = EnvironmentalMonitor(root)
        root.protocol("WM_DELETE_WINDOW", app.on_closing)
        root.mainloop()
    except Exception as e:
        print(f"Application error: {e}")
        messagebox.showerror("Error", f"Application failed to start: {str(e)}")