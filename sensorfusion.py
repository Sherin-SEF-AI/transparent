import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Optional
from datetime import datetime
import json

@dataclass
class SensorReading:
    sensor_id: str
    sensor_type: str
    value: float
    timestamp: datetime
    unit: str
    location: str
    confidence: float

class SensorFusionSystem:
    def __init__(self):
        self.sensor_weights = {
            'temperature': 0.25,
            'gas': 0.15,
            'chemical': 0.15,
            'microbial': 0.20,
            'physical': 0.10,
            'ph': 0.05,
            'allergen': 0.10
        }
        
        # Thresholds for different sensor types
        self.thresholds = {
            'temperature': {'min': 0, 'max': 5, 'unit': 'C'},  # For cold storage
            'gas': {'co2_max': 1000, 'voc_max': 500, 'unit': 'ppm'},
            'ph': {'min': 3.5, 'max': 7.0, 'unit': 'pH'},
            'microbial': {'max': 100, 'unit': 'cfu/g'},
        }
        
        self.current_readings: Dict[str, List[SensorReading]] = {}
        self.alerts = []

    def add_sensor_reading(self, reading: SensorReading):
        """Add a new sensor reading to the system"""
        if reading.sensor_type not in self.current_readings:
            self.current_readings[reading.sensor_type] = []
        
        self.current_readings[reading.sensor_type].append(reading)
        self._check_thresholds(reading)

    def _check_thresholds(self, reading: SensorReading):
        """Check if sensor reading exceeds defined thresholds"""
        if reading.sensor_type in self.thresholds:
            threshold = self.thresholds[reading.sensor_type]
            
            if 'min' in threshold and reading.value < threshold['min']:
                self._create_alert(reading, f"Below minimum threshold: {reading.value} {reading.unit}")
            
            if 'max' in threshold and reading.value > threshold['max']:
                self._create_alert(reading, f"Exceeds maximum threshold: {reading.value} {reading.unit}")

    def _create_alert(self, reading: SensorReading, message: str):
        """Create and store an alert"""
        alert = {
            'timestamp': reading.timestamp,
            'sensor_type': reading.sensor_type,
            'location': reading.location,
            'message': message,
            'severity': self._calculate_alert_severity(reading)
        }
        self.alerts.append(alert)

    def _calculate_alert_severity(self, reading: SensorReading) -> str:
        """Calculate alert severity based on deviation from thresholds"""
        if reading.sensor_type not in self.thresholds:
            return 'INFO'
            
        threshold = self.thresholds[reading.sensor_type]
        if 'max' in threshold:
            deviation = (reading.value - threshold['max']) / threshold['max']
            if deviation > 0.5:
                return 'CRITICAL'
            elif deviation > 0.2:
                return 'WARNING'
        return 'INFO'

    def calculate_food_safety_score(self) -> float:
        """Calculate overall food safety score based on all sensor readings"""
        scores = {}
        
        for sensor_type, readings in self.current_readings.items():
            if not readings:
                continue
                
            # Get most recent reading
            latest_reading = max(readings, key=lambda x: x.timestamp)
            
            # Calculate normalized score (0-1) based on thresholds
            if sensor_type in self.thresholds:
                threshold = self.thresholds[sensor_type]
                if 'max' in threshold:
                    score = 1 - min(1, max(0, latest_reading.value / threshold['max']))
                else:
                    score = 1  # Default score if no threshold defined
            else:
                score = 1
                
            # Apply confidence factor
            score *= latest_reading.confidence
            scores[sensor_type] = score
        
        # Calculate weighted average
        weighted_score = 0
        total_weight = 0
        
        for sensor_type, score in scores.items():
            weight = self.sensor_weights.get(sensor_type, 0)
            weighted_score += score * weight
            total_weight += weight
        
        if total_weight == 0:
            return 0
            
        return (weighted_score / total_weight) * 100

    def get_status_report(self) -> Dict:
        """Generate a comprehensive status report"""
        return {
            'timestamp': datetime.now().isoformat(),
            'safety_score': self.calculate_food_safety_score(),
            'active_alerts': [alert for alert in self.alerts if alert['severity'] != 'INFO'],
            'sensor_status': {
                sensor_type: {
                    'latest_reading': max(readings, key=lambda x: x.timestamp).value,
                    'unit': readings[0].unit,
                    'confidence': readings[0].confidence
                }
                for sensor_type, readings in self.current_readings.items()
                if readings
            }
        }

    def export_data(self, filename: str):
        """Export all sensor data and alerts to a JSON file"""
        export_data = {
            'sensor_readings': {
                sensor_type: [
                    {
                        'timestamp': reading.timestamp.isoformat(),
                        'value': reading.value,
                        'unit': reading.unit,
                        'location': reading.location,
                        'confidence': reading.confidence
                    }
                    for reading in readings
                ]
                for sensor_type, readings in self.current_readings.items()
            },
            'alerts': self.alerts,
            'safety_score': self.calculate_food_safety_score()
        }
        
        with open(filename, 'w') as f:
            json.dump(export_data, f, indent=2)

# Example usage
if __name__ == "__main__":
    # Initialize the system
    fusion_system = SensorFusionSystem()
    
    # Add some example readings
    reading = SensorReading(
        sensor_id="temp_001",
        sensor_type="temperature",
        value=4.5,
        timestamp=datetime.now(),
        unit="C",
        location="cold_storage_1",
        confidence=0.95
    )
    
    fusion_system.add_sensor_reading(reading)
    
    # Generate status report
    status = fusion_system.get_status_report()
    print(json.dumps(status, indent=2))