# Project E: Environmental Sensor Data Streaming

**Time**: 2-3 days | **Difficulty**: Advanced | **Focus**: IoT data processing for environmental monitoring

Build a real-time environmental monitoring system processing sensor data streams. This project demonstrates practical applications of streaming data processing, IoT integration, and real-time analytics for environmental science.

## Learning Objectives

- Design and implement IoT sensor data pipelines
- Process real-time environmental data streams
- Build anomaly detection for environmental hazards
- Create real-time monitoring dashboards
- Implement alerting systems for threshold breaches

## Tech Stack

- **Python**: Core programming language
- **Apache Kafka**: Message streaming platform
- **InfluxDB**: Time series database
- **Grafana**: Real-time visualization
- **MQTT**: IoT messaging protocol
- **pandas/NumPy**: Data processing
- **scikit-learn**: Anomaly detection
- **Docker**: Containerization

## Project Structure

```
project-e-environmental-streaming/
├── README.md
├── requirements.txt
├── docker-compose.yml          # Full stack deployment
├── src/
│   ├── __init__.py
│   ├── sensor_simulator.py     # IoT sensor data simulation
│   ├── kafka_producer.py       # Stream data to Kafka
│   ├── kafka_consumer.py       # Process streaming data
│   ├── anomaly_detector.py     # Real-time anomaly detection
│   ├── alert_system.py         # Threshold-based alerts
│   └── dashboard.py            # Streamlit monitoring dashboard
├── notebooks/
│   ├── 01_sensor_data_analysis.ipynb
│   ├── 02_streaming_setup.ipynb
│   ├── 03_anomaly_detection.ipynb
│   └── 04_dashboard_development.ipynb
├── data/
│   ├── sensors/               # Sensor configuration and metadata
│   ├── historical/            # Historical sensor data
│   └── alerts/                # Alert logs and notifications
├── config/
│   ├── kafka/                 # Kafka configuration
│   ├── influxdb/              # InfluxDB setup
│   └── grafana/               # Grafana dashboards
└── tests/
    └── test_streaming.py
```

## Implementation Steps

### Day 1: Infrastructure Setup and Data Simulation
1. **Environment Setup**
   - Deploy Kafka, InfluxDB, and Grafana with Docker
   - Configure MQTT broker for IoT communication
   - Setup Python streaming libraries

2. **Sensor Simulation**
   - Create realistic environmental sensor simulators
   - Implement air quality, water quality, weather sensors
   - Add noise, drift, and failure scenarios

3. **Data Pipeline**
   - Build Kafka producers for sensor data ingestion
   - Implement data validation and preprocessing
   - Setup InfluxDB for time series storage

### Day 2: Stream Processing and Anomaly Detection
1. **Stream Processing**
   - Develop Kafka consumers for real-time processing
   - Implement sliding window aggregations
   - Create data quality checks and filtering

2. **Anomaly Detection**
   - Statistical anomaly detection (Z-score, IQR)
   - Machine learning models for pattern recognition
   - Threshold-based alerting system

3. **Alert System**
   - Email and SMS notifications
   - Webhook integrations
   - Alert escalation and acknowledgment

### Day 3: Visualization and Monitoring
1. **Real-time Dashboard**
   - Grafana dashboards for sensor monitoring
   - Streamlit application for detailed analysis
   - Interactive maps for sensor locations

2. **Historical Analysis**
   - Trend analysis and pattern recognition
   - Data quality reporting
   - Performance metrics and SLA monitoring

3. **Deployment**
   - Docker containerization
   - Kubernetes deployment (optional)
   - Monitoring and logging setup

## Key Features

### Sensor Data Simulation
- **Air Quality**: PM2.5, PM10, NO2, O3, CO sensors
- **Water Quality**: pH, dissolved oxygen, turbidity, temperature
- **Weather**: Temperature, humidity, pressure, wind speed
- **Realistic Patterns**: Diurnal cycles, seasonal trends, weather events

### Streaming Architecture
- **Kafka Topics**: Separate topics for different sensor types
- **Schema Registry**: Avro schemas for data validation
- **Stream Processing**: Real-time aggregations and transformations
- **Fault Tolerance**: Replication and error handling

### Anomaly Detection
- **Statistical Methods**: Z-score, modified Z-score, IQR
- **Machine Learning**: Isolation Forest, One-Class SVM
- **Threshold-based**: User-defined limits for each parameter
- **Contextual**: Time-of-day and seasonal adjustments

### Real-time Monitoring
- **Live Dashboards**: Current sensor readings and status
- **Historical Trends**: Time series visualization
- **Geographic Mapping**: Sensor locations and readings
- **Alert Management**: Active alerts and acknowledgments

### Data Storage
- **Time Series**: InfluxDB for high-frequency sensor data
- **Metadata**: PostgreSQL for sensor configuration
- **Alerts**: Structured logging for alert history
- **Aggregations**: Pre-computed statistics for fast queries

## Sample Sensor Network

The project simulates a network of environmental sensors:
- **Urban Air Quality**: 10 sensors across city locations
- **River Water Quality**: 5 sensors along waterway
- **Weather Stations**: 3 comprehensive weather monitoring sites
- **Industrial Monitoring**: 2 sensors near emission sources

## Expected Outcomes

1. **Real-time streaming pipeline** processing 1000+ messages/second
2. **Anomaly detection system** with <5% false positive rate
3. **Interactive monitoring dashboard** with <1 second latency
4. **Automated alert system** for environmental threshold breaches

## Real-World Applications

- **Smart Cities**: Urban environmental monitoring
- **Industrial Compliance**: Emission and discharge monitoring
- **Research**: Long-term environmental data collection
- **Emergency Response**: Early warning for environmental hazards

## Performance Targets

- **Throughput**: >1000 messages/second
- **Latency**: <1 second end-to-end processing
- **Availability**: 99.9% uptime
- **Data Retention**: 1 year of high-resolution data

## Technical Challenges

1. **Data Volume**: High-frequency sensor data processing
2. **Network Reliability**: Handling sensor connectivity issues
3. **Data Quality**: Sensor calibration and drift detection
4. **Scalability**: Adding new sensors and locations

## Extensions

- **Edge Computing**: Local processing on IoT gateways
- **Machine Learning**: Predictive maintenance for sensors
- **Mobile App**: Field technician interface
- **API Integration**: Third-party data sources and services

## Getting Started

```bash
# Setup environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt

# Start infrastructure
docker-compose up -d

# Run sensor simulation
python src/sensor_simulator.py --sensors air_quality,water_quality,weather

# Start stream processing
python src/kafka_consumer.py

# Access dashboards
# Grafana: http://localhost:3000 (admin/admin)
# Streamlit: streamlit run src/dashboard.py
```

## Monitoring Stack

### Kafka
- **Topics**: sensor-data, alerts, system-metrics
- **Partitions**: Distributed processing across consumers
- **Retention**: 7 days for raw data, 30 days for alerts

### InfluxDB
- **Measurements**: One per sensor type
- **Tags**: sensor_id, location, sensor_type
- **Fields**: Sensor readings and quality metrics
- **Retention**: 1 year high-resolution, 5 years aggregated

### Grafana
- **Real-time Panels**: Current readings and trends
- **Alert Rules**: Threshold-based notifications
- **Variables**: Dynamic sensor and location selection
- **Annotations**: Mark significant events

## Resources

- [Apache Kafka Documentation](https://kafka.apache.org/documentation/)
- [InfluxDB Time Series Guide](https://docs.influxdata.com/influxdb/)
- [Grafana Dashboard Guide](https://grafana.com/docs/grafana/latest/)
- [MQTT Protocol](https://mqtt.org/)
- [Environmental Sensor Networks](https://www.epa.gov/air-sensor-toolbox)
