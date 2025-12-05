# Data Sources for Databricks Projects

Comprehensive guide to data sources for Databricks projects, including built-in datasets, external sources, and data generation scripts.

## Quick Reference

| Project | Primary Source | Size | Type | Setup Time |
|---------|---------------|------|------|------------|
| D1. Unity Catalog | Databricks samples + CSV | 1-10GB | Multi-domain | 15 min |
| D2. Delta Live Tables | Streaming + batch | 10-100GB | CDC, events | 30 min |
| D3. MLOps | ML datasets | 10-50GB | Training data | 30 min |

---

## Databricks Built-in Datasets

Databricks workspaces include sample datasets in `/databricks-datasets/`.

### Available Datasets

```python
# List all available datasets
display(dbutils.fs.ls("/databricks-datasets/"))

# Common datasets
datasets = {
    "nyctaxi": "/databricks-datasets/nyctaxi/",
    "iot": "/databricks-datasets/iot-stream/",
    "retail": "/databricks-datasets/retail-org/",
    "flights": "/databricks-datasets/flights/",
    "songs": "/databricks-datasets/songs/",
    "wikipedia": "/databricks-datasets/wikipedia-datasets/",
    "lending_club": "/databricks-datasets/lending-club-loan-stats/"
}
```

### Loading Built-in Datasets

```python
# Read NYC Taxi data
df_taxi = spark.read.format("delta").load("/databricks-datasets/nyctaxi/tables/nyctaxi_yellow")

# Read IoT data
df_iot = spark.read.format("json").load("/databricks-datasets/iot-stream/data-device/")

# Read retail data
df_retail = spark.read.format("delta").load("/databricks-datasets/retail-org/customers/")
```

---

## Project D1: Unity Catalog Governance

### Data Sources

#### 1. Multi-Domain Sample Data (Recommended)

```python
# Create sample data for different domains
from pyspark.sql import functions as F
from faker import Faker
import random

fake = Faker()

# Sales domain
sales_data = spark.createDataFrame([
    (i, fake.company(), fake.random_int(100, 10000), fake.date_this_year(), fake.country())
    for i in range(100000)
], ["transaction_id", "customer", "amount", "date", "country"])

sales_data.write.format("delta").mode("overwrite").saveAsTable("sales.transactions")

# HR domain (with PII)
hr_data = spark.createDataFrame([
    (i, fake.name(), fake.email(), fake.ssn(), fake.random_int(30000, 150000))
    for i in range(10000)
], ["employee_id", "name", "email", "ssn", "salary"])

hr_data.write.format("delta").mode("overwrite").saveAsTable("hr.employees")

# Marketing domain
marketing_data = spark.createDataFrame([
    (i, fake.email(), fake.random_element(['email', 'social', 'search']), fake.date_this_month())
    for i in range(50000)
], ["campaign_id", "customer_email", "channel", "date"])

marketing_data.write.format("delta").mode("overwrite").saveAsTable("marketing.campaigns")
```

#### 2. CSV Upload for Testing

```python
# Create CSV files for upload testing
import pandas as pd

# Customer data
customers = pd.DataFrame({
    'customer_id': range(1000),
    'name': [fake.name() for _ in range(1000)],
    'email': [fake.email() for _ in range(1000)],
    'country': [fake.country() for _ in range(1000)]
})
customers.to_csv('customers.csv', index=False)

# Upload via Databricks UI: Data → Create Table → Upload File
```

#### 3. External Data Sources

```python
# Connect to external database
jdbc_url = "jdbc:postgresql://hostname:5432/database"
properties = {
    "user": "username",
    "password": "password",
    "driver": "org.postgresql.Driver"
}

df_external = spark.read.jdbc(
    url=jdbc_url,
    table="public.orders",
    properties=properties
)

# Register as external table
df_external.write.format("delta").saveAsTable("external.orders")
```

### Data Requirements for D1
- **Multiple domains**: Sales, HR, Marketing (3+ catalogs)
- **PII data**: For masking and row-level security
- **Size**: 10K-100K rows per domain
- **Format**: Delta Lake tables

---

## Project D2: Delta Live Tables Pipeline

### Data Sources

#### 1. Streaming Data (Recommended)

```python
# Generate streaming IoT data
from pyspark.sql.types import *
from pyspark.sql.functions import *
import time
import json

# Schema for IoT events
schema = StructType([
    StructField("device_id", StringType()),
    StructField("temperature", DoubleType()),
    StructField("humidity", DoubleType()),
    StructField("timestamp", TimestampType())
])

# Write streaming data to cloud storage
def generate_iot_stream():
    while True:
        events = [
            {
                "device_id": f"device_{random.randint(1, 100)}",
                "temperature": random.uniform(15, 35),
                "humidity": random.uniform(30, 80),
                "timestamp": datetime.now().isoformat()
            }
            for _ in range(100)
        ]
        
        # Write to DBFS
        with open(f"/dbfs/tmp/iot-stream/{int(time.time())}.json", "w") as f:
            for event in events:
                f.write(json.dumps(event) + "\n")
        
        time.sleep(10)  # Generate every 10 seconds

# Run in background
import threading
thread = threading.Thread(target=generate_iot_stream)
thread.daemon = True
thread.start()
```

#### 2. CDC (Change Data Capture) Data

```python
# Simulate CDC events
cdc_events = spark.createDataFrame([
    (1, "John Doe", "john@example.com", "INSERT", "2024-01-01"),
    (1, "John Doe", "john.doe@example.com", "UPDATE", "2024-01-02"),
    (2, "Jane Smith", "jane@example.com", "INSERT", "2024-01-01"),
    (1, None, None, "DELETE", "2024-01-03")
], ["id", "name", "email", "operation", "timestamp"])

# Write to Delta table
cdc_events.write.format("delta").mode("append").save("/tmp/cdc-source")
```

#### 3. Batch Data Sources

```python
# Use NYC Taxi data (built-in)
df_batch = spark.read.format("delta").load("/databricks-datasets/nyctaxi/tables/nyctaxi_yellow")

# Or generate large batch data
from pyspark.sql.functions import *

large_batch = spark.range(0, 10000000).select(
    col("id"),
    (rand() * 1000).cast("int").alias("value"),
    date_add(current_date(), (rand() * 365).cast("int")).alias("date")
)

large_batch.write.format("delta").mode("overwrite").save("/tmp/batch-source")
```

#### 4. Log Data

```python
# Generate web server logs
log_data = spark.createDataFrame([
    (
        fake.ipv4(),
        fake.user_agent(),
        fake.uri_path(),
        random.choice([200, 404, 500]),
        random.randint(100, 10000),
        fake.date_time_this_month()
    )
    for _ in range(1000000)
], ["ip", "user_agent", "path", "status_code", "response_time_ms", "timestamp"])

log_data.write.format("delta").mode("overwrite").save("/tmp/logs-source")
```

### Data Requirements for D2
- **Streaming source**: Continuous data generation
- **Batch source**: Large historical data (1M+ rows)
- **CDC source**: Insert/Update/Delete operations
- **Format**: JSON, CSV, or Delta Lake

---

## Project D3: MLOps with MLflow

### Data Sources

#### 1. Tabular Data (Classification)

```python
# Use built-in lending club dataset
df_lending = spark.read.format("delta").load("/databricks-datasets/lending-club-loan-stats/")

# Or generate synthetic classification data
from sklearn.datasets import make_classification
import pandas as pd

X, y = make_classification(
    n_samples=100000,
    n_features=20,
    n_informative=15,
    n_redundant=5,
    random_state=42
)

df_classification = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(20)])
df_classification['target'] = y

spark_df = spark.createDataFrame(df_classification)
spark_df.write.format("delta").mode("overwrite").saveAsTable("ml.classification_data")
```

#### 2. Time Series Data (Forecasting)

```python
# Generate time series data
import pandas as pd
import numpy as np

dates = pd.date_range('2020-01-01', '2024-01-01', freq='D')
values = np.cumsum(np.random.randn(len(dates))) + 100

df_timeseries = pd.DataFrame({
    'date': dates,
    'value': values,
    'day_of_week': dates.dayofweek,
    'month': dates.month
})

spark.createDataFrame(df_timeseries).write.format("delta").mode("overwrite").saveAsTable("ml.timeseries_data")
```

#### 3. Image Data (Computer Vision)

```python
# Download CIFAR-10 or MNIST
from torchvision import datasets
import torch

# Download MNIST
mnist = datasets.MNIST(root='/dbfs/tmp/mnist', download=True)

# Convert to Delta Lake
# (Store image paths and labels, not raw images in Delta)
image_metadata = []
for idx, (img, label) in enumerate(mnist):
    img_path = f"/dbfs/tmp/mnist/images/{idx}.png"
    img.save(img_path)
    image_metadata.append((idx, img_path, int(label)))

df_images = spark.createDataFrame(image_metadata, ["id", "path", "label"])
df_images.write.format("delta").mode("overwrite").saveAsTable("ml.mnist_images")
```

#### 4. Feature Store Data

```python
# Create feature tables for Feature Store
from databricks.feature_store import FeatureStoreClient

fs = FeatureStoreClient()

# Customer features
customer_features = spark.createDataFrame([
    (i, random.randint(18, 80), random.choice(['M', 'F']), random.randint(0, 50))
    for i in range(10000)
], ["customer_id", "age", "gender", "purchase_count"])

# Create feature table
fs.create_table(
    name="ml.customer_features",
    primary_keys=["customer_id"],
    df=customer_features,
    description="Customer demographic and behavioral features"
)
```

### Data Requirements for D3
- **Training data**: 10K-1M rows
- **Features**: 10-100 features
- **Target**: Classification or regression
- **Format**: Delta Lake tables
- **Feature Store**: Separate feature tables

---

## External Data Sources

### 1. Kaggle Datasets

```python
# Install Kaggle CLI
%pip install kaggle

# Set up credentials (upload kaggle.json to DBFS)
import os
os.environ['KAGGLE_CONFIG_DIR'] = '/dbfs/tmp/kaggle'

# Download dataset
!kaggle datasets download -d mlg-ulb/creditcardfraud -p /dbfs/tmp/kaggle-data

# Load into Spark
df = spark.read.csv("/dbfs/tmp/kaggle-data/creditcard.csv", header=True, inferSchema=True)
df.write.format("delta").mode("overwrite").saveAsTable("external.credit_fraud")
```

### 2. AWS S3 Data

```python
# Configure S3 access
spark.conf.set("fs.s3a.access.key", "YOUR_ACCESS_KEY")
spark.conf.set("fs.s3a.secret.key", "YOUR_SECRET_KEY")

# Read from S3
df_s3 = spark.read.format("parquet").load("s3a://bucket-name/path/to/data")
df_s3.write.format("delta").mode("overwrite").saveAsTable("external.s3_data")
```

### 3. Azure Blob Storage

```python
# Configure Azure access
spark.conf.set("fs.azure.account.key.ACCOUNT_NAME.blob.core.windows.net", "YOUR_KEY")

# Read from Azure
df_azure = spark.read.format("parquet").load("wasbs://container@account.blob.core.windows.net/path")
df_azure.write.format("delta").mode("overwrite").saveAsTable("external.azure_data")
```

---

## Data Generation Scripts

### Complete Setup Script

```python
# Run this notebook to set up all data sources

from faker import Faker
from pyspark.sql import functions as F
import random

fake = Faker()

print("Setting up data sources for Databricks projects...")

# D1: Unity Catalog data
print("Creating Unity Catalog sample data...")
# (Use code from D1 section above)

# D2: Delta Live Tables data
print("Creating streaming and batch data...")
# (Use code from D2 section above)

# D3: MLOps data
print("Creating ML training data...")
# (Use code from D3 section above)

print("Data setup complete!")
print("Tables created:")
display(spark.sql("SHOW TABLES"))
```

---

## Storage Locations

### DBFS (Databricks File System)
```python
# Temporary data
/dbfs/tmp/

# User data
/dbfs/user/your-email/

# Shared data
/dbfs/FileStore/

# Databricks datasets
/databricks-datasets/
```

### Unity Catalog Managed Storage
```python
# Managed tables (Unity Catalog handles storage)
CREATE TABLE catalog.schema.table ...

# External tables (you manage storage)
CREATE EXTERNAL TABLE catalog.schema.table
LOCATION 's3://bucket/path'
```

---

## Data Size Recommendations

### Development
- **D1**: 10K-100K rows per domain
- **D2**: 1M rows batch, 1K events/sec streaming
- **D3**: 100K rows training data

### Production Simulation
- **D1**: 1M-10M rows per domain
- **D2**: 100M rows batch, 10K events/sec streaming
- **D3**: 1M-10M rows training data

---

## Cost Optimization

### Storage Costs
- **DBFS**: Included in Databricks pricing
- **External (S3/Azure)**: ~$0.023/GB/month
- **Delta Lake**: Same as underlying storage

### Data Transfer Costs
- **Within region**: Free
- **Cross-region**: $0.01-0.02/GB
- **Internet egress**: $0.09/GB

### Tips
1. Use **Delta Lake** for efficient storage
2. **Partition** large tables
3. **Compress** data (Snappy, Zstd)
4. **Clean up** temporary data
5. Use **OPTIMIZE** and **VACUUM** commands

---

## Troubleshooting

### Issue: Dataset too large
**Solution**: Sample the data
```python
df_sample = df.sample(fraction=0.1)
```

### Issue: Slow data generation
**Solution**: Use parallel processing
```python
df = spark.range(0, 1000000, numPartitions=100)
```

### Issue: Out of memory
**Solution**: Increase cluster size or use streaming
```python
df.write.format("delta").mode("append").save("/path")  # Streaming write
```

---

## Quick Start Commands

```python
# Install required libraries
%pip install faker

# Generate all sample data
%run ./setup_data_sources

# Verify data
display(spark.sql("SHOW DATABASES"))
display(spark.sql("SHOW TABLES IN sales"))

# Check data size
spark.sql("SELECT COUNT(*) FROM sales.transactions").show()
```

---

## Additional Resources

- **Databricks Datasets**: https://docs.databricks.com/data/databricks-datasets.html
- **Delta Lake**: https://docs.delta.io/
- **Unity Catalog**: https://docs.databricks.com/data-governance/unity-catalog/
- **Kaggle**: https://www.kaggle.com/datasets
- **Faker Documentation**: https://faker.readthedocs.io/
