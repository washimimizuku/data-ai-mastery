# Data Sources for Snowflake Projects

Comprehensive guide to data sources for Snowflake projects, including built-in datasets, external sources, and data generation scripts.

## Quick Reference

| Project | Primary Source | Size | Type | Setup Time |
|---------|---------------|------|------|------------|
| S1. Data Sharing | Snowflake samples + CSV | 1-10GB | Multi-domain | 15 min |
| S2. CDC Streaming | Kafka + Python | Streaming | Real-time CDC | 30 min |
| S3. Snowpark ML | ML datasets | 10-50GB | Training data | 30 min |

---

## Snowflake Built-in Datasets

Snowflake accounts include sample datasets in `SNOWFLAKE_SAMPLE_DATA` database.

### Available Datasets

```sql
-- Use sample data database
USE DATABASE SNOWFLAKE_SAMPLE_DATA;

-- List available schemas
SHOW SCHEMAS;

-- Common datasets
-- TPCH_SF1: TPC-H benchmark (1GB scale)
-- TPCH_SF10: TPC-H benchmark (10GB scale)
-- TPCDS_SF10TCL: TPC-DS benchmark
-- WEATHER: Weather observations
```

### Loading Built-in Datasets

```sql
-- Query TPC-H data
SELECT * FROM SNOWFLAKE_SAMPLE_DATA.TPCH_SF1.CUSTOMER LIMIT 100;

-- Query weather data
SELECT * FROM SNOWFLAKE_SAMPLE_DATA.WEATHER.DAILY_14_TOTAL LIMIT 100;

-- Create your own copy
CREATE TABLE my_db.my_schema.customers AS
SELECT * FROM SNOWFLAKE_SAMPLE_DATA.TPCH_SF1.CUSTOMER;
```

---

## Project S1: Enterprise Data Sharing

### Data Sources

#### 1. Multi-Domain Sample Data (Recommended)

```sql
-- Create databases for different domains
CREATE DATABASE sales_db;
CREATE DATABASE hr_db;
CREATE DATABASE marketing_db;

-- Sales domain
CREATE OR REPLACE TABLE sales_db.public.transactions AS
SELECT
    SEQ8() as transaction_id,
    UNIFORM(1, 10000, RANDOM()) as customer_id,
    UNIFORM(10, 10000, RANDOM()) as amount,
    DATEADD(day, -UNIFORM(0, 365, RANDOM()), CURRENT_DATE()) as transaction_date,
    ARRAY_CONSTRUCT('US', 'UK', 'DE', 'FR', 'JP')[UNIFORM(0, 4, RANDOM())] as country
FROM TABLE(GENERATOR(ROWCOUNT => 100000));

-- HR domain (with PII for masking)
CREATE OR REPLACE TABLE hr_db.public.employees AS
SELECT
    SEQ8() as employee_id,
    CONCAT('Employee_', SEQ8()) as name,
    CONCAT('emp', SEQ8(), '@company.com') as email,
    CONCAT(UNIFORM(100, 999, RANDOM()), '-', UNIFORM(10, 99, RANDOM()), '-', UNIFORM(1000, 9999, RANDOM())) as ssn,
    UNIFORM(30000, 150000, RANDOM()) as salary,
    ARRAY_CONSTRUCT('Engineering', 'Sales', 'Marketing', 'HR', 'Finance')[UNIFORM(0, 4, RANDOM())] as department
FROM TABLE(GENERATOR(ROWCOUNT => 10000));

-- Marketing domain
CREATE OR REPLACE TABLE marketing_db.public.campaigns AS
SELECT
    SEQ8() as campaign_id,
    CONCAT('campaign_', SEQ8(), '@email.com') as customer_email,
    ARRAY_CONSTRUCT('email', 'social', 'search', 'display')[UNIFORM(0, 3, RANDOM())] as channel,
    DATEADD(day, -UNIFORM(0, 90, RANDOM()), CURRENT_DATE()) as campaign_date,
    UNIFORM(0, 1000, RANDOM()) as clicks,
    UNIFORM(0, 100, RANDOM()) as conversions
FROM TABLE(GENERATOR(ROWCOUNT => 50000));
```

#### 2. CSV Upload for Testing

```python
# Generate CSV files locally
import pandas as pd
from faker import Faker

fake = Faker()

# Customer data
customers = pd.DataFrame({
    'customer_id': range(1, 1001),
    'name': [fake.name() for _ in range(1000)],
    'email': [fake.email() for _ in range(1000)],
    'country': [fake.country_code() for _ in range(1000)],
    'signup_date': [fake.date_this_year() for _ in range(1000)]
})
customers.to_csv('customers.csv', index=False)

# Upload via Snowflake UI or SnowSQL
# PUT file://customers.csv @~/staged;
# COPY INTO customers FROM @~/staged/customers.csv FILE_FORMAT = (TYPE = CSV SKIP_HEADER = 1);
```

#### 3. External Stage (S3/Azure/GCS)

```sql
-- Create external stage
CREATE OR REPLACE STAGE my_s3_stage
  URL = 's3://my-bucket/data/'
  CREDENTIALS = (AWS_KEY_ID = 'xxx' AWS_SECRET_KEY = 'yyy');

-- List files
LIST @my_s3_stage;

-- Load data
COPY INTO my_table
FROM @my_s3_stage/data.csv
FILE_FORMAT = (TYPE = CSV SKIP_HEADER = 1);
```

### Data Requirements for S1
- **Multiple domains**: Sales, HR, Marketing (3+ databases)
- **PII data**: For masking and row-level security
- **Size**: 10K-100K rows per domain
- **Format**: Snowflake tables

---

## Project S2: Real-Time CDC with Snowpipe Streaming

### Data Sources

#### 1. Snowpipe Streaming with Python (Recommended)

```python
from snowflake.snowpark import Session
from snowflake.ingest import SimpleIngestManager
from snowflake.ingest.utils.uris import DEFAULT_SCHEME
import json
import time
from datetime import datetime

# Snowflake connection
connection_parameters = {
    "account": "your_account",
    "user": "your_user",
    "password": "your_password",
    "role": "your_role",
    "warehouse": "your_warehouse",
    "database": "your_database",
    "schema": "your_schema"
}

session = Session.builder.configs(connection_parameters).create()

# Generate CDC events
def generate_cdc_event(operation='INSERT'):
    return {
        'id': random.randint(1, 10000),
        'name': fake.name(),
        'email': fake.email(),
        'operation': operation,
        'timestamp': datetime.now().isoformat()
    }

# Stream to Snowflake
from snowflake.snowpark.functions import col, parse_json

# Create stream table
session.sql("""
    CREATE OR REPLACE TABLE cdc_raw (
        data VARIANT,
        ingestion_time TIMESTAMP_LTZ DEFAULT CURRENT_TIMESTAMP()
    )
""").collect()

# Ingest data continuously
while True:
    events = [generate_cdc_event() for _ in range(100)]
    
    # Insert into Snowflake
    df = session.create_dataframe(
        [(json.dumps(event),) for event in events],
        schema=["data"]
    )
    df.write.mode("append").save_as_table("cdc_raw")
    
    time.sleep(1)  # 100 events/sec
```

#### 2. Kafka to Snowflake

```python
from kafka import KafkaConsumer
from snowflake.connector import connect
import json

# Kafka consumer
consumer = KafkaConsumer(
    'cdc-events',
    bootstrap_servers=['localhost:9092'],
    value_deserializer=lambda m: json.loads(m.decode('utf-8'))
)

# Snowflake connection
conn = connect(
    user='your_user',
    password='your_password',
    account='your_account',
    warehouse='your_warehouse',
    database='your_database',
    schema='your_schema'
)

cursor = conn.cursor()

# Stream from Kafka to Snowflake
for message in consumer:
    event = message.value
    
    cursor.execute("""
        INSERT INTO cdc_raw (data)
        SELECT PARSE_JSON(%s)
    """, (json.dumps(event),))
    
    conn.commit()
```

#### 3. Simulated CDC Data

```sql
-- Create source table
CREATE OR REPLACE TABLE source_table (
    id INT,
    name STRING,
    email STRING,
    updated_at TIMESTAMP_LTZ DEFAULT CURRENT_TIMESTAMP()
);

-- Insert initial data
INSERT INTO source_table (id, name, email)
SELECT
    SEQ8() as id,
    CONCAT('User_', SEQ8()) as name,
    CONCAT('user', SEQ8(), '@example.com') as email
FROM TABLE(GENERATOR(ROWCOUNT => 10000));

-- Simulate updates
UPDATE source_table
SET name = CONCAT(name, '_updated'),
    updated_at = CURRENT_TIMESTAMP()
WHERE id <= 1000;

-- Simulate deletes
DELETE FROM source_table WHERE id <= 100;

-- Create stream to track changes
CREATE OR REPLACE STREAM source_stream ON TABLE source_table;

-- View changes
SELECT * FROM source_stream;
```

#### 4. Log Data Streaming

```python
# Generate web server logs
import random
from faker import Faker

fake = Faker()

def generate_log_entry():
    return {
        'timestamp': datetime.now().isoformat(),
        'ip': fake.ipv4(),
        'method': random.choice(['GET', 'POST', 'PUT', 'DELETE']),
        'path': fake.uri_path(),
        'status_code': random.choice([200, 201, 400, 404, 500]),
        'response_time_ms': random.randint(10, 5000),
        'user_agent': fake.user_agent()
    }

# Stream to Snowflake
while True:
    logs = [generate_log_entry() for _ in range(1000)]
    # Insert into Snowflake (use code from above)
    time.sleep(1)
```

### Data Requirements for S2
- **Streaming source**: Continuous data generation (100-1000 events/sec)
- **CDC operations**: INSERT, UPDATE, DELETE
- **Format**: JSON or VARIANT in Snowflake
- **Volume**: Millions of events for testing

---

## Project S3: Snowpark ML Platform

### Data Sources

#### 1. Tabular Data (Classification) - Recommended

```sql
-- Generate classification dataset
CREATE OR REPLACE TABLE ml_classification_data AS
SELECT
    SEQ8() as id,
    UNIFORM(18, 80, RANDOM()) as age,
    ARRAY_CONSTRUCT('M', 'F')[UNIFORM(0, 1, RANDOM())] as gender,
    UNIFORM(20000, 150000, RANDOM()) as income,
    UNIFORM(0, 50, RANDOM()) as purchase_count,
    UNIFORM(0, 10000, RANDOM()) as total_spent,
    UNIFORM(0, 100, RANDOM()) as days_since_last_purchase,
    CASE WHEN UNIFORM(0, 1, RANDOM()) < 0.3 THEN 1 ELSE 0 END as churned
FROM TABLE(GENERATOR(ROWCOUNT => 100000));

-- Split into train/test
CREATE OR REPLACE TABLE ml_train AS
SELECT * FROM ml_classification_data WHERE id % 10 < 8;

CREATE OR REPLACE TABLE ml_test AS
SELECT * FROM ml_classification_data WHERE id % 10 >= 8;
```

#### 2. Time Series Data (Forecasting)

```sql
-- Generate time series data
CREATE OR REPLACE TABLE ml_timeseries_data AS
SELECT
    DATEADD(day, SEQ8(), '2020-01-01'::DATE) as date,
    UNIFORM(100, 1000, RANDOM()) + (SEQ8() * 0.1) as value,
    DAYOFWEEK(DATEADD(day, SEQ8(), '2020-01-01'::DATE)) as day_of_week,
    MONTH(DATEADD(day, SEQ8(), '2020-01-01'::DATE)) as month,
    CASE WHEN DAYOFWEEK(DATEADD(day, SEQ8(), '2020-01-01'::DATE)) IN (0, 6) THEN 1 ELSE 0 END as is_weekend
FROM TABLE(GENERATOR(ROWCOUNT => 1460));  -- 4 years of daily data
```

#### 3. Regression Data

```sql
-- Generate regression dataset (house prices)
CREATE OR REPLACE TABLE ml_regression_data AS
SELECT
    SEQ8() as id,
    UNIFORM(500, 5000, RANDOM()) as square_feet,
    UNIFORM(1, 6, RANDOM()) as bedrooms,
    UNIFORM(1, 4, RANDOM()) as bathrooms,
    UNIFORM(1, 100, RANDOM()) as age_years,
    ARRAY_CONSTRUCT('urban', 'suburban', 'rural')[UNIFORM(0, 2, RANDOM())] as location_type,
    -- Price formula with some noise
    (UNIFORM(500, 5000, RANDOM()) * 200) + 
    (UNIFORM(1, 6, RANDOM()) * 50000) + 
    (UNIFORM(1, 4, RANDOM()) * 30000) - 
    (UNIFORM(1, 100, RANDOM()) * 1000) + 
    UNIFORM(-50000, 50000, RANDOM()) as price
FROM TABLE(GENERATOR(ROWCOUNT => 50000));
```

#### 4. Feature Engineering Data

```sql
-- Create feature tables for Snowpark ML
CREATE OR REPLACE TABLE customer_features AS
SELECT
    customer_id,
    COUNT(*) as transaction_count,
    SUM(amount) as total_spent,
    AVG(amount) as avg_transaction,
    MAX(transaction_date) as last_transaction_date,
    DATEDIFF(day, MAX(transaction_date), CURRENT_DATE()) as days_since_last_purchase
FROM sales_db.public.transactions
GROUP BY customer_id;

-- Demographic features
CREATE OR REPLACE TABLE demographic_features AS
SELECT
    customer_id,
    age,
    gender,
    country,
    DATEDIFF(year, signup_date, CURRENT_DATE()) as customer_tenure_years
FROM customers;
```

#### 5. External ML Datasets

```python
# Load Kaggle dataset into Snowflake
import pandas as pd
from snowflake.snowpark import Session

# Read CSV
df = pd.read_csv('creditcard.csv')

# Create Snowpark session
session = Session.builder.configs(connection_parameters).create()

# Upload to Snowflake
snowpark_df = session.create_dataframe(df)
snowpark_df.write.mode("overwrite").save_as_table("ml_credit_fraud")
```

### Data Requirements for S3
- **Training data**: 10K-1M rows
- **Features**: 10-100 features
- **Target**: Classification or regression
- **Format**: Snowflake tables
- **Split**: Train/test/validation

---

## External Data Sources

### 1. Kaggle Datasets

```bash
# Download locally
kaggle datasets download -d mlg-ulb/creditcardfraud

# Upload to Snowflake stage
snowsql -q "PUT file://creditcard.csv @~/staged AUTO_COMPRESS=TRUE"

# Load into table
snowsql -q "
COPY INTO credit_fraud
FROM @~/staged/creditcard.csv.gz
FILE_FORMAT = (TYPE = CSV SKIP_HEADER = 1 COMPRESSION = GZIP)
"
```

### 2. AWS S3 Integration

```sql
-- Create storage integration
CREATE OR REPLACE STORAGE INTEGRATION s3_integration
  TYPE = EXTERNAL_STAGE
  STORAGE_PROVIDER = S3
  ENABLED = TRUE
  STORAGE_AWS_ROLE_ARN = 'arn:aws:iam::123456789:role/snowflake-role'
  STORAGE_ALLOWED_LOCATIONS = ('s3://my-bucket/data/');

-- Create external stage
CREATE OR REPLACE STAGE my_s3_stage
  STORAGE_INTEGRATION = s3_integration
  URL = 's3://my-bucket/data/';

-- Load data
COPY INTO my_table FROM @my_s3_stage
FILE_FORMAT = (TYPE = PARQUET);
```

### 3. Azure Blob Storage

```sql
-- Create storage integration
CREATE OR REPLACE STORAGE INTEGRATION azure_integration
  TYPE = EXTERNAL_STAGE
  STORAGE_PROVIDER = AZURE
  ENABLED = TRUE
  AZURE_TENANT_ID = 'your-tenant-id'
  STORAGE_ALLOWED_LOCATIONS = ('azure://account.blob.core.windows.net/container/');

-- Create external stage
CREATE OR REPLACE STAGE my_azure_stage
  STORAGE_INTEGRATION = azure_integration
  URL = 'azure://account.blob.core.windows.net/container/';
```

---

## Data Generation Scripts

### Complete Setup Script

```sql
-- Run this script to set up all data sources

-- S1: Data Sharing
CREATE DATABASE IF NOT EXISTS sales_db;
CREATE DATABASE IF NOT EXISTS hr_db;
CREATE DATABASE IF NOT EXISTS marketing_db;

-- (Use code from S1 section above)

-- S2: CDC Streaming
CREATE DATABASE IF NOT EXISTS streaming_db;
CREATE SCHEMA IF NOT EXISTS streaming_db.cdc;

CREATE OR REPLACE TABLE streaming_db.cdc.cdc_raw (
    data VARIANT,
    ingestion_time TIMESTAMP_LTZ DEFAULT CURRENT_TIMESTAMP()
);

CREATE OR REPLACE STREAM streaming_db.cdc.cdc_stream 
ON TABLE streaming_db.cdc.cdc_raw;

-- S3: ML Data
CREATE DATABASE IF NOT EXISTS ml_db;
CREATE SCHEMA IF NOT EXISTS ml_db.datasets;

-- (Use code from S3 section above)

SELECT 'Data setup complete!' as status;
```

---

## Snowflake-Specific Features

### 1. GENERATOR Function
```sql
-- Generate large datasets efficiently
SELECT
    SEQ8() as id,
    UNIFORM(1, 100, RANDOM()) as value
FROM TABLE(GENERATOR(ROWCOUNT => 1000000));
```

### 2. VARIANT Data Type
```sql
-- Store semi-structured data
CREATE TABLE json_data (
    id INT,
    data VARIANT
);

INSERT INTO json_data
SELECT 1, PARSE_JSON('{"name": "John", "age": 30}');

-- Query JSON
SELECT data:name::STRING as name FROM json_data;
```

### 3. Time Travel
```sql
-- Query historical data
SELECT * FROM my_table AT(OFFSET => -3600);  -- 1 hour ago
SELECT * FROM my_table BEFORE(STATEMENT => 'query_id');
```

---

## Storage Locations

### Internal Stages
```sql
-- User stage
PUT file://data.csv @~;

-- Table stage
PUT file://data.csv @%my_table;

-- Named stage
CREATE STAGE my_stage;
PUT file://data.csv @my_stage;
```

### External Stages
```sql
-- S3
CREATE STAGE s3_stage URL='s3://bucket/path/';

-- Azure
CREATE STAGE azure_stage URL='azure://account.blob.core.windows.net/container/';

-- GCS
CREATE STAGE gcs_stage URL='gcs://bucket/path/';
```

---

## Data Size Recommendations

### Development
- **S1**: 10K-100K rows per domain
- **S2**: 1K events/sec, 1M total events
- **S3**: 100K rows training data

### Production Simulation
- **S1**: 1M-10M rows per domain
- **S2**: 10K events/sec, 100M+ total events
- **S3**: 1M-10M rows training data

---

## Cost Optimization

### Storage Costs
- **Snowflake storage**: $40/TB/month (compressed)
- **External (S3/Azure)**: ~$23/TB/month

### Compute Costs
- **X-Small**: 1 credit/hour
- **Small**: 2 credits/hour
- **Medium**: 4 credits/hour

### Tips
1. Use **GENERATOR** for synthetic data (fast, free)
2. **Compress** data (automatic in Snowflake)
3. **Cluster keys** for large tables
4. **Suspend warehouses** when not in use
5. Use **result caching** (free)
6. **Right-size warehouses**

---

## Troubleshooting

### Issue: Slow data loading
**Solution**: Use larger warehouse or COPY command
```sql
ALTER WAREHOUSE my_wh SET WAREHOUSE_SIZE = 'LARGE';
```

### Issue: Out of credits
**Solution**: Set resource monitor
```sql
CREATE RESOURCE MONITOR my_monitor
  WITH CREDIT_QUOTA = 100
  TRIGGERS ON 90 PERCENT DO SUSPEND;
```

### Issue: Query timeout
**Solution**: Increase statement timeout
```sql
ALTER SESSION SET STATEMENT_TIMEOUT_IN_SECONDS = 3600;
```

---

## Quick Start Commands

```sql
-- Set up environment
USE ROLE ACCOUNTADMIN;
CREATE WAREHOUSE IF NOT EXISTS learning_wh WITH WAREHOUSE_SIZE = 'X-SMALL';
CREATE DATABASE IF NOT EXISTS learning_db;
USE WAREHOUSE learning_wh;
USE DATABASE learning_db;

-- Generate sample data
CREATE OR REPLACE TABLE sample_data AS
SELECT
    SEQ8() as id,
    UNIFORM(1, 100, RANDOM()) as value,
    CURRENT_TIMESTAMP() as created_at
FROM TABLE(GENERATOR(ROWCOUNT => 100000));

-- Verify
SELECT COUNT(*) FROM sample_data;
```

---

## Additional Resources

- **Snowflake Documentation**: https://docs.snowflake.com/
- **Snowflake Sample Data**: https://docs.snowflake.com/en/user-guide/sample-data
- **Snowpark Guide**: https://docs.snowflake.com/en/developer-guide/snowpark/
- **Kaggle**: https://www.kaggle.com/datasets
- **Faker Documentation**: https://faker.readthedocs.io/
