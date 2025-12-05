# Project S2: Real-Time CDC with Snowpipe Streaming

## Overview

Build a real-time change data capture (CDC) pipeline using Snowflake's Snowpipe Streaming API, demonstrating low-latency data ingestion, stream processing with Streams and Tasks, and SCD Type 2 patterns.

**What You'll Build**: A complete CDC pipeline that captures changes from source systems, streams them to Snowflake in real-time (<1 second latency), processes them with Streams and Tasks, and maintains historical data with SCD Type 2.

**What You'll Learn**: Snowpipe Streaming API, Snowflake Streams, Tasks orchestration, Dynamic Tables, CDC patterns, and SCD Type 2 implementation.

## Time Estimate

**3-4 days (24-32 hours)**

### Day 1: Streaming Ingestion (8 hours)
- Hours 1-2: Snowpipe Streaming setup
- Hours 3-4: Python client implementation
- Hours 5-6: Event ingestion testing
- Hours 7-8: Error handling

### Day 2: Stream Processing (8 hours)
- Hours 1-3: Streams creation
- Hours 4-5: Tasks for CDC parsing
- Hours 6-7: Task orchestration
- Hour 8: Testing

### Day 3: SCD Type 2 (8 hours)
- Hours 1-3: SCD Type 2 logic
- Hours 4-5: Merge operations
- Hours 6-7: Historical tracking
- Hour 8: Validation

### Day 4: Optimization (6-8 hours)
- Hours 1-2: Dynamic Tables
- Hours 3-4: Monitoring setup
- Hours 5-6: Performance tuning
- Hours 7-8: Documentation

## Prerequisites

### Required Knowledge
- [30 Days of Snowflake](https://github.com/washimimizuku/30-days-snowflake-data-ai) - Days 1-30
  - Days 6-10: Streams and Tasks
  - Days 16-20: Snowpipe and data loading
  - Days 26-30: Advanced patterns
- [60 Days Advanced](https://github.com/washimimizuku/60-days-advanced-data-ai) - Days 1-10
  - CDC patterns and real-time processing

### Technical Requirements
- Snowflake account with Snowpipe Streaming enabled
- Python 3.9+ for Snowpipe Streaming SDK
- Understanding of CDC concepts
- SQL knowledge (intermediate to advanced)

### Snowflake Setup
- Enterprise Edition or higher
- ACCOUNTADMIN role access
- Virtual warehouse configured
- Key pair authentication set up

## Getting Started

### Step 1: Review Documentation
1. `prd.md` - Product requirements and goals
2. `tech-spec.md` - Technical architecture and code examples
3. `implementation-plan.md` - Day-by-day implementation guide

### Step 2: Set Up Key Pair Authentication

**Generate RSA Key Pair:**
```bash
# Generate private key
openssl genrsa 2048 | openssl pkcs8 -topk8 -inform PEM -out rsa_key.p8 -nocrypt

# Generate public key
openssl rsa -in rsa_key.p8 -pubout -out rsa_key.pub

# Get public key value (remove headers)
cat rsa_key.pub | grep -v "BEGIN PUBLIC" | grep -v "END PUBLIC" | tr -d '\n'
```

**Assign Public Key to User (Snowflake UI):**
1. Go to **Admin** → **Users & Roles**
2. Click on your user
3. Scroll to **RSA Public Key**
4. Paste public key value
5. Click **Save**

**Or via SQL:**
```sql
ALTER USER <your_username> SET RSA_PUBLIC_KEY='<public_key_value>';
```

### Step 3: Install Python Dependencies

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install packages
pip install snowflake-connector-python
pip install snowflake-ingest
pip install cryptography

# Verify installation
python -c "from snowflake.ingest.streaming import SnowflakeStreamingIngestClient; print('✓ SDK installed')"
```

### Step 4: Create Snowflake Objects

```sql
-- Switch to ACCOUNTADMIN
USE ROLE ACCOUNTADMIN;

-- Create database and schema
CREATE DATABASE CDC_PIPELINE;
CREATE SCHEMA CDC_PIPELINE.RAW;
CREATE SCHEMA CDC_PIPELINE.STAGING;
CREATE SCHEMA CDC_PIPELINE.PROD;

-- Create warehouse
CREATE WAREHOUSE CDC_WH
  WAREHOUSE_SIZE = 'XSMALL'
  AUTO_SUSPEND = 60
  AUTO_RESUME = TRUE;

USE WAREHOUSE CDC_WH;

-- Create landing table for raw CDC events
CREATE TABLE CDC_PIPELINE.RAW.CHANGES (
    operation VARCHAR(10),
    table_name VARCHAR(255),
    before VARIANT,
    after VARIANT,
    ts_ms BIGINT,
    ingested_at TIMESTAMP_LTZ DEFAULT CURRENT_TIMESTAMP()
);

SELECT '✓ Snowflake objects created' AS status;
```

## Core Implementation

### 1. Snowpipe Streaming Client

**Create Python Client:**
```python
# cdc_streaming_client.py
from snowflake.ingest.streaming import SnowflakeStreamingIngestClient
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.backends import default_backend
import json
import time

# Load private key
with open('rsa_key.p8', 'rb') as key_file:
    private_key = serialization.load_pem_private_key(
        key_file.read(),
        password=None,
        backend=default_backend()
    )

# Initialize client
client = SnowflakeStreamingIngestClient(
    account='<your_account>',
    user='<your_user>',
    private_key=private_key,
    role='ACCOUNTADMIN',
    warehouse='CDC_WH',
    database='CDC_PIPELINE',
    schema='RAW'
)

# Open channel
channel = client.open_channel(
    channel_name='cdc_channel',
    table_name='CHANGES',
    on_error='CONTINUE'
)

print('✓ Snowpipe Streaming channel opened')

# Insert CDC event
def insert_cdc_event(operation, table_name, before_data, after_data):
    row = {
        'OPERATION': operation,
        'TABLE_NAME': table_name,
        'BEFORE': json.dumps(before_data) if before_data else None,
        'AFTER': json.dumps(after_data) if after_data else None,
        'TS_MS': int(time.time() * 1000)
    }
    
    response = channel.insert_row(row)
    return response

# Example: Insert a customer update
before = {'customer_id': 1, 'email': 'old@example.com', 'name': 'John Doe'}
after = {'customer_id': 1, 'email': 'new@example.com', 'name': 'John Doe'}

insert_cdc_event('UPDATE', 'customers', before, after)
print('✓ CDC event inserted')

# Close channel when done
channel.close()
client.close()
```

**Generate Sample CDC Events:**
```python
# generate_cdc_events.py
import random
import time
from cdc_streaming_client import insert_cdc_event

# Simulate customer changes
customers = {}

def generate_insert():
    customer_id = len(customers) + 1
    customer = {
        'customer_id': customer_id,
        'email': f'customer{customer_id}@example.com',
        'name': f'Customer {customer_id}',
        'created_at': time.time()
    }
    customers[customer_id] = customer
    insert_cdc_event('INSERT', 'customers', None, customer)
    print(f'✓ INSERT customer {customer_id}')

def generate_update():
    if not customers:
        return
    customer_id = random.choice(list(customers.keys()))
    before = customers[customer_id].copy()
    after = before.copy()
    after['email'] = f'updated{customer_id}@example.com'
    after['updated_at'] = time.time()
    customers[customer_id] = after
    insert_cdc_event('UPDATE', 'customers', before, after)
    print(f'✓ UPDATE customer {customer_id}')

def generate_delete():
    if not customers:
        return
    customer_id = random.choice(list(customers.keys()))
    before = customers[customer_id]
    del customers[customer_id]
    insert_cdc_event('DELETE', 'customers', before, None)
    print(f'✓ DELETE customer {customer_id}')

# Generate events continuously
while True:
    operation = random.choices(
        ['INSERT', 'UPDATE', 'DELETE'],
        weights=[0.5, 0.4, 0.1]
    )[0]
    
    if operation == 'INSERT':
        generate_insert()
    elif operation == 'UPDATE':
        generate_update()
    else:
        generate_delete()
    
    time.sleep(0.1)  # 10 events per second
```

### 2. Create Streams

**Stream on Raw Changes:**
```sql
-- Create stream to track changes
CREATE STREAM CDC_PIPELINE.RAW.CHANGES_STREAM 
  ON TABLE CDC_PIPELINE.RAW.CHANGES;

-- View stream metadata
SHOW STREAMS;

-- Check if stream has data
SELECT SYSTEM$STREAM_HAS_DATA('CDC_PIPELINE.RAW.CHANGES_STREAM');

-- Query stream (shows changes since last consumption)
SELECT * FROM CDC_PIPELINE.RAW.CHANGES_STREAM LIMIT 10;
```

### 3. Create Staging Tables

```sql
-- Staging table for customers
CREATE TABLE CDC_PIPELINE.STAGING.CUSTOMERS (
    customer_id INTEGER,
    email VARCHAR(255),
    name VARCHAR(255),
    created_at TIMESTAMP,
    updated_at TIMESTAMP,
    _operation VARCHAR(10),
    _ingested_at TIMESTAMP_LTZ
);

-- Stream on staging
CREATE STREAM CDC_PIPELINE.STAGING.CUSTOMERS_STREAM 
  ON TABLE CDC_PIPELINE.STAGING.CUSTOMERS;
```

### 4. Create Tasks for Processing

**Task 1: Parse CDC Events**
```sql
-- Task to parse raw CDC events into staging
CREATE TASK CDC_PIPELINE.RAW.PARSE_CDC_EVENTS
  WAREHOUSE = CDC_WH
  SCHEDULE = '1 MINUTE'
WHEN
  SYSTEM$STREAM_HAS_DATA('CDC_PIPELINE.RAW.CHANGES_STREAM')
AS
INSERT INTO CDC_PIPELINE.STAGING.CUSTOMERS
SELECT
    PARSE_JSON(after):customer_id::INTEGER AS customer_id,
    PARSE_JSON(after):email::VARCHAR AS email,
    PARSE_JSON(after):name::VARCHAR AS name,
    TO_TIMESTAMP(PARSE_JSON(after):created_at::NUMBER) AS created_at,
    TO_TIMESTAMP(PARSE_JSON(after):updated_at::NUMBER) AS updated_at,
    operation AS _operation,
    ingested_at AS _ingested_at
FROM CDC_PIPELINE.RAW.CHANGES_STREAM
WHERE table_name = 'customers'
  AND operation IN ('INSERT', 'UPDATE');

-- Resume task
ALTER TASK CDC_PIPELINE.RAW.PARSE_CDC_EVENTS RESUME;

-- Check task status
SHOW TASKS LIKE 'PARSE_CDC_EVENTS';
```

**Monitor Task Execution:**
```sql
-- View task history
SELECT
    name,
    state,
    scheduled_time,
    completed_time,
    DATEDIFF('second', scheduled_time, completed_time) AS execution_seconds,
    error_code,
    error_message
FROM TABLE(INFORMATION_SCHEMA.TASK_HISTORY(
    TASK_NAME => 'PARSE_CDC_EVENTS',
    SCHEDULED_TIME_RANGE_START => DATEADD('hour', -1, CURRENT_TIMESTAMP())
))
ORDER BY scheduled_time DESC;
```

### 5. SCD Type 2 Implementation

**Create SCD Type 2 Table:**
```sql
-- Target table with historical tracking
CREATE TABLE CDC_PIPELINE.PROD.CUSTOMERS_SCD (
    customer_key INTEGER AUTOINCREMENT,
    customer_id INTEGER,
    email VARCHAR(255),
    name VARCHAR(255),
    valid_from TIMESTAMP,
    valid_to TIMESTAMP,
    is_current BOOLEAN,
    PRIMARY KEY (customer_key)
);
```

**Task 2: Apply SCD Type 2 Logic**
```sql
-- Task to maintain SCD Type 2
CREATE TASK CDC_PIPELINE.STAGING.APPLY_SCD_TYPE2
  WAREHOUSE = CDC_WH
  AFTER CDC_PIPELINE.RAW.PARSE_CDC_EVENTS
WHEN
  SYSTEM$STREAM_HAS_DATA('CDC_PIPELINE.STAGING.CUSTOMERS_STREAM')
AS
BEGIN
    -- Step 1: Expire changed records
    UPDATE CDC_PIPELINE.PROD.CUSTOMERS_SCD t
    SET 
        valid_to = s.updated_at,
        is_current = FALSE
    FROM (
        SELECT DISTINCT
            customer_id,
            updated_at
        FROM CDC_PIPELINE.STAGING.CUSTOMERS_STREAM
        WHERE _operation IN ('INSERT', 'UPDATE')
    ) s
    WHERE t.customer_id = s.customer_id
      AND t.is_current = TRUE
      AND (t.email != s.email OR t.name != s.name);
    
    -- Step 2: Insert new current records
    INSERT INTO CDC_PIPELINE.PROD.CUSTOMERS_SCD (
        customer_id, email, name, valid_from, valid_to, is_current
    )
    SELECT
        customer_id,
        email,
        name,
        COALESCE(updated_at, created_at) AS valid_from,
        NULL AS valid_to,
        TRUE AS is_current
    FROM CDC_PIPELINE.STAGING.CUSTOMERS_STREAM
    WHERE _operation IN ('INSERT', 'UPDATE');
END;

-- Resume task
ALTER TASK CDC_PIPELINE.STAGING.APPLY_SCD_TYPE2 RESUME;
```

**Query Historical Data:**
```sql
-- Get current records
SELECT * FROM CDC_PIPELINE.PROD.CUSTOMERS_SCD
WHERE is_current = TRUE;

-- Get historical records for a customer
SELECT
    customer_id,
    email,
    name,
    valid_from,
    valid_to,
    is_current
FROM CDC_PIPELINE.PROD.CUSTOMERS_SCD
WHERE customer_id = 1
ORDER BY valid_from;

-- Point-in-time query (as of specific date)
SELECT
    customer_id,
    email,
    name
FROM CDC_PIPELINE.PROD.CUSTOMERS_SCD
WHERE customer_id = 1
  AND valid_from <= '2024-01-15'::TIMESTAMP
  AND (valid_to IS NULL OR valid_to > '2024-01-15'::TIMESTAMP);
```

### 6. Dynamic Tables

**Create Dynamic Table for Aggregations:**
```sql
-- Real-time customer metrics
CREATE DYNAMIC TABLE CDC_PIPELINE.PROD.CUSTOMER_METRICS
  TARGET_LAG = '1 MINUTE'
  WAREHOUSE = CDC_WH
AS
SELECT
    customer_id,
    name,
    email,
    COUNT(*) AS version_count,
    MIN(valid_from) AS first_seen,
    MAX(valid_from) AS last_updated
FROM CDC_PIPELINE.PROD.CUSTOMERS_SCD
GROUP BY customer_id, name, email;

-- Query dynamic table (automatically refreshed)
SELECT * FROM CDC_PIPELINE.PROD.CUSTOMER_METRICS;

-- Check refresh status
SHOW DYNAMIC TABLES LIKE 'CUSTOMER_METRICS';
```

### 7. Monitoring & Alerting

**Stream Lag Monitoring:**
```sql
-- Check stream lag
SELECT
    'CHANGES_STREAM' AS stream_name,
    SYSTEM$STREAM_GET_TABLE_TIMESTAMP('CDC_PIPELINE.RAW.CHANGES_STREAM') AS stream_timestamp,
    CURRENT_TIMESTAMP() AS current_timestamp,
    DATEDIFF('second', 
        SYSTEM$STREAM_GET_TABLE_TIMESTAMP('CDC_PIPELINE.RAW.CHANGES_STREAM'),
        CURRENT_TIMESTAMP()
    ) AS lag_seconds
FROM DUAL;
```

**Task Execution Monitoring:**
```sql
-- Monitor all tasks
SELECT
    database_name,
    schema_name,
    name AS task_name,
    state,
    schedule,
    CASE 
        WHEN state = 'started' THEN 'Running'
        WHEN state = 'suspended' THEN 'Paused'
        ELSE 'Unknown'
    END AS status
FROM TABLE(INFORMATION_SCHEMA.TASKS)
WHERE database_name = 'CDC_PIPELINE';

-- Failed task executions
SELECT
    name,
    scheduled_time,
    error_code,
    error_message
FROM TABLE(INFORMATION_SCHEMA.TASK_HISTORY(
    SCHEDULED_TIME_RANGE_START => DATEADD('day', -1, CURRENT_TIMESTAMP())
))
WHERE state = 'FAILED'
ORDER BY scheduled_time DESC;
```

**Data Quality Checks:**
```sql
-- Check for duplicate current records
SELECT
    customer_id,
    COUNT(*) AS duplicate_count
FROM CDC_PIPELINE.PROD.CUSTOMERS_SCD
WHERE is_current = TRUE
GROUP BY customer_id
HAVING COUNT(*) > 1;

-- Check for SCD gaps
SELECT
    customer_id,
    valid_from,
    valid_to,
    LEAD(valid_from) OVER (PARTITION BY customer_id ORDER BY valid_from) AS next_valid_from,
    CASE 
        WHEN valid_to IS NOT NULL 
         AND valid_to != LEAD(valid_from) OVER (PARTITION BY customer_id ORDER BY valid_from)
        THEN 'GAP DETECTED'
        ELSE 'OK'
    END AS gap_status
FROM CDC_PIPELINE.PROD.CUSTOMERS_SCD
WHERE valid_to IS NOT NULL
ORDER BY customer_id, valid_from;
```

### 8. Performance Optimization

**Clustering Keys:**
```sql
-- Add clustering for better query performance
ALTER TABLE CDC_PIPELINE.PROD.CUSTOMERS_SCD 
  CLUSTER BY (customer_id, valid_from);

-- Check clustering information
SELECT SYSTEM$CLUSTERING_INFORMATION('CDC_PIPELINE.PROD.CUSTOMERS_SCD');
```

**Task Optimization:**
```sql
-- Adjust task schedule based on load
ALTER TASK CDC_PIPELINE.RAW.PARSE_CDC_EVENTS 
  SET SCHEDULE = '30 SECONDS';  -- More frequent for high-volume

-- Use larger warehouse for heavy processing
ALTER TASK CDC_PIPELINE.STAGING.APPLY_SCD_TYPE2 
  SET WAREHOUSE = CDC_WH_LARGE;
```

## Success Criteria

- [ ] Snowpipe Streaming client working with key pair auth
- [ ] CDC events ingesting in real-time (<1 second latency)
- [ ] Streams tracking changes on raw and staging tables
- [ ] Tasks processing streams automatically
- [ ] SCD Type 2 table maintaining historical records
- [ ] Dynamic Tables refreshing aggregations
- [ ] Monitoring queries tracking lag and failures
- [ ] Data quality checks passing
- [ ] Task orchestration working (AFTER dependencies)
- [ ] Documentation complete with architecture diagram

## Cost Optimization

### Compute Costs
- Use **XSMALL** warehouses for tasks
- Set **AUTO_SUSPEND** to 60 seconds
- Use **AFTER** dependencies to avoid concurrent task runs
- Monitor task execution time and optimize SQL

### Snowpipe Streaming Costs
- Charged per 1000 rows ingested
- More cost-effective than Snowpipe for high-volume
- No file staging costs

### Storage Costs
- SCD Type 2 increases storage (historical records)
- Use **Time Travel** retention wisely (default 1 day)
- Consider archiving old historical data

**Cost Comparison:**
```
Snowpipe (file-based):
- File staging: $0.023/GB
- Compute: $2/credit
- Latency: Minutes

Snowpipe Streaming:
- No staging costs
- $0.014 per 1000 rows
- Latency: <1 second
```

## Common Challenges

### Key Pair Authentication Fails
Verify public key format (no headers/footers), check user has key assigned

### Stream Shows No Data
Ensure data was inserted after stream creation, check `SYSTEM$STREAM_HAS_DATA()`

### Task Not Running
Verify task is RESUMED, check WHEN condition, ensure warehouse is available

### SCD Type 2 Duplicates
Add unique constraint on (customer_id, is_current=TRUE), check merge logic

## Learning Outcomes

- Implement Snowpipe Streaming for sub-second ingestion
- Design CDC pipelines with Streams and Tasks
- Build SCD Type 2 patterns for historical tracking
- Create Dynamic Tables for real-time aggregations
- Monitor streaming pipelines with Snowflake views
- Optimize task scheduling and warehouse usage
- Handle errors and ensure data quality

## Next Steps

1. Add to portfolio with CDC architecture diagram
2. Write blog post: "Snowpipe vs Snowpipe Streaming: When to Use Each"
3. Continue to Project S3: Snowpark ML
4. Extend with Kafka integration for real sources

## Resources

- [Snowpipe Streaming Docs](https://docs.snowflake.com/en/user-guide/data-load-snowpipe-streaming-overview)
- [Streams Guide](https://docs.snowflake.com/en/user-guide/streams-intro)
- [Tasks Guide](https://docs.snowflake.com/en/user-guide/tasks-intro)
- [Dynamic Tables](https://docs.snowflake.com/en/user-guide/dynamic-tables-intro)
- [SCD Type 2 Patterns](https://docs.snowflake.com/en/user-guide/data-pipelines-scd)
