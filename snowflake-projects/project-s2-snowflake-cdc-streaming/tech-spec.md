# Technical Specification: Real-Time CDC Pipeline with Snowpipe Streaming

## Architecture Overview

### System Components
```
PostgreSQL → Debezium/DMS → Kafka → Snowpipe Streaming → Snowflake
                                                              ↓
                                                         Streams
                                                              ↓
                                                          Tasks
                                                              ↓
                                                    Dynamic Tables
```

## Technology Stack

### Core Technologies
- **Source**: PostgreSQL 15+
- **CDC**: Debezium 2.5+ or AWS DMS
- **Streaming**: Apache Kafka 3.6+ (optional)
- **Ingestion**: Snowpipe Streaming API
- **Processing**: Snowflake Streams, Tasks, Dynamic Tables
- **Language**: Python 3.11+ (for Snowpipe Streaming client)

### Supporting Technologies
- **Orchestration**: Snowflake Tasks (native)
- **Monitoring**: Snowflake query history, task history
- **IaC**: Terraform
- **CI/CD**: GitHub Actions

## Detailed Design

### 1. Source Database Setup

#### PostgreSQL Configuration
```sql
-- Enable logical replication
ALTER SYSTEM SET wal_level = logical;
ALTER SYSTEM SET max_replication_slots = 4;

-- Create sample tables
CREATE TABLE customers (
    customer_id SERIAL PRIMARY KEY,
    email VARCHAR(255),
    name VARCHAR(255),
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

CREATE TABLE orders (
    order_id SERIAL PRIMARY KEY,
    customer_id INTEGER REFERENCES customers(customer_id),
    amount DECIMAL(10,2),
    status VARCHAR(50),
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);
```

### 2. CDC Implementation

#### Debezium Configuration
```json
{
  "name": "postgres-connector",
  "config": {
    "connector.class": "io.debezium.connector.postgresql.PostgresConnector",
    "database.hostname": "localhost",
    "database.port": "5432",
    "database.user": "debezium",
    "database.password": "dbz",
    "database.dbname": "sourcedb",
    "database.server.name": "postgres",
    "table.include.list": "public.customers,public.orders",
    "plugin.name": "pgoutput",
    "publication.autocreate.mode": "filtered"
  }
}
```

### 3. Snowpipe Streaming Implementation

#### Python Client
```python
from snowflake.ingest import SimpleIngestManager
from snowflake.ingest import StagedFile
from snowflake.ingest.streaming import SnowflakeStreamingIngestClient
from snowflake.ingest.streaming import SnowflakeStreamingIngestChannel

# Initialize client
client = SnowflakeStreamingIngestClient(
    account="<account>",
    user="<user>",
    private_key="<key>",
    role="<role>",
    warehouse="<warehouse>",
    database="<database>",
    schema="<schema>"
)

# Create channel
channel = client.open_channel(
    channel_name="cdc_channel",
    table_name="raw_changes",
    on_error="CONTINUE"
)

# Insert rows
def process_cdc_event(event):
    row = {
        "operation": event["op"],
        "table_name": event["source"]["table"],
        "before": event.get("before"),
        "after": event.get("after"),
        "ts_ms": event["ts_ms"]
    }
    channel.insert_row(row)
```

### 4. Snowflake Schema Design

#### Landing Tables
```sql
-- Raw CDC events
CREATE TABLE raw_changes (
    operation VARCHAR(10),
    table_name VARCHAR(255),
    before VARIANT,
    after VARIANT,
    ts_ms BIGINT,
    ingested_at TIMESTAMP_LTZ DEFAULT CURRENT_TIMESTAMP()
);

-- Staging tables
CREATE TABLE staging_customers (
    customer_id INTEGER,
    email VARCHAR(255),
    name VARCHAR(255),
    created_at TIMESTAMP,
    updated_at TIMESTAMP,
    _operation VARCHAR(10),
    _ingested_at TIMESTAMP_LTZ
);
```

### 5. Streams Implementation

#### Change Tracking
```sql
-- Create stream on raw changes
CREATE STREAM raw_changes_stream ON TABLE raw_changes;

-- Create stream on staging
CREATE STREAM staging_customers_stream ON TABLE staging_customers;
```

### 6. Tasks for Processing

#### Parse CDC Events
```sql
-- Task to parse raw CDC events
CREATE TASK parse_cdc_events
  WAREHOUSE = compute_wh
  SCHEDULE = '1 MINUTE'
WHEN
  SYSTEM$STREAM_HAS_DATA('raw_changes_stream')
AS
INSERT INTO staging_customers
SELECT
    after:customer_id::INTEGER as customer_id,
    after:email::VARCHAR as email,
    after:name::VARCHAR as name,
    after:created_at::TIMESTAMP as created_at,
    after:updated_at::TIMESTAMP as updated_at,
    operation as _operation,
    ingested_at as _ingested_at
FROM raw_changes_stream
WHERE table_name = 'customers';

ALTER TASK parse_cdc_events RESUME;
```

#### SCD Type 2 Implementation
```sql
-- Target table with SCD Type 2
CREATE TABLE customers_scd (
    customer_key INTEGER AUTOINCREMENT,
    customer_id INTEGER,
    email VARCHAR(255),
    name VARCHAR(255),
    valid_from TIMESTAMP,
    valid_to TIMESTAMP,
    is_current BOOLEAN,
    PRIMARY KEY (customer_key)
);

-- Task for SCD Type 2 merge
CREATE TASK apply_scd_type2
  WAREHOUSE = compute_wh
  AFTER parse_cdc_events
WHEN
  SYSTEM$STREAM_HAS_DATA('staging_customers_stream')
AS
MERGE INTO customers_scd t
USING (
    SELECT
        customer_id,
        email,
        name,
        updated_at as valid_from
    FROM staging_customers_stream
    WHERE _operation IN ('INSERT', 'UPDATE')
) s
ON t.customer_id = s.customer_id AND t.is_current = TRUE
WHEN MATCHED AND (
    t.email != s.email OR t.name != s.name
) THEN UPDATE SET
    valid_to = s.valid_from,
    is_current = FALSE
WHEN NOT MATCHED THEN INSERT (
    customer_id, email, name, valid_from, valid_to, is_current
) VALUES (
    s.customer_id, s.email, s.name, s.valid_from, NULL, TRUE
);

-- Insert new current records
INSERT INTO customers_scd (customer_id, email, name, valid_from, valid_to, is_current)
SELECT
    s.customer_id,
    s.email,
    s.name,
    s.valid_from,
    NULL,
    TRUE
FROM (
    SELECT
        customer_id,
        email,
        name,
        updated_at as valid_from
    FROM staging_customers_stream
    WHERE _operation IN ('INSERT', 'UPDATE')
) s
JOIN customers_scd t
    ON s.customer_id = t.customer_id
    AND t.is_current = FALSE
    AND NOT EXISTS (
        SELECT 1 FROM customers_scd
        WHERE customer_id = s.customer_id AND is_current = TRUE
    );

ALTER TASK apply_scd_type2 RESUME;
```

### 7. Dynamic Tables

#### Real-Time Aggregations
```sql
-- Dynamic table for customer metrics
CREATE DYNAMIC TABLE customer_metrics
  TARGET_LAG = '1 MINUTE'
  WAREHOUSE = compute_wh
AS
SELECT
    c.customer_id,
    c.name,
    c.email,
    COUNT(o.order_id) as total_orders,
    SUM(o.amount) as total_spent,
    MAX(o.created_at) as last_order_date
FROM customers_scd c
LEFT JOIN orders o ON c.customer_id = o.customer_id
WHERE c.is_current = TRUE
GROUP BY c.customer_id, c.name, c.email;
```

### 8. Monitoring & Alerting

#### Latency Monitoring
```sql
-- Monitor stream lag
SELECT
    'raw_changes_stream' as stream_name,
    SYSTEM$STREAM_GET_TABLE_TIMESTAMP('raw_changes_stream') as stream_timestamp,
    CURRENT_TIMESTAMP() as current_timestamp,
    DATEDIFF('second', stream_timestamp, CURRENT_TIMESTAMP()) as lag_seconds
FROM dual;

-- Monitor task execution
SELECT
    name,
    state,
    scheduled_time,
    completed_time,
    DATEDIFF('second', scheduled_time, completed_time) as execution_seconds,
    error_code,
    error_message
FROM TABLE(INFORMATION_SCHEMA.TASK_HISTORY())
WHERE scheduled_time >= DATEADD('hour', -1, CURRENT_TIMESTAMP())
ORDER BY scheduled_time DESC;
```

#### Data Quality Checks
```sql
-- Check for duplicate records
SELECT
    customer_id,
    COUNT(*) as record_count
FROM customers_scd
WHERE is_current = TRUE
GROUP BY customer_id
HAVING COUNT(*) > 1;

-- Check for gaps in SCD
SELECT
    customer_id,
    valid_from,
    valid_to,
    LEAD(valid_from) OVER (PARTITION BY customer_id ORDER BY valid_from) as next_valid_from
FROM customers_scd
WHERE valid_to IS NOT NULL
HAVING valid_to != next_valid_from;
```

### 9. Performance Comparison

#### Snowpipe vs. Snowpipe Streaming
```python
import time
from datetime import datetime

# Benchmark Snowpipe Streaming
start = time.time()
for i in range(10000):
    channel.insert_row(generate_test_row())
streaming_duration = time.time() - start

# Benchmark traditional Snowpipe
# (stage files, trigger Snowpipe, measure end-to-end)
```

## Implementation Phases

### Phase 1: Foundation (Day 1)
- Set up PostgreSQL with sample data
- Configure Debezium for CDC
- Implement Snowpipe Streaming client
- Create landing tables in Snowflake

### Phase 2: Processing (Day 2)
- Implement Streams for change tracking
- Create Tasks for CDC parsing
- Build SCD Type 2 logic
- Test end-to-end flow

### Phase 3: Advanced Features (Day 3)
- Implement Dynamic Tables
- Add data quality checks
- Create monitoring queries
- Performance benchmarking

### Phase 4: Documentation (Day 4)
- Architecture diagrams
- CDC migration guide
- Performance analysis
- Cost optimization recommendations

## Deliverables

### Code
- Python Snowpipe Streaming client
- SQL scripts for all objects
- Debezium configuration
- Monitoring queries

### Documentation
- Architecture diagram
- "CDC Migration Guide: PostgreSQL → Snowflake"
- Performance benchmarks (Snowpipe vs. Streaming)
- Cost analysis
- SCD Type 2 implementation guide

### Metrics
- Latency measurements (p50, p95, p99)
- Throughput benchmarks
- Cost per million events
- Query performance on streaming data
