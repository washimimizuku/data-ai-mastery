# Design Document: High-Performance Data Pipeline with Rust + FastAPI

## Overview

This design document describes a production-grade data pipeline that combines high-performance Rust processing with modern Python-based data engineering tools. The system ingests data through REST APIs, processes it using Rust for 10-50x performance improvements, orchestrates workflows with Airflow, transforms data using dbt, validates quality with Great Expectations, and visualizes results through Streamlit dashboards.

The architecture follows a layered approach: ingestion layer (FastAPI), streaming layer (Kinesis), processing layer (Rust), storage layer (S3/Iceberg), cataloging layer (Iceberg Catalog), transformation layer (dbt/Snowflake), quality layer (Great Expectations), and visualization layer (Streamlit).

**Apache Iceberg Integration**: The pipeline uses Apache Iceberg as the table format for all data stored in S3. Iceberg provides ACID transactions, schema evolution, time travel, partition evolution, and hidden partitioning, making it ideal for production data lakes.

## Architecture

### High-Level Architecture

```
┌─────────────┐
│   Clients   │
└──────┬──────┘
       │ HTTP POST
       ▼
┌─────────────────┐
│  FastAPI Service│
│  (Ingestion)    │
└────────┬────────┘
         │ Publish
         ▼
┌─────────────────┐
│ Kinesis Stream  │
└────────┬────────┘
         │ Consume
         ▼
┌─────────────────┐      ┌──────────────────┐
│ Rust Processor  │─────▶│   S3 Bucket      │
│ (PyO3 Bindings) │      │ (Iceberg Tables) │
└─────────────────┘      └──────┬───────────┘
                                │
                                ▼
                         ┌──────────────────┐
                         │ Iceberg Catalog  │
                         │ (Glue/REST/Hive) │
                         └──────┬───────────┘
                                │
         ┌──────────────────────┴────────────────┐
         ▼                                       ▼
┌─────────────────┐                    ┌─────────────────┐
│     Airflow     │                    │   Snowflake     │
│  (Orchestrator) │───────────────────▶│ (Data Warehouse)│
└────────┬────────┘                    └────────┬────────┘
         │                                      │
         │ Trigger                              │
         ▼                                      │
┌─────────────────┐                            │
│   dbt Models    │────────────────────────────┘
│ (Transformation)│
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│Great Expectations│
│ (Data Quality)  │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│    Streamlit    │
│   (Dashboard)   │
└─────────────────┘
```

### Technology Stack

- **Languages**: Python 3.11+, Rust 1.75+
- **API Framework**: FastAPI 0.104+
- **Streaming**: AWS Kinesis
- **Processing**: Rust with PyO3 for Python bindings
- **Storage**: AWS S3 with Apache Iceberg table format
- **Iceberg**: Apache Iceberg 1.4+ (Parquet file format)
- **Cataloging**: Iceberg Catalog (AWS Glue Catalog, REST Catalog, or Hive Metastore)
- **Orchestration**: Apache Airflow 2.7+
- **Transformation**: dbt 1.7+ with dbt-iceberg adapter
- **Data Warehouse**: Snowflake (with Iceberg table support)
- **Data Quality**: Great Expectations
- **Visualization**: Streamlit
- **Testing**: cargo test (Rust), pytest (Python), dbt test

## Components and Interfaces

### 1. FastAPI Service (Ingestion Layer)

**Purpose**: Provides REST API endpoints for data ingestion and job tracking.

**Endpoints**:
- `POST /api/v1/ingest` - Accept data for processing
- `GET /api/v1/status/{job_id}` - Query job status
- `GET /api/v1/metrics` - Retrieve processing metrics
- `POST /api/v1/trigger/{pipeline}` - Manually trigger pipeline
- `GET /api/v1/health` - Health check endpoint

**Key Responsibilities**:
- Validate incoming requests
- Generate unique Job IDs
- Publish data to Kinesis Stream
- Track job status in database
- Handle async operations

**Interface**:
```python
class IngestionService:
    async def ingest_data(self, data: dict) -> JobResponse
    async def get_job_status(self, job_id: str) -> JobStatus
    async def publish_to_kinesis(self, job_id: str, data: dict) -> None
    async def get_metrics(self) -> MetricsResponse
```

### 2. Rust Processor (Processing Layer)

**Purpose**: High-performance data parsing, validation, transformation, and Parquet writing.

**Core Functions**:
```rust
pub fn parse_json(data: &[u8]) -> Result<Vec<Event>, ParseError>
pub fn parse_csv(data: &[u8]) -> Result<Vec<Event>, ParseError>
pub fn validate_schema(event: &Event, schema: &Schema) -> Result<(), ValidationError>
pub fn transform_batch(events: Vec<Event>, rules: &TransformRules) -> Vec<TransformedEvent>
pub fn deduplicate(events: Vec<Event>) -> Vec<Event>
pub fn write_iceberg_parquet(events: &[TransformedEvent], table_location: &str, schema: &IcebergSchema) -> Result<DataFile, IOError>
pub fn create_iceberg_manifest(data_files: Vec<DataFile>, snapshot_id: i64) -> Result<ManifestFile, IOError>
```

**Python Bindings (PyO3)**:
```python
import data_processor_rs
from pyiceberg.catalog import load_catalog

# Exposed to Python
events = data_processor_rs.parse_json(raw_bytes)
validated = data_processor_rs.validate_schema(events, schema)
transformed = data_processor_rs.transform_batch(validated, rules)

# Write to Iceberg table via PyIceberg
catalog = load_catalog("default")
table = catalog.load_table("events.raw_events")
data_file = data_processor_rs.write_iceberg_parquet(transformed, table.location(), table.schema())

# Commit to Iceberg table
table.append(data_file)
```

**Performance Targets**:
- JSON parsing: 20x faster than Python
- Schema validation: 25x faster than Python
- Transformation: 15x faster than Python
- Parquet writing: 14x faster than Python

### 3. Kinesis Consumer (Streaming Layer)

**Purpose**: Consume data from Kinesis and orchestrate Rust processing.

**Key Responsibilities**:
- Poll Kinesis Stream for new records
- Batch records for efficient processing
- Call Rust processing functions
- Write results to Iceberg tables via PyIceberg
- Commit Iceberg snapshots with ACID guarantees
- Maintain checkpoints
- Handle retries on failures

**Interface**:
```python
class KinesisConsumer:
    def __init__(self, catalog: Catalog):
        self.catalog = catalog
        self.table = catalog.load_table("events.raw_events")
    
    def consume_stream(self, stream_name: str) -> None
    def process_batch(self, records: List[Record]) -> ProcessingResult
    def write_to_iceberg(self, data_files: List[DataFile]) -> None
    def checkpoint(self, sequence_number: str) -> None
    def handle_failure(self, batch: Batch, error: Exception) -> None
```

### 4. Airflow DAGs (Orchestration Layer)

**Purpose**: Orchestrate pipeline workflows with dependencies and error handling.

**Main Pipeline DAG**:
```python
# dag_data_pipeline.py
detect_new_iceberg_snapshot >> validate_iceberg_metadata >> 
run_dbt_staging >> run_dbt_intermediate >> run_dbt_marts >> 
run_quality_checks >> update_dashboard >> send_notifications
```

**dbt Transformation DAG**:
```python
# dag_dbt_transform.py
dbt_deps >> dbt_seed >> dbt_run >> dbt_test >> dbt_docs_generate
```

**Key Features**:
- Iceberg snapshot sensors for detecting new data commits
- Retry logic with exponential backoff
- Task dependencies and parallelization
- Error notifications
- Manual trigger support
- Iceberg time travel for data recovery

### 5. dbt Models (Transformation Layer)

**Purpose**: Transform raw data into analytics-ready dimensional models.

**Layer Structure**:

**Staging Layer** (`models/staging/`):
- Clean and standardize raw data
- Apply basic type conversions
- Remove duplicates
- Example: `stg_events.sql`, `stg_users.sql`

**Intermediate Layer** (`models/intermediate/`):
- Apply business logic
- Join related entities
- Calculate derived fields
- Example: `int_user_events.sql`, `int_event_aggregates.sql`

**Marts Layer** (`models/marts/`):
- Create dimensional models (star schema)
- Implement slowly changing dimensions (SCD Type 2)
- Build fact tables with metrics
- Example: `dim_users.sql`, `dim_products.sql`, `fact_events.sql`

**Interface**:
```sql
-- dim_users.sql (Dimension Table - Iceberg)
{{ config(
    materialized='incremental',
    file_format='iceberg',
    unique_key='user_sk'
) }}

SELECT
    {{ dbt_utils.surrogate_key(['user_id', 'valid_from']) }} as user_sk,
    user_id,
    user_name,
    email,
    created_at,
    valid_from,
    valid_to,
    is_current
FROM {{ ref('int_users_with_history') }}

-- fact_events.sql (Fact Table - Iceberg)
{{ config(
    materialized='incremental',
    file_format='iceberg',
    partition_by=['event_date'],
    unique_key='event_sk'
) }}

SELECT
    {{ dbt_utils.surrogate_key(['event_id']) }} as event_sk,
    {{ dbt_utils.surrogate_key(['user_id', 'event_timestamp']) }} as user_sk,
    event_timestamp,
    DATE(event_timestamp) as event_date,
    event_type,
    event_value,
    event_properties
FROM {{ ref('int_events_enriched') }}
```

### 6. Great Expectations (Data Quality Layer)

**Purpose**: Validate data quality at each pipeline stage.

**Expectation Suites**:
- `staging_suite`: Validate raw data quality
- `intermediate_suite`: Validate business logic correctness
- `marts_suite`: Validate dimensional model integrity

**Key Validations**:
```python
# Example expectations
expect_column_values_to_not_be_null(column="user_id")
expect_column_values_to_be_of_type(column="event_timestamp", type_="datetime")
expect_column_values_to_be_between(column="event_value", min_value=0, max_value=1000000)
expect_column_values_to_be_in_set(column="event_type", value_set=["click", "view", "purchase"])
expect_table_row_count_to_be_between(min_value=1000, max_value=10000000)
```

**Integration**:
- Airflow tasks trigger validation checkpoints
- Results stored in validation database
- Failures trigger alerts
- Dashboard displays quality metrics

### 7. Streamlit Dashboard (Visualization Layer)

**Purpose**: Provide interactive visualizations of pipeline metrics and data quality.

**Key Features**:
- Real-time pipeline status
- Processing throughput metrics
- Data quality scorecards
- Time series visualizations
- Drill-down capabilities
- Filter and search functionality

**Interface**:
```python
class DashboardService:
    def get_pipeline_metrics(self) -> PipelineMetrics
    def get_quality_metrics(self) -> QualityMetrics
    def query_dimensional_data(self, filters: dict) -> pd.DataFrame
    def get_processing_history(self, time_range: TimeRange) -> pd.DataFrame
```

### 8. Apache Iceberg Integration

**Purpose**: Provide ACID transactions, schema evolution, and efficient data management for the data lake.

**Key Features**:

**ACID Transactions**:
- Atomic commits ensure data consistency
- Snapshot isolation for concurrent reads/writes
- Rollback capability using time travel

**Schema Evolution**:
- Add, drop, rename columns without rewriting data
- Schema changes are metadata-only operations
- Full schema history tracking

**Hidden Partitioning**:
- Partition evolution without breaking queries
- Automatic partition pruning
- No need to specify partitions in queries

**Time Travel**:
```python
# Query data as of specific timestamp
table.scan(snapshot_id=12345).to_arrow()

# Query data as of specific time
table.scan(as_of_timestamp=datetime(2024, 1, 1)).to_arrow()

# Rollback to previous snapshot
table.rollback_to_snapshot(12345)
```

**Catalog Configuration**:
```python
# AWS Glue Catalog
catalog_config = {
    "type": "glue",
    "warehouse": "s3://data-lake/warehouse",
    "catalog-impl": "org.apache.iceberg.aws.glue.GlueCatalog",
    "io-impl": "org.apache.iceberg.aws.s3.S3FileIO"
}

# REST Catalog (alternative)
catalog_config = {
    "type": "rest",
    "uri": "https://iceberg-catalog.example.com",
    "warehouse": "s3://data-lake/warehouse"
}
```

**Table Creation**:
```python
from pyiceberg.catalog import load_catalog
from pyiceberg.schema import Schema
from pyiceberg.types import NestedField, StringType, LongType, DoubleType, TimestampType

catalog = load_catalog("default", **catalog_config)

schema = Schema(
    NestedField(1, "event_id", StringType(), required=True),
    NestedField(2, "user_id", StringType(), required=True),
    NestedField(3, "event_type", StringType(), required=True),
    NestedField(4, "event_timestamp", TimestampType(), required=True),
    NestedField(5, "event_value", DoubleType(), required=False),
    NestedField(6, "event_date", StringType(), required=True),
    NestedField(7, "event_hour", LongType(), required=True),
)

# Create table with partitioning
table = catalog.create_table(
    identifier="events.raw_events",
    schema=schema,
    partition_spec=PartitionSpec(
        PartitionField(source_id=6, field_id=1000, transform=IdentityTransform(), name="event_date")
    )
)
```

**Data Writing**:
```python
# Append data to Iceberg table
from pyiceberg.io.pyarrow import write_file

# Write Parquet file with Iceberg metadata
data_file = write_file(
    table=table,
    data=arrow_table,
    file_path=f"s3://bucket/data/file-{uuid4()}.parquet"
)

# Commit to Iceberg table (ACID transaction)
table.append(data_file)
```

**Integration with Snowflake**:
```sql
-- Create Iceberg external table in Snowflake
CREATE EXTERNAL TABLE events.raw_events
  USING TEMPLATE (
    LOCATION = 's3://data-lake/warehouse/events/raw_events'
    FILE_FORMAT = (TYPE = PARQUET)
  )
  CATALOG = 'ICEBERG_CATALOG'
  EXTERNAL_VOLUME = 'iceberg_volume'
  AUTO_REFRESH = TRUE;
```

## Data Models

### Event Data Model

```rust
// Rust structures
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Event {
    pub event_id: String,
    pub user_id: String,
    pub event_type: String,
    pub event_timestamp: i64,
    pub event_value: Option<f64>,
    pub properties: HashMap<String, Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TransformedEvent {
    pub event_id: String,
    pub user_id: String,
    pub event_type: String,
    pub event_timestamp: i64,
    pub event_value: f64,
    pub event_date: String,
    pub event_hour: i32,
    pub properties_json: String,
}
```

### Job Tracking Model

```python
class Job(BaseModel):
    job_id: str
    status: JobStatus  # PENDING, PROCESSING, COMPLETED, FAILED
    created_at: datetime
    updated_at: datetime
    record_count: Optional[int]
    error_message: Optional[str]
```

### Dimensional Model (Snowflake)

```sql
-- Dimension: Users (SCD Type 2)
CREATE TABLE dim_users (
    user_sk NUMBER AUTOINCREMENT PRIMARY KEY,
    user_id STRING NOT NULL,
    user_name STRING,
    email STRING,
    created_at TIMESTAMP,
    valid_from TIMESTAMP NOT NULL,
    valid_to TIMESTAMP,
    is_current BOOLEAN NOT NULL,
    UNIQUE (user_id, valid_from)
);

-- Dimension: Products
CREATE TABLE dim_products (
    product_sk NUMBER AUTOINCREMENT PRIMARY KEY,
    product_id STRING NOT NULL UNIQUE,
    product_name STRING,
    category STRING,
    price NUMBER(10,2)
);

-- Fact: Events
CREATE TABLE fact_events (
    event_sk NUMBER AUTOINCREMENT PRIMARY KEY,
    user_sk NUMBER NOT NULL,
    product_sk NUMBER,
    event_timestamp TIMESTAMP NOT NULL,
    event_type STRING NOT NULL,
    event_value NUMBER(10,2),
    event_date DATE NOT NULL,
    event_hour NUMBER(2),
    FOREIGN KEY (user_sk) REFERENCES dim_users(user_sk),
    FOREIGN KEY (product_sk) REFERENCES dim_products(product_sk)
);

-- Partitioning for performance
ALTER TABLE fact_events ADD CLUSTERING BY (event_date, event_type);
```


## Correctness Properties

*A property is a characteristic or behavior that should hold true across all valid executions of a system-essentially, a formal statement about what the system should do. Properties serve as the bridge between human-readable specifications and machine-verifiable correctness guarantees.*

### Property 1: Valid ingestion returns Job ID
*For any* valid data payload submitted to the ingestion endpoint, the FastAPI Service should return a unique Job ID.
**Validates: Requirements 1.1**

### Property 2: Ingested data appears in Kinesis
*For any* data submitted to the FastAPI Service, that data should appear in the Kinesis Stream.
**Validates: Requirements 1.2**

### Property 3: Job status lookup returns status
*For any* Job ID created by the system, querying the status endpoint should return the current status of that job.
**Validates: Requirements 1.3**

### Property 4: Malformed data is rejected
*For any* malformed or invalid data payload, the FastAPI Service should reject the request and return an error message.
**Validates: Requirements 1.4**

### Property 5: Schema validation detects all violations
*For any* data with known schema violations, the Rust Processor should identify all violations when validating against the schema.
**Validates: Requirements 2.2**

### Property 6: Transformations applied to all events
*For any* batch of events and transformation rules, the Rust Processor should apply all transformation rules to each event in the batch.
**Validates: Requirements 2.3**

### Property 7: Iceberg Parquet round-trip preserves data
*For any* valid event data, writing to Iceberg table (Parquet format) and then reading back should produce equivalent data.
**Validates: Requirements 2.4**

### Property 8: Invalid data returns error details
*For any* invalid data provided to the Rust Processor, the processor should return detailed error information describing the problem.
**Validates: Requirements 2.5**

### Property 9: Python-Rust parsing interface works
*For any* byte data passed from Python to Rust parsing functions, the Rust Processor should return parsed events to Python.
**Validates: Requirements 3.2**

### Property 10: Python-Rust validation interface works
*For any* data passed from Python to Rust validation functions, the Rust Processor should return validation results to Python.
**Validates: Requirements 3.3**

### Property 11: Python-Rust transformation interface works
*For any* events passed from Python to Rust transformation functions, the Rust Processor should return transformed data to Python.
**Validates: Requirements 3.4**

### Property 12: Rust errors propagate to Python
*For any* error raised in Rust functions, the error should propagate to Python as a Python exception with error details.
**Validates: Requirements 3.5**

### Property 13: Processed batches written to Iceberg tables
*For any* batch processed by the Data Pipeline System, the results should be written to Iceberg tables in Parquet format.
**Validates: Requirements 4.2**

### Property 14: Iceberg commits update catalog metadata
*For any* data written to Iceberg tables, the Iceberg Catalog should be updated with new snapshot and partition information.
**Validates: Requirements 4.3**

### Property 15: Processing maintains checkpoints
*For any* records processed by the Data Pipeline System, checkpoints should be maintained to track processing progress.
**Validates: Requirements 4.4**

### Property 16: Failed batches retry correctly
*For any* batch that fails processing, the Data Pipeline System should retry processing up to 3 times before marking as failed.
**Validates: Requirements 4.5**

### Property 17: New Iceberg snapshots trigger workflow
*For any* new Iceberg snapshot detected, the Orchestrator should trigger the transformation workflow.
**Validates: Requirements 5.1**

### Property 18: Task failures execute retry logic
*For any* workflow task that fails, the Orchestrator should execute the configured retry logic.
**Validates: Requirements 5.2**

### Property 19: Successful workflows marked complete
*For any* workflow where all tasks complete successfully, the Orchestrator should mark the workflow as successful.
**Validates: Requirements 5.3**

### Property 20: Workflow executions are logged
*For any* workflow executed by the Orchestrator, all task executions and outcomes should be logged.
**Validates: Requirements 5.4**

### Property 21: Manual triggers start workflows
*For any* manual trigger request, the Orchestrator should start the specified workflow immediately.
**Validates: Requirements 5.5**

### Property 22: Raw data transforms to staging
*For any* raw data processed by dbt Models, staging tables should be created with cleaned and standardized data.
**Validates: Requirements 6.1**

### Property 23: Staging data transforms to intermediate
*For any* staging data processed by dbt Models, intermediate tables should be created with business logic applied.
**Validates: Requirements 6.2**

### Property 24: Intermediate data transforms to dimensional
*For any* intermediate data processed by dbt Models, dimensional tables should be created in star schema format.
**Validates: Requirements 6.3**

### Property 25: Dimensions include SCD Type 2 fields
*For any* dimension table created by dbt Models, the table should include valid_from, valid_to, and is_current fields for slowly changing dimension logic.
**Validates: Requirements 6.4**

### Property 26: Fact tables include foreign keys
*For any* fact table created by dbt Models, the table should include foreign keys to dimension tables.
**Validates: Requirements 6.5**

### Property 27: Null checks detect missing values
*For any* dataset validated by the Data Quality Framework, null values in required fields should be detected and reported.
**Validates: Requirements 7.1**

### Property 28: Type checks detect mismatches
*For any* dataset validated by the Data Quality Framework, data type mismatches should be detected and reported.
**Validates: Requirements 7.2**

### Property 29: Range checks detect out-of-bounds values
*For any* dataset validated by the Data Quality Framework, numeric values outside expected ranges should be detected and reported.
**Validates: Requirements 7.3**

### Property 30: Validation failures generate reports
*For any* validation failures detected by the Data Quality Framework, a detailed validation report should be generated.
**Validates: Requirements 7.4**

### Property 31: Critical failures trigger alerts
*For any* critical validation check failure, the Data Quality Framework should send alerts to configured notification channels.
**Validates: Requirements 7.5**

### Property 32: Dashboard queries use dimensional model
*For any* query executed by the Analytics Dashboard, data should be retrieved from the Dimensional Model.
**Validates: Requirements 8.2**

### Property 33: Dashboard displays quality metrics
*For any* validation results from the Data Quality Framework, the Analytics Dashboard should display the quality metrics.
**Validates: Requirements 8.3**

### Property 34: Filters update visualizations
*For any* filter applied in the Analytics Dashboard, visualizations should update to reflect the filtered data.
**Validates: Requirements 8.4**

### Property 35: Time series shows user timezone
*For any* time series data displayed in the Analytics Dashboard, timestamps should be shown in the user's timezone.
**Validates: Requirements 8.5**

### Property 36: Iceberg tables partitioned by date
*For any* data written to Iceberg tables, the data should be partitioned by date using Iceberg's hidden partitioning.
**Validates: Requirements 9.4**

### Property 37: Incremental models process only changes
*For any* incremental update processed by dbt Models, only new or changed records should be processed.
**Validates: Requirements 9.5**

### Property 38: Errors logged with details
*For any* error encountered by any component, the error should be logged with timestamp, component name, and error details.
**Validates: Requirements 10.1**

### Property 39: Parse failures continue processing
*For any* parse failure in the Rust Processor, the invalid data should be logged and processing should continue with remaining records.
**Validates: Requirements 10.2**

### Property 40: Task failures logged and retried
*For any* task failure detected by the Orchestrator, the failure should be logged and retry logic should execute.
**Validates: Requirements 10.3**

### Property 41: Quality failures logged with details
*For any* data quality check failure, the Data Quality Framework should log which records failed and which validation rules were violated.
**Validates: Requirements 10.4**

### Property 42: Iceberg commits create audit logs
*For any* data written to Iceberg tables, the Data Pipeline System should log the snapshot ID, data file paths, and record count for audit purposes.
**Validates: Requirements 10.5**

### Property 43: End-to-end pipeline correctness
*For any* input data flowing through the complete pipeline, the Data Pipeline System should produce correct results at each stage.
**Validates: Requirements 11.5**

### Property 44: JSON parsing round-trip
*For any* valid JSON data, parsing the JSON and then serializing back to JSON should produce equivalent data.
**Validates: Requirements 12.1**

### Property 45: CSV parsing round-trip
*For any* valid CSV data, parsing the CSV and then serializing back to CSV should produce equivalent data.
**Validates: Requirements 12.2**

### Property 46: Iceberg table reading round-trip
*For any* valid Iceberg table, reading the data and then writing back to a new Iceberg table should produce equivalent data.
**Validates: Requirements 12.3**

### Property 47: Iceberg Parquet files use compression
*For any* data written to Iceberg tables, the Parquet files should have compression enabled (Snappy or ZSTD).
**Validates: Requirements 12.4**

### Property 48: Unsupported formats return errors
*For any* unsupported data format provided to the Rust Processor, the processor should return an error indicating the unsupported format.
**Validates: Requirements 12.5**

## Error Handling

### Error Categories

**1. Input Validation Errors**
- Malformed JSON/CSV data
- Schema validation failures
- Missing required fields
- Invalid data types

**Strategy**: Return detailed error messages to clients, log errors, do not retry.

**2. Processing Errors**
- Rust processing failures
- Transformation errors
- Deduplication issues

**Strategy**: Log invalid records, continue processing remaining records, alert on high error rates.

**3. Infrastructure Errors**
- Kinesis connection failures
- S3 write failures
- Iceberg commit failures
- Snowflake connection issues
- Iceberg Catalog errors

**Strategy**: Retry with exponential backoff (up to 3 attempts), log failures, alert on persistent failures. Iceberg's ACID guarantees ensure no partial writes on failure.

**4. Orchestration Errors**
- Airflow task failures
- dbt model failures
- Sensor timeouts

**Strategy**: Airflow retry logic, send notifications, maintain workflow state for recovery.

**5. Data Quality Errors**
- Validation failures
- Expectation violations
- Referential integrity issues

**Strategy**: Generate detailed reports, alert on critical failures, allow workflow to continue for non-critical issues.

### Error Handling Patterns

```python
# FastAPI error handling
@app.exception_handler(ValidationError)
async def validation_exception_handler(request: Request, exc: ValidationError):
    logger.error(f"Validation error: {exc}")
    return JSONResponse(
        status_code=400,
        content={"error": "Validation failed", "details": str(exc)}
    )

# Rust error handling
pub fn parse_json(data: &[u8]) -> Result<Vec<Event>, ParseError> {
    match serde_json::from_slice(data) {
        Ok(events) => Ok(events),
        Err(e) => Err(ParseError::InvalidJson(e.to_string()))
    }
}

# Kinesis consumer error handling
def process_batch(self, records: List[Record]) -> ProcessingResult:
    results = []
    errors = []
    for record in records:
        try:
            result = self.process_record(record)
            results.append(result)
        except Exception as e:
            logger.error(f"Failed to process record: {e}")
            errors.append((record, e))
    return ProcessingResult(results=results, errors=errors)

# Airflow retry configuration
task = PythonOperator(
    task_id='process_data',
    python_callable=process_data,
    retries=3,
    retry_delay=timedelta(minutes=5),
    retry_exponential_backoff=True,
    max_retry_delay=timedelta(hours=1)
)
```

### Monitoring and Alerting

**Metrics to Monitor**:
- API error rates
- Processing throughput
- Kinesis lag
- Iceberg commit failures
- Iceberg snapshot count and size
- dbt test failures
- Data quality scores
- Pipeline latency

**Alert Conditions**:
- Error rate > 5% for 5 minutes
- Kinesis lag > 1 minute
- Iceberg commit failures > 3 consecutive attempts
- Critical data quality checks fail
- Pipeline latency > 10 minutes
- Disk space < 10%
- Iceberg table metadata size > 100MB (indicates need for maintenance)

## Testing Strategy

### Dual Testing Approach

This project employs both unit testing and property-based testing to ensure comprehensive correctness:

- **Unit tests** verify specific examples, edge cases, and integration points
- **Property tests** verify universal properties that should hold across all inputs
- Together they provide comprehensive coverage: unit tests catch concrete bugs, property tests verify general correctness

### Unit Testing

**Rust Unit Tests** (using cargo test):
- Test individual parsing functions with specific examples
- Test schema validation with known valid/invalid data
- Test transformation logic with edge cases (empty strings, null values, boundary values)
- Test error handling with specific error conditions
- Test Parquet writing with sample data

**Python Unit Tests** (using pytest):
- Test FastAPI endpoints with specific requests
- Test Kinesis producer/consumer with mock streams
- Test job tracking with specific job states
- Test integration between Python and Rust components
- Test error propagation across language boundaries

**dbt Tests** (using dbt test):
- Test unique constraints on dimension keys
- Test not-null constraints on required fields
- Test referential integrity between facts and dimensions
- Test custom business logic with specific examples

### Property-Based Testing

**Framework**: Use `proptest` for Rust and `hypothesis` for Python

**Configuration**: Each property-based test should run a minimum of 100 iterations to ensure thorough coverage of the input space.

**Tagging Convention**: Each property-based test MUST be tagged with a comment explicitly referencing the correctness property in this design document using the format: `**Feature: rust-data-pipeline, Property {number}: {property_text}**`

**Key Property Tests**:

**Rust Property Tests**:
```rust
// **Feature: rust-data-pipeline, Property 7: Iceberg Parquet round-trip preserves data**
#[test]
fn prop_iceberg_parquet_roundtrip() {
    proptest!(|(events: Vec<Event>)| {
        let table_location = "s3://test-bucket/test-table";
        let schema = create_test_iceberg_schema();
        let data_file = write_iceberg_parquet(&events, table_location, &schema).unwrap();
        let read_events = read_iceberg_parquet(&data_file.file_path).unwrap();
        assert_eq!(events, read_events);
    });
}

// **Feature: rust-data-pipeline, Property 5: Schema validation detects all violations**
#[test]
fn prop_schema_validation_detects_violations() {
    proptest!(|(event: Event, violations: Vec<Violation>)| {
        let invalid_event = inject_violations(event, violations.clone());
        let result = validate_schema(&invalid_event, &schema);
        assert!(result.is_err());
        let detected = result.unwrap_err().violations;
        assert_eq!(detected.len(), violations.len());
    });
}

// **Feature: rust-data-pipeline, Property 44: JSON parsing round-trip**
#[test]
fn prop_json_roundtrip() {
    proptest!(|(events: Vec<Event>)| {
        let json = serialize_json(&events).unwrap();
        let parsed = parse_json(&json).unwrap();
        assert_eq!(events, parsed);
    });
}
```

**Python Property Tests**:
```python
# **Feature: rust-data-pipeline, Property 1: Valid ingestion returns Job ID**
@given(st.dictionaries(st.text(), st.text()))
def test_prop_valid_ingestion_returns_job_id(data):
    response = client.post("/api/v1/ingest", json=data)
    assert response.status_code == 200
    assert "job_id" in response.json()
    assert len(response.json()["job_id"]) > 0

# **Feature: rust-data-pipeline, Property 4: Malformed data is rejected**
@given(st.binary())
def test_prop_malformed_data_rejected(malformed_data):
    assume(not is_valid_json(malformed_data))
    response = client.post("/api/v1/ingest", data=malformed_data)
    assert response.status_code == 400
    assert "error" in response.json()

# **Feature: rust-data-pipeline, Property 12: Rust errors propagate to Python**
@given(st.binary())
def test_prop_rust_errors_propagate(invalid_data):
    assume(not is_valid_format(invalid_data))
    with pytest.raises(Exception) as exc_info:
        data_processor_rs.parse_json(invalid_data)
    assert exc_info.value is not None
```

**Integration Property Tests**:
```python
# **Feature: rust-data-pipeline, Property 43: End-to-end pipeline correctness**
@given(st.lists(st.dictionaries(st.text(), st.text()), min_size=1, max_size=100))
def test_prop_end_to_end_pipeline(input_data):
    # Submit data to API
    response = client.post("/api/v1/ingest", json=input_data)
    job_id = response.json()["job_id"]
    
    # Wait for processing
    wait_for_job_completion(job_id)
    
    # Verify data in Snowflake
    result = query_snowflake(f"SELECT COUNT(*) FROM fact_events WHERE job_id = '{job_id}'")
    assert result[0][0] == len(input_data)
```

### Test Organization

```
tests/
├── rust/
│   ├── unit/
│   │   ├── test_parsing.rs
│   │   ├── test_validation.rs
│   │   └── test_transformation.rs
│   └── property/
│       ├── test_roundtrip.rs
│       ├── test_validation_props.rs
│       └── test_error_handling.rs
├── python/
│   ├── unit/
│   │   ├── test_api.py
│   │   ├── test_kinesis.py
│   │   └── test_integration.py
│   └── property/
│       ├── test_api_props.py
│       ├── test_pipeline_props.py
│       └── test_error_props.py
└── integration/
    ├── test_end_to_end.py
    └── test_data_quality.py
```

### Performance Testing

**Load Testing** (using Locust):
- Test API throughput (target: 100 req/s)
- Test concurrent batch processing
- Test Snowflake query performance

**Benchmark Testing**:
- Compare Rust vs Python processing speed
- Measure end-to-end pipeline latency
- Document performance improvements

## Deployment Architecture

### Infrastructure Components

**AWS Services**:
- **ECS**: Host FastAPI service containers
- **Kinesis**: Data streaming
- **S3**: Data lake storage
- **Glue**: Data catalog and crawlers
- **CloudWatch**: Logging and monitoring
- **IAM**: Access control

**External Services**:
- **Snowflake**: Data warehouse
- **Airflow**: Workflow orchestration (AWS MWAA or self-hosted)

### Containerization

**FastAPI Service Dockerfile**:
```dockerfile
FROM python:3.11-slim

# Install Rust processor wheel
COPY dist/data_processor_rs-*.whl /tmp/
RUN pip install /tmp/data_processor_rs-*.whl

# Install Python dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy application code
COPY app/ /app/
WORKDIR /app

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

**Airflow Dockerfile**:
```dockerfile
FROM apache/airflow:2.7.0-python3.11

# Install additional dependencies
COPY requirements-airflow.txt .
RUN pip install -r requirements-airflow.txt

# Copy DAGs
COPY dags/ ${AIRFLOW_HOME}/dags/
```

### CI/CD Pipeline

**GitHub Actions Workflow**:
```yaml
name: CI/CD Pipeline

on: [push, pull_request]

jobs:
  test-rust:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - uses: actions-rs/toolchain@v1
        with:
          toolchain: stable
      - run: cargo test
      - run: cargo build --release
  
  test-python:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - uses: actions/setup-python@v2
        with:
          python-version: '3.11'
      - run: pip install -r requirements-dev.txt
      - run: pytest tests/
  
  build-and-deploy:
    needs: [test-rust, test-python]
    runs-on: ubuntu-latest
    steps:
      - name: Build Rust wheel
        run: maturin build --release
      - name: Build Docker images
        run: docker build -t data-pipeline-api .
      - name: Push to ECR
        run: |
          aws ecr get-login-password | docker login --username AWS --password-stdin $ECR_REGISTRY
          docker push $ECR_REGISTRY/data-pipeline-api:latest
      - name: Deploy to ECS
        run: aws ecs update-service --cluster prod --service api --force-new-deployment
```

### Infrastructure as Code

**Terraform** will be used to provision:
- VPC and networking
- ECS cluster and services
- Kinesis streams
- S3 buckets with lifecycle policies
- Iceberg Catalog (AWS Glue or REST Catalog)
- IAM roles and policies for Iceberg operations
- CloudWatch alarms

## Security Considerations

- **API Authentication**: Use API keys or OAuth2 for FastAPI endpoints
- **AWS IAM**: Principle of least privilege for all service roles
- **Data Encryption**: Enable encryption at rest (S3, Snowflake) and in transit (TLS)
- **Secrets Management**: Use AWS Secrets Manager for credentials
- **Network Security**: VPC isolation, security groups, private subnets
- **Audit Logging**: CloudTrail for AWS API calls, application logs for data access

## Scalability Considerations

- **Horizontal Scaling**: ECS auto-scaling for FastAPI service
- **Kinesis Sharding**: Increase shards based on throughput
- **Iceberg Partitioning**: Hidden partitioning by date and event type with automatic partition pruning
- **Iceberg Compaction**: Scheduled compaction jobs to optimize file sizes and reduce metadata overhead
- **Snowflake Scaling**: Use auto-scaling warehouses with Iceberg external tables
- **Airflow Workers**: Scale worker count based on DAG load
- **Rust Processing**: Parallel batch processing with thread pools
- **Iceberg Snapshot Expiration**: Automatic cleanup of old snapshots to manage storage costs

## Future Enhancements

- Real-time streaming analytics with Kinesis Analytics
- Machine learning model integration for anomaly detection
- Multi-region deployment for disaster recovery with Iceberg multi-catalog support
- GraphQL API for flexible data querying
- Advanced data lineage tracking with Apache Atlas and Iceberg metadata
- Cost optimization with S3 Intelligent-Tiering and Iceberg file compaction
- Iceberg branching and tagging for experimentation and rollback
- Integration with Apache Spark for large-scale batch processing on Iceberg tables
- Iceberg merge-on-read for faster updates and deletes
