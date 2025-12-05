# Design Document: Modern Data Lakehouse Architecture

## Overview

This design document describes a modern data lakehouse architecture built on Databricks that implements the medallion pattern with three progressive data refinement layers (bronze, silver, gold). The system ingests streaming data from Kafka, processes it through Delta Lake tables with ACID guarantees, and exposes the data through a FastAPI service. Unity Catalog provides comprehensive governance including metadata management, access control, and data lineage tracking.

The architecture demonstrates key lakehouse capabilities: unified batch and streaming processing, time travel queries, performance optimization through Z-ordering and file compaction, and enterprise-grade data governance. The design emphasizes scalability, reliability, and maintainability while showcasing Delta Lake's advanced features.

## Architecture

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         Data Sources                                 │
│                         (Kafka Topics)                               │
└────────────────────────────┬────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    Spark Structured Streaming                        │
│                    (Ingestion Pipeline)                              │
└────────────────────────────┬────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      Bronze Layer (Raw)                              │
│                      Delta Lake Tables                               │
│                      - Append-only writes                            │
│                      - Full audit trail                              │
└────────────────────────────┬────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    Spark Batch/Stream Processing                     │
│                    (Validation & Cleaning)                           │
└────────────────────────────┬────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    Silver Layer (Cleaned)                            │
│                    Delta Lake Tables                                 │
│                    - Schema enforcement                              │
│                    - Deduplication                                   │
└────────────────────────────┬────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    Spark Batch Processing                            │
│                    (Aggregation & Modeling)                          │
└────────────────────────────┬────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    Gold Layer (Curated)                              │
│                    Delta Lake Tables                                 │
│                    - Business aggregates                             │
│                    - Dimensional models                              │
│                    - Z-ordered for queries                           │
└────────────────────────────┬────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│                         FastAPI Service                              │
│                         (Query Interface)                            │
└─────────────────────────────────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│                         Unity Catalog                                │
│                    (Governance & Lineage)                            │
└─────────────────────────────────────────────────────────────────────┘
```

### Technology Stack

- **Platform**: Databricks on AWS/Azure
- **Storage Format**: Delta Lake 3.0+
- **Processing Engine**: Apache Spark 3.5+
- **Streaming Source**: Apache Kafka
- **Stream Processing**: Spark Structured Streaming
- **API Framework**: FastAPI with Uvicorn
- **Governance**: Unity Catalog
- **Orchestration**: Databricks Workflows
- **Language**: Python 3.10+

## Components and Interfaces

### 1. Bronze Layer Ingestion Component

**Purpose**: Ingest raw streaming data from Kafka into Delta Lake with minimal transformation.

**Interfaces**:
- **Input**: Kafka topics with JSON/Avro messages
- **Output**: Delta Lake tables in bronze layer path (`/bronze/{topic_name}`)

**Key Functions**:
```python
def create_bronze_stream(
    kafka_bootstrap_servers: str,
    topic: str,
    checkpoint_location: str,
    output_path: str
) -> StreamingQuery
```

**Responsibilities**:
- Connect to Kafka and consume messages
- Add ingestion metadata (timestamp, source topic)
- Write to Delta Lake in append-only mode
- Maintain checkpoints for exactly-once semantics
- Handle schema evolution gracefully

### 2. Silver Layer Transformation Component

**Purpose**: Validate, clean, and deduplicate bronze data into structured silver tables.

**Interfaces**:
- **Input**: Bronze Delta Lake tables
- **Output**: Silver Delta Lake tables (`/silver/{entity_name}`)

**Key Functions**:
```python
def transform_to_silver(
    bronze_path: str,
    silver_path: str,
    schema: StructType,
    business_keys: List[str],
    validation_rules: Dict[str, Callable]
) -> DataFrame
```

**Responsibilities**:
- Validate records against schema
- Apply data type conversions
- Deduplicate based on business keys
- Enforce NOT NULL constraints
- Track data quality metrics
- Reject invalid records with logging

### 3. Gold Layer Aggregation Component

**Purpose**: Create business-level aggregations and dimensional models optimized for analytics.

**Interfaces**:
- **Input**: Silver Delta Lake tables
- **Output**: Gold Delta Lake tables (`/gold/{model_name}`)

**Key Functions**:
```python
def create_gold_aggregate(
    silver_path: str,
    gold_path: str,
    aggregation_logic: Callable,
    partition_cols: List[str],
    zorder_cols: List[str]
) -> None

def create_scd_type2_dimension(
    silver_path: str,
    dimension_path: str,
    business_key: str,
    tracked_columns: List[str]
) -> None
```

**Responsibilities**:
- Compute business aggregations
- Implement SCD Type 2 for dimensions
- Apply incremental processing
- Optimize with Z-ordering
- Update table statistics

### 4. Delta Lake Management Component

**Purpose**: Provide time travel, optimization, and maintenance operations for Delta tables.

**Interfaces**:
- **Input**: Delta table paths and operation parameters
- **Output**: Optimized Delta tables with version history

**Key Functions**:
```python
def query_time_travel(
    table_path: str,
    version: Optional[int] = None,
    timestamp: Optional[str] = None
) -> DataFrame

def optimize_table(
    table_path: str,
    zorder_columns: Optional[List[str]] = None
) -> None

def vacuum_table(
    table_path: str,
    retention_hours: int = 168
) -> None
```

**Responsibilities**:
- Execute time travel queries
- Compact small files with OPTIMIZE
- Apply Z-ORDER clustering
- Clean up old versions with VACUUM
- Maintain transaction log

### 5. FastAPI Query Service

**Purpose**: Expose Delta Lake data through REST API with authentication and authorization.

**Interfaces**:
- **Input**: HTTP requests with query parameters
- **Output**: JSON responses with query results

**API Endpoints**:
```python
@app.get("/api/v1/query")
async def query_table(
    table: str,
    filters: Optional[Dict[str, Any]] = None,
    page: int = 1,
    page_size: int = 100,
    user: User = Depends(get_current_user)
) -> QueryResponse

@app.get("/api/v1/time-travel")
async def query_time_travel(
    table: str,
    version: Optional[int] = None,
    timestamp: Optional[str] = None,
    user: User = Depends(get_current_user)
) -> QueryResponse
```

**Responsibilities**:
- Authenticate requests
- Enforce row-level security
- Execute queries with filters
- Implement pagination
- Handle query timeouts
- Return formatted results

### 6. Unity Catalog Integration Component

**Purpose**: Manage metadata, enforce access control, and track data lineage.

**Interfaces**:
- **Input**: Table registration requests, access control policies
- **Output**: Governed tables with lineage tracking

**Key Functions**:
```python
def register_table(
    catalog: str,
    schema: str,
    table_name: str,
    table_path: str,
    table_schema: StructType
) -> None

def set_access_policy(
    table_fqn: str,
    policy: AccessPolicy
) -> None

def capture_lineage(
    source_tables: List[str],
    target_table: str,
    transformation_logic: str
) -> None
```

**Responsibilities**:
- Register tables in Unity Catalog
- Enforce access control policies
- Capture lineage relationships
- Log audit events
- Apply governance policies

### 7. Monitoring and Observability Component

**Purpose**: Track pipeline health, performance metrics, and resource utilization.

**Interfaces**:
- **Input**: Spark metrics, Delta Lake metrics, API metrics
- **Output**: Dashboards, alerts, logs

**Key Functions**:
```python
def track_streaming_metrics(
    stream_query: StreamingQuery
) -> StreamingMetrics

def track_batch_metrics(
    job_id: str,
    start_time: datetime,
    end_time: datetime,
    rows_processed: int
) -> BatchMetrics

def check_performance_thresholds(
    metrics: Metrics,
    thresholds: Dict[str, float]
) -> List[Alert]
```

**Responsibilities**:
- Monitor streaming job health
- Track batch job performance
- Measure query response times
- Alert on threshold violations
- Track cost metrics

## Data Models

### Bronze Layer Schema

```python
bronze_schema = StructType([
    StructField("raw_data", StringType(), nullable=False),  # Original message
    StructField("ingestion_timestamp", TimestampType(), nullable=False),
    StructField("source_topic", StringType(), nullable=False),
    StructField("kafka_partition", IntegerType(), nullable=False),
    StructField("kafka_offset", LongType(), nullable=False),
    StructField("kafka_timestamp", TimestampType(), nullable=True)
])
```

### Silver Layer Schema (Example: User Events)

```python
silver_user_events_schema = StructType([
    StructField("event_id", StringType(), nullable=False),
    StructField("user_id", StringType(), nullable=False),
    StructField("event_type", StringType(), nullable=False),
    StructField("event_timestamp", TimestampType(), nullable=False),
    StructField("properties", MapType(StringType(), StringType()), nullable=True),
    StructField("processed_timestamp", TimestampType(), nullable=False),
    StructField("data_quality_score", DoubleType(), nullable=False)
])
```

### Gold Layer Schema (Example: Daily User Aggregates)

```python
gold_daily_user_aggregates_schema = StructType([
    StructField("date", DateType(), nullable=False),
    StructField("user_id", StringType(), nullable=False),
    StructField("event_count", LongType(), nullable=False),
    StructField("unique_event_types", IntegerType(), nullable=False),
    StructField("first_event_timestamp", TimestampType(), nullable=False),
    StructField("last_event_timestamp", TimestampType(), nullable=False),
    StructField("aggregation_timestamp", TimestampType(), nullable=False)
])
```

### SCD Type 2 Dimension Schema (Example: User Dimension)

```python
dim_user_scd2_schema = StructType([
    StructField("user_key", LongType(), nullable=False),  # Surrogate key
    StructField("user_id", StringType(), nullable=False),  # Business key
    StructField("user_name", StringType(), nullable=True),
    StructField("user_email", StringType(), nullable=True),
    StructField("user_status", StringType(), nullable=False),
    StructField("effective_date", DateType(), nullable=False),
    StructField("end_date", DateType(), nullable=True),
    StructField("is_current", BooleanType(), nullable=False)
])
```

### API Response Models

```python
class QueryResponse(BaseModel):
    data: List[Dict[str, Any]]
    total_rows: int
    page: int
    page_size: int
    execution_time_ms: float
    
class TimeTravel QueryResponse(BaseModel):
    data: List[Dict[str, Any]]
    version: Optional[int]
    timestamp: Optional[str]
    execution_time_ms: float
```


## Correctness Properties

*A property is a characteristic or behavior that should hold true across all valid executions of a system—essentially, a formal statement about what the system should do. Properties serve as the bridge between human-readable specifications and machine-verifiable correctness guarantees.*

### Bronze Layer Properties

**Property 1: Append-only persistence**
*For any* streaming message from Kafka, when ingested into the bronze layer, the message should appear in the Delta table and no existing records should be deleted or modified.
**Validates: Requirements 1.1**

**Property 2: Field preservation (Round-trip)**
*For any* message with arbitrary fields, when ingested into bronze and then read back, all source fields should be preserved without transformation.
**Validates: Requirements 1.2**

**Property 3: Metadata completeness**
*For any* record written to the bronze layer, the record should include an ingestion timestamp in the metadata.
**Validates: Requirements 1.3**

**Property 4: Error resilience**
*For any* batch of messages containing both valid and malformed messages, when processed by the bronze layer, errors should be logged and subsequent valid messages should still be processed successfully.
**Validates: Requirements 1.4**

**Property 5: Exactly-once semantics (Idempotence)**
*For any* batch of messages, when processed multiple times with checkpointing enabled, the total record count in the bronze table should remain consistent (no duplicates created).
**Validates: Requirements 1.5**

### Silver Layer Properties

**Property 6: Schema validation**
*For any* set of bronze records containing both schema-compliant and non-compliant records, when processed to silver, only the schema-compliant records should appear in the silver table.
**Validates: Requirements 2.1**

**Property 7: Deduplication by business key**
*For any* set of records with duplicate business keys but different timestamps, when processed to silver, only the record with the most recent timestamp should be retained.
**Validates: Requirements 2.2**

**Property 8: Type conversion error handling**
*For any* record with invalid data types for required fields, when processed to silver, the record should be rejected and a validation failure should be logged.
**Validates: Requirements 2.3**

**Property 9: NOT NULL constraint enforcement**
*For any* record with null values in required fields, when processed to silver, the record should be rejected.
**Validates: Requirements 2.4**

**Property 10: Quality metrics generation**
*For any* silver layer processing run, when processing completes, data quality metrics including validation pass rate should be recorded.
**Validates: Requirements 2.5**

### Gold Layer Properties

**Property 11: Date partitioning**
*For any* silver data processed to gold, the resulting gold table should be partitioned by date.
**Validates: Requirements 3.1**

**Property 12: SCD Type 2 versioning**
*For any* sequence of changes to a dimensional entity, when processed to gold, all versions should be preserved with proper effective dates, end dates, and is_current flags.
**Validates: Requirements 3.2**

**Property 13: Z-ORDER application**
*For any* gold table created with specified Z-ORDER columns, the table metadata should reflect that Z-ORDER clustering has been applied to those columns.
**Validates: Requirements 3.3**

**Property 14: Incremental processing**
*For any* gold table with existing data, when new silver data is added and processing runs, only the new or changed data should be processed (not reprocessing all historical data).
**Validates: Requirements 3.4**

**Property 15: Statistics update**
*For any* gold table write operation, when the write completes, table statistics should be updated.
**Validates: Requirements 3.5**

### Delta Lake Time Travel Properties

**Property 16: Time travel correctness**
*For any* Delta table with multiple versions, when querying by version number or timestamp, the returned data should match the data as it existed at that version or point in time.
**Validates: Requirements 4.1, 4.2**

**Property 17: Version availability within retention**
*For any* Delta table with multiple versions within the retention period, all versions should be queryable via time travel.
**Validates: Requirements 4.4**

**Property 18: VACUUM cleanup**
*For any* Delta table with versions older than the specified retention period, when VACUUM is executed, those old versions should be removed and no longer queryable.
**Validates: Requirements 4.5**

### FastAPI Service Properties

**Property 19: Query execution**
*For any* valid table name and filter parameters, when a query request is sent to the API, the query should execute against the specified Delta table and return matching results.
**Validates: Requirements 5.1**

**Property 20: Pagination for large results**
*For any* query that returns more than 10000 rows, the API response should implement pagination with the configured page size.
**Validates: Requirements 5.2**

**Property 21: Authentication rejection**
*For any* API request with invalid authentication credentials, the API should return HTTP 401 status and reject the request.
**Validates: Requirements 5.3**

**Property 22: Row-level security enforcement**
*For any* user with specific permissions, when executing queries, only rows that the user is authorized to access should be returned.
**Validates: Requirements 5.4**

**Property 23: Query timeout**
*For any* query that exceeds 30 seconds execution time, the API should timeout and return an error response.
**Validates: Requirements 5.5**

### Unity Catalog Properties

**Property 24: Table registration**
*For any* table created in bronze, silver, or gold layers, the table metadata including schema and location should be registered in Unity Catalog.
**Validates: Requirements 6.1**

**Property 25: Access control enforcement**
*For any* user attempting to access a table, Unity Catalog should enforce the configured access control policies and deny access if the user lacks permissions.
**Validates: Requirements 6.2**

**Property 26: Lineage capture**
*For any* data transformation from source tables to target table, Unity Catalog should capture the lineage relationship between them.
**Validates: Requirements 6.3**

**Property 27: Policy propagation**
*For any* governance policy update, the changes should be immediately effective on all affected tables.
**Validates: Requirements 6.4**

**Property 28: Audit logging**
*For any* audit event (table access, modification, etc.), Unity Catalog should log the event with user, action, and timestamp details.
**Validates: Requirements 6.5**

### Delta Lake Optimization Properties

**Property 29: File compaction**
*For any* Delta table with many small files, when OPTIMIZE is executed, the file count should be reduced through compaction into larger files.
**Validates: Requirements 7.1**

**Property 30: Z-ORDER reorganization**
*For any* Delta table, when Z-ORDER is applied with specified columns, the data should be reorganized to colocate related values (verifiable through file statistics).
**Validates: Requirements 7.2**

**Property 31: Version preservation during optimization**
*For any* Delta table with multiple versions, when OPTIMIZE is executed, all versions within the retention period should remain accessible via time travel.
**Validates: Requirements 7.3**

**Property 32: VACUUM retention enforcement**
*For any* Delta table, when VACUUM is executed with a specified retention period, only files older than that retention period should be removed.
**Validates: Requirements 7.5**

### Unified Processing Properties

**Property 33: Batch-streaming consistency**
*For any* transformation logic, when executed in both batch mode and streaming mode on the same input data, the results should be consistent.
**Validates: Requirements 8.1, 8.2, 8.3**

**Property 34: Incremental processing efficiency**
*For any* dataset that has been processed, when new records are added and incremental processing runs, only the new or changed records should be processed.
**Validates: Requirements 8.4**

**Property 35: Watermark management**
*For any* streaming processing run, when processing completes, watermarks should be updated to reflect the processed data boundaries.
**Validates: Requirements 8.5**

### Monitoring Properties

**Property 36: Metrics tracking**
*For any* streaming or batch job execution, the monitoring system should track relevant metrics including processing rate, latency, execution time, rows processed, and error counts.
**Validates: Requirements 9.1, 9.2**

**Property 37: Performance alerting**
*For any* query with response time exceeding configured thresholds, the monitoring system should trigger an alert to administrators.
**Validates: Requirements 9.3**

**Property 38: Delta Lake metrics capture**
*For any* Delta Lake operation, the monitoring system should capture metrics including file count, table size, and version count.
**Validates: Requirements 9.4**

**Property 39: Cost threshold alerting**
*For any* scenario where cost thresholds are exceeded, the monitoring system should send notifications to designated administrators.
**Validates: Requirements 9.5**

### ACID Transaction Properties

**Property 40: Concurrent write serialization**
*For any* set of concurrent write operations to the same Delta table, the writes should be serialized to prevent conflicts and maintain consistency.
**Validates: Requirements 10.1**

**Property 41: Transaction atomicity**
*For any* write operation that fails midway, no partial data should be committed to the Delta table (all-or-nothing).
**Validates: Requirements 10.2**

**Property 42: Snapshot isolation**
*For any* concurrent read and write operations on a Delta table, readers should see a consistent snapshot of data without seeing partial writes.
**Validates: Requirements 10.3**

**Property 43: Transaction log atomicity**
*For any* completed transaction, the Delta Lake transaction log should be updated atomically.
**Validates: Requirements 10.4**

**Property 44: Conflict retry with backoff**
*For any* concurrent operations that conflict, the system should retry the operation with exponential backoff up to the maximum retry limit.
**Validates: Requirements 10.5**

## Error Handling

### Bronze Layer Error Handling

- **Kafka Connection Failures**: Implement retry logic with exponential backoff; alert after max retries exceeded
- **Schema Evolution**: Handle new fields gracefully by storing raw JSON; log schema changes
- **Checkpoint Corruption**: Detect corrupted checkpoints and restart from last valid checkpoint
- **Write Failures**: Log failures with full context; implement dead letter queue for problematic messages

### Silver Layer Error Handling

- **Validation Failures**: Log rejected records with validation reasons; maintain rejection metrics
- **Type Conversion Errors**: Capture conversion errors with source values; provide data quality reports
- **Deduplication Conflicts**: Log when business key conflicts occur; track deduplication statistics
- **Schema Mismatches**: Reject records with clear error messages; alert on schema drift

### Gold Layer Error Handling

- **Aggregation Errors**: Validate aggregation inputs; log errors with source data references
- **SCD Type 2 Conflicts**: Handle overlapping effective dates; ensure referential integrity
- **Optimization Failures**: Retry OPTIMIZE operations; alert if repeated failures occur
- **Incremental Processing Errors**: Fall back to full refresh if incremental logic fails

### API Error Handling

- **Authentication Failures**: Return 401 with clear error messages; log authentication attempts
- **Authorization Failures**: Return 403 with minimal information; log unauthorized access attempts
- **Query Timeouts**: Cancel long-running queries; return 504 with timeout information
- **Invalid Parameters**: Validate inputs; return 400 with detailed validation errors
- **Internal Errors**: Return 500 with request ID; log full stack traces for debugging

### Unity Catalog Error Handling

- **Registration Failures**: Retry registration; alert if catalog is unavailable
- **Policy Conflicts**: Validate policies before application; provide conflict resolution guidance
- **Lineage Capture Failures**: Log lineage errors; continue processing (lineage is non-blocking)
- **Audit Log Failures**: Buffer audit events; retry with backoff; alert on persistent failures

## Testing Strategy

### Unit Testing Approach

The system will use **pytest** as the primary testing framework for Python components. Unit tests will focus on:

- **Component Isolation**: Test individual functions and classes in isolation
- **Edge Cases**: Empty inputs, boundary values, null handling
- **Error Conditions**: Invalid inputs, missing dependencies, timeout scenarios
- **Mock External Dependencies**: Mock Kafka, Spark, Delta Lake for fast unit tests
- **Integration Points**: Test interfaces between components

**Example Unit Tests**:
- Test bronze ingestion with empty Kafka messages
- Test silver validation with null values in required fields
- Test gold aggregation with zero records
- Test API authentication with expired tokens
- Test Unity Catalog registration with invalid table names

### Property-Based Testing Approach

The system will use **Hypothesis** (Python) as the property-based testing library. Property-based tests will verify universal properties across randomly generated inputs.

**Configuration**:
- Each property-based test will run a minimum of **100 iterations**
- Tests will use custom strategies for generating valid Spark DataFrames, Delta tables, and API requests
- Each test will be tagged with a comment referencing the correctness property from this design document

**Tagging Format**:
```python
# Feature: data-lakehouse, Property 1: Append-only persistence
@given(kafka_messages=st.lists(st.text()))
def test_bronze_append_only(kafka_messages):
    # Test implementation
```

**Property Test Coverage**:
- Each of the 44 correctness properties listed above will be implemented as a property-based test
- Tests will generate random data matching the expected schemas
- Tests will verify properties hold across diverse inputs
- Tests will use Spark's testing utilities for DataFrame comparisons

**Custom Strategies**:
```python
# Strategy for generating valid Spark rows
@st.composite
def spark_row_strategy(draw):
    return Row(
        event_id=draw(st.uuids()),
        user_id=draw(st.text(min_size=1)),
        event_type=draw(st.sampled_from(['click', 'view', 'purchase'])),
        event_timestamp=draw(st.datetimes())
    )

# Strategy for generating Delta table paths
@st.composite
def delta_table_path_strategy(draw):
    layer = draw(st.sampled_from(['bronze', 'silver', 'gold']))
    table_name = draw(st.text(alphabet=st.characters(whitelist_categories=('Ll', 'Nd')), min_size=1))
    return f"/{layer}/{table_name}"
```

### Integration Testing

- **End-to-End Pipeline Tests**: Test data flow from Kafka through all layers to API
- **Multi-Component Tests**: Test interactions between Spark, Delta Lake, and Unity Catalog
- **Performance Tests**: Verify query performance meets SLA requirements
- **Concurrency Tests**: Test concurrent reads/writes to validate ACID properties

### Test Environment

- **Local Development**: Use Docker Compose for Kafka, MinIO (S3-compatible), and Spark
- **CI/CD**: Use GitHub Actions with Databricks Connect for integration tests
- **Test Data**: Generate synthetic data with realistic distributions
- **Cleanup**: Ensure tests clean up Delta tables and checkpoints after execution

