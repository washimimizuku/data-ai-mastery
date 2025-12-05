# Project 2: High-Performance Data Pipeline with Rust + FastAPI

## Objective

Build a high-performance data pipeline using Rust for data processing, FastAPI for APIs, Apache Iceberg for storage, and Airflow + dbt for orchestration.

**What You'll Build**: A production pipeline with Rust parsers (10-50x faster than Python), FastAPI endpoints, Iceberg lakehouse, and orchestrated workflows.

**What You'll Learn**: Rust for data engineering, PyO3 bindings, Iceberg table format, Airflow orchestration, dbt transformations, and hybrid language architectures.

## Time Estimate

**2-3 months (160-240 hours)**

- Weeks 1-2: Rust parsers and PyO3 bindings (40-60h)
- Weeks 3-4: FastAPI services and APIs (40-60h)
- Weeks 5-6: Iceberg lakehouse setup (40-60h)
- Weeks 7-8: Airflow + dbt orchestration (40-60h)

## Prerequisites

### Required Knowledge
- [30 Days of Rust](https://github.com/washimimizuku/30-days-rust-data-ai) - Days 1-30
- [100 Days Data & AI](https://github.com/washimimizuku/100-days-data-ai) - Days 1-50
- [60 Days Advanced](https://github.com/washimimizuku/60-days-advanced-data-ai) - Days 12-21

### Technical Requirements
- Rust 1.70+, Python 3.9+, Docker, PostgreSQL, Airflow, dbt
- Understanding of systems programming
- Familiarity with lakehouse architectures

## Architecture Overview

### System Components

```
Raw Data → Rust Parsers → FastAPI → Iceberg Tables → dbt → Analytics
              ↓                         ↓
         PyO3 Bindings            Airflow DAGs
```

**Core Components:**
- **Rust Parsers**: High-performance data processing (CSV, JSON, Parquet)
- **PyO3 Layer**: Python bindings for Rust modules
- **FastAPI**: REST APIs for data ingestion and queries
- **Iceberg**: ACID lakehouse with time travel
- **Airflow**: Workflow orchestration
- **dbt**: SQL transformations and data modeling

### Technology Stack

**Processing Layer:**
- Rust 1.70+ (data parsing, transformations)
- PyO3 0.20+ (Python-Rust interop)
- Polars or DataFusion (Rust DataFrame libraries)

**API Layer:**
- FastAPI (async Python web framework)
- Pydantic (data validation)
- uvicorn (ASGI server)

**Storage Layer:**
- Apache Iceberg 1.4+ (table format)
- Parquet (file format)
- S3/MinIO (object storage)
- PostgreSQL (Iceberg catalog)

**Orchestration:**
- Apache Airflow 2.8+ (workflow management)
- dbt 1.7+ (SQL transformations)
- Great Expectations (data quality)

**Infrastructure:**
- Docker + Docker Compose
- Kubernetes (production)
- Terraform (IaC)

## Core Implementation

### 1. Rust Data Parsers

**High-Performance Parsing:**
- CSV parser with `csv` crate (10-20x faster than pandas)
- JSON parser with `serde_json` (15-30x faster)
- Parquet reader with `parquet` crate (5-10x faster)
- Memory-efficient streaming for large files
- Parallel processing with `rayon`

**PyO3 Bindings:**
- Expose Rust functions to Python
- Zero-copy data transfer where possible
- Python-friendly error handling
- Type conversions (Rust ↔ Python)

**Example Interface:**
```python
import rust_parsers

# Parse CSV with Rust (10x faster)
df = rust_parsers.parse_csv("large_file.csv")

# Parse JSON with Rust (20x faster)
records = rust_parsers.parse_json("events.json")
```

### 2. FastAPI Services

**API Endpoints:**
- POST `/ingest` - Upload and parse data files
- GET `/query/{table}` - Query Iceberg tables
- GET `/schema/{table}` - Get table schema
- POST `/transform` - Trigger dbt transformations
- GET `/health` - Service health check

**Features:**
- Async request handling
- Background tasks for long-running operations
- Request validation with Pydantic
- OpenAPI documentation
- Rate limiting and authentication

### 3. Apache Iceberg Lakehouse

**Table Format Features:**
- ACID transactions
- Schema evolution
- Time travel queries
- Partition evolution
- Hidden partitioning
- Snapshot isolation

**Catalog Options:**
- PostgreSQL catalog (metadata storage)
- AWS Glue catalog (cloud-native)
- Hive Metastore (legacy compatibility)

**Table Organization:**
- Bronze layer: Raw data (append-only)
- Silver layer: Cleaned data (upserts)
- Gold layer: Aggregated data (optimized)

### 4. Airflow Orchestration

**DAG Structure:**
- Data ingestion DAG (hourly)
- Transformation DAG (daily)
- Data quality DAG (post-transform)
- Maintenance DAG (weekly compaction)

**Task Dependencies:**
- Sensor tasks for file arrival
- Rust parser tasks (via BashOperator or PythonOperator)
- dbt run tasks
- Data quality checks
- Notifications on failure

### 5. dbt Transformations

**Model Layers:**
- Staging models: Raw → cleaned
- Intermediate models: Business logic
- Mart models: Analytics-ready tables

**Features:**
- Incremental models for efficiency
- Tests for data quality
- Documentation generation
- Lineage tracking

## Integration Points

### Rust → Python (PyO3)
- Compile Rust to Python extension module
- Install with `pip install .` or `maturin develop`
- Import and use like native Python library
- Performance: Near-native Rust speed from Python

### FastAPI → Iceberg
- Use `pyiceberg` library for table operations
- Read/write Parquet files to S3/MinIO
- Update Iceberg metadata in catalog
- Query with predicate pushdown

### Airflow → dbt
- Use `BashOperator` or `DbtRunOperator`
- Pass variables for incremental runs
- Handle failures with retries
- Log dbt output to Airflow logs

### dbt → Iceberg
- Use `dbt-iceberg` adapter
- Define models as Iceberg tables
- Leverage Iceberg features (time travel, schema evolution)
- Optimize with table maintenance

## Performance Targets

**Parsing Speed:**
- CSV: 1M rows/second (Rust) vs 50K rows/second (pandas)
- JSON: 500K records/second (Rust) vs 25K records/second (Python)
- Parquet: 2M rows/second (Rust) vs 200K rows/second (pandas)

**API Latency:**
- Ingestion: <500ms for 10MB file
- Query: <100ms for simple queries
- Transform trigger: <50ms response

**Storage Efficiency:**
- Parquet compression: 5-10x vs CSV
- Iceberg metadata overhead: <1% of data size
- Query performance: 10-100x faster with partitioning

## Success Criteria

- [ ] Rust parsers 10-50x faster than Python equivalents
- [ ] PyO3 bindings working seamlessly
- [ ] FastAPI endpoints deployed and documented
- [ ] Iceberg tables with Bronze/Silver/Gold layers
- [ ] Time travel queries working
- [ ] Airflow DAGs orchestrating end-to-end pipeline
- [ ] dbt models transforming data
- [ ] Data quality tests passing
- [ ] Performance benchmarks documented
- [ ] Monitoring and alerting configured

## Learning Outcomes

- Build high-performance data tools with Rust
- Create Python bindings with PyO3
- Design FastAPI microservices
- Implement Iceberg lakehouse architecture
- Orchestrate complex workflows with Airflow
- Model data transformations with dbt
- Optimize data pipelines for performance
- Compare Rust vs Python for data engineering

## Deployment Strategy

**Local Development:**
- Docker Compose for all services
- MinIO for S3-compatible storage
- PostgreSQL for Iceberg catalog
- Airflow with LocalExecutor

**Production:**
- Kubernetes for container orchestration
- AWS S3 for object storage
- RDS PostgreSQL for catalog
- Airflow with CeleryExecutor or KubernetesExecutor
- Horizontal scaling for FastAPI

## Next Steps

1. Add to portfolio with architecture diagrams
2. Write blog post: "Why Rust for Data Engineering?"
3. Continue to Project 3: Lakehouse Architecture
4. Extend with real-time streaming (Kafka + Rust)

## Resources

- [Rust Book](https://doc.rust-lang.org/book/)
- [PyO3 Docs](https://pyo3.rs/)
- [Apache Iceberg](https://iceberg.apache.org/)
- [Airflow Docs](https://airflow.apache.org/docs/)
- [dbt Docs](https://docs.getdbt.com/)
- [Polars](https://pola-rs.github.io/polars-book/)
