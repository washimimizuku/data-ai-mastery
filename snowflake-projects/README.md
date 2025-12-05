# Snowflake Platform Projects

Platform-specific projects demonstrating Snowflake capabilities including Data Sharing, Snowpipe Streaming, and Snowpark ML. These projects require a Snowflake account (trial available).

## Projects Overview

### Project S1: Enterprise Data Sharing
**Time**: 3-4 days (24-32h) | **Focus**: Secure data sharing and collaboration

Implement enterprise data sharing with secure shares, reader accounts, data marketplace integration, and cross-cloud sharing.

**Stack**: Snowflake, SQL, Python, Secure Shares, Data Marketplace

**Key Features**:
- Secure data sharing (no data movement)
- Reader accounts for external consumers
- Row-level security and masking
- Data marketplace publishing
- Cross-cloud and cross-region sharing

---

### Project S2: Real-Time CDC with Snowpipe Streaming
**Time**: 3-4 days (24-32h) | **Focus**: Real-time data ingestion and CDC

Build real-time CDC pipeline using Snowpipe Streaming with change tracking, Streams, Tasks, and SCD Type 2 implementation.

**Stack**: Snowflake, Snowpipe Streaming, Streams, Tasks, Python, Kafka

**Key Features**:
- Snowpipe Streaming for real-time ingestion
- Change tracking with Streams
- Automated processing with Tasks
- SCD Type 2 for historical tracking
- Monitoring and error handling

---

### Project S3: Snowpark ML Platform
**Time**: 3-4 days (24-32h) | **Focus**: ML in Snowflake with Snowpark

Implement end-to-end ML workflow using Snowpark for Python, including feature engineering, model training, UDFs, and deployment.

**Stack**: Snowflake, Snowpark, Python, scikit-learn, XGBoost, UDFs

**Key Features**:
- Snowpark DataFrame API for transformations
- Feature engineering in Snowflake
- Model training with Snowpark ML
- Vectorized UDFs for inference
- Model deployment and monitoring

---

## Recommended Learning Path

### Sequential Path (9-12 days)
1. **Project S1** - Data Sharing (3-4 days)
   - Foundation for Snowflake governance
   - Understanding data sharing model

2. **Project S2** - CDC Streaming (3-4 days)
   - Real-time data ingestion
   - Streams and Tasks automation

3. **Project S3** - Snowpark ML (3-4 days)
   - ML in Snowflake
   - Complete ML lifecycle

**Total**: 9-12 days (72-96 hours)

---

## Technology Coverage

| Project | Secure Shares | Snowpipe Streaming | Streams | Tasks | Snowpark | ML |
|---------|--------------|-------------------|---------|-------|----------|-----|
| S1. Data Sharing | ✅✅ | - | - | - | - | - |
| S2. CDC Streaming | - | ✅✅ | ✅✅ | ✅✅ | ✅ | - |
| S3. Snowpark ML | - | - | - | ✅ | ✅✅ | ✅✅ |

---

## Prerequisites

### Required Knowledge
- SQL fundamentals (strong)
- Python programming
- Data engineering basics
- ML fundamentals (for S3)

### Recommended Bootcamps
- [100 Days Data & AI](https://github.com/washimimizuku/100-days-data-ai) - Days 1-50
- [30 Days of Snowflake](https://github.com/washimimizuku/30-days-snowflake-data-ai) - Complete

### Snowflake Account
- **Trial** (30 days, $400 credit) - Recommended
- **Paid** (production use)

Sign up: https://signup.snowflake.com/

---

## Key Features

**All projects include**:
- Step-by-step Snowflake UI navigation
- SQL and Python code examples
- Snowflake worksheet implementations
- Architecture diagrams
- Best practices and patterns
- Cost optimization tips

**All projects demonstrate**:
- Platform-specific features
- Production-ready patterns
- Security and governance
- Performance optimization
- Monitoring and observability

---

## Data Sources

**See [DATA_SOURCES.md](./DATA_SOURCES.md) for comprehensive data source guide.**

### Quick Reference

| Project | Data Sources | Size | Notes |
|---------|-------------|------|-------|
| S1. Data Sharing | Sample datasets, CSV uploads | 1-10GB | Multi-domain data |
| S2. CDC Streaming | Kafka, real-time events | Streaming | CDC, logs, IoT |
| S3. Snowpark ML | ML datasets (tabular) | 10-50GB | Training data |

### Snowflake Sample Data
```sql
-- Available in all Snowflake accounts
USE DATABASE SNOWFLAKE_SAMPLE_DATA;

-- Common samples
-- TPCH_SF1: TPC-H benchmark data
-- TPCDS_SF10TCL: TPC-DS benchmark data
-- WEATHER: Weather data
```

---

## Cost Considerations

### Snowflake Pricing
- **Trial**: $400 credit (30 days)
- **Standard**: $2/credit (on-demand)
- **Enterprise**: $3/credit (on-demand)
- **Business Critical**: $4/credit (on-demand)

### Credit Consumption
- **Compute**: 1-128 credits/hour (warehouse size)
- **Storage**: $40/TB/month (compressed)
- **Data Transfer**: $0.01-0.02/GB (cross-region)

### Estimated Costs (Trial/Paid)
- **S1 Data Sharing**: $20-40 (mostly compute)
- **S2 CDC Streaming**: $30-50 (streaming costs)
- **S3 Snowpark ML**: $40-60 (compute for ML)
- **Total**: $90-150 for all 3 projects

### Cost Optimization Tips
1. Use **trial credits** for learning (30 days free)
2. **Suspend warehouses** when not in use (auto-suspend)
3. Use **X-Small warehouses** for development
4. **Right-size warehouses** (don't over-provision)
5. Use **query result caching** (free)
6. **Cluster keys** for large tables
7. Set **auto-suspend** to 5-10 minutes
8. Use **resource monitors** to prevent overspend

---

## Snowflake Account Setup

### 1. Create Account
```bash
# Sign up at https://signup.snowflake.com/
# Choose cloud provider (AWS, Azure, GCP)
# Select region closest to you
# Get $400 trial credit (30 days)
```

### 2. Create Warehouse
```sql
-- Create compute warehouse
CREATE WAREHOUSE learning_wh
  WITH WAREHOUSE_SIZE = 'X-SMALL'
  AUTO_SUSPEND = 300  -- 5 minutes
  AUTO_RESUME = TRUE
  INITIALLY_SUSPENDED = TRUE;

-- Use warehouse
USE WAREHOUSE learning_wh;
```

### 3. Create Database and Schema
```sql
-- Create database
CREATE DATABASE learning_db;

-- Create schema
CREATE SCHEMA learning_db.projects;

-- Use schema
USE SCHEMA learning_db.projects;
```

### 4. Install Snowpark (for S2, S3)
```bash
pip install snowflake-snowpark-python[pandas]
pip install snowflake-ml-python
```

---

## Project Structure

Each project folder contains:
- `prd.md` - Product requirements and goals
- `tech-spec.md` - Technical implementation details
- `implementation-plan.md` - Day-by-day guide
- `README.md` - Overview and getting started

---

## Learning Outcomes

### After S1 (Data Sharing)
- Implement secure data sharing
- Create and manage reader accounts
- Configure row-level security
- Publish to data marketplace
- Manage cross-cloud sharing

### After S2 (CDC Streaming)
- Ingest real-time data with Snowpipe Streaming
- Track changes with Streams
- Automate processing with Tasks
- Implement SCD Type 2
- Monitor streaming pipelines

### After S3 (Snowpark ML)
- Use Snowpark DataFrame API
- Engineer features in Snowflake
- Train models with Snowpark ML
- Deploy models as UDFs
- Monitor model performance

---

## Integration with Other Projects

### With Core Projects
- Use **core-data-engineering** projects as foundation
- Apply **core-ai-development** ML skills

### With Advanced Projects
- **Project 3 (Lakehouse)**: Compare with Snowflake architecture
- **Project 4 (MLOps)**: ML deployment patterns
- **Project 9 (Full Platform)**: Snowflake as data warehouse

### With Databricks Projects
- Compare governance approaches (Snowflake vs Unity Catalog)
- Evaluate platform trade-offs
- Understand multi-cloud strategies

---

## Why Snowflake?

### Strengths
1. **Zero management** - Fully managed, no infrastructure
2. **Separation of storage and compute** - Scale independently
3. **Data sharing** - Share without copying data
4. **Multi-cloud** - AWS, Azure, GCP
5. **SQL-first** - Familiar SQL interface
6. **Performance** - Automatic optimization

### Use Cases
- Data warehousing and analytics
- Data sharing and collaboration
- Real-time data ingestion
- SQL-based ML (Snowpark)
- Cross-cloud data access

---

## Comparison: Snowflake vs Databricks

| Feature | Snowflake | Databricks |
|---------|-----------|------------|
| **Focus** | Data warehousing | Data + AI |
| **Processing** | SQL-based | Spark-based |
| **ML Support** | Limited (Snowpark ML) | Native (MLflow) |
| **Streaming** | Good (Snowpipe Streaming) | Strong (Delta Live Tables) |
| **Notebooks** | External (Hex, Jupyter) | Built-in |
| **Governance** | Native governance | Unity Catalog |
| **Cost Model** | Compute + storage | DBU + compute |
| **Best For** | SQL analytics, BI | ML/AI, complex ETL |

**Recommendation**: Learn both platforms to maximize career opportunities.

---

## Success Criteria

### Technical Milestones
- [ ] Secure data shares with external consumers
- [ ] Real-time CDC pipeline with Snowpipe Streaming
- [ ] Snowpark ML model training and deployment
- [ ] Automated Tasks for data processing
- [ ] Monitoring dashboards for all projects

### Portfolio Deliverables
- [ ] Snowflake worksheets with documentation
- [ ] Architecture diagrams
- [ ] SQL and Python code examples
- [ ] Performance benchmarks
- [ ] Cost analysis report
- [ ] Screenshots/videos of Snowflake UI

---

## Getting Started

### Quick Start (Trial)
1. Sign up for Snowflake trial ($400 credit)
2. Create warehouse and database
3. Load sample data
4. Start with Project S1

### Step-by-Step
1. **Week 1**: Set up Snowflake, complete S1
2. **Week 2**: Build CDC pipeline (S2)
3. **Week 3**: Implement ML workflow (S3)
4. **Week 4**: Documentation and portfolio

---

## Additional Resources

### Official Documentation
- [Snowflake Documentation](https://docs.snowflake.com/)
- [Data Sharing Guide](https://docs.snowflake.com/en/user-guide/data-sharing-intro)
- [Snowpipe Streaming](https://docs.snowflake.com/en/user-guide/data-load-snowpipe-streaming-overview)
- [Snowpark Guide](https://docs.snowflake.com/en/developer-guide/snowpark/index)
- [Snowpark ML](https://docs.snowflake.com/en/developer-guide/snowpark-ml/index)

### Learning Resources
- [Snowflake University](https://learn.snowflake.com/) - Free courses
- [Snowflake Community](https://community.snowflake.com/) - Forums
- [Snowflake Blog](https://www.snowflake.com/blog/) - Best practices

### Certifications
- **SnowPro Core Certification**
- **SnowPro Advanced: Data Engineer**
- **SnowPro Advanced: Data Scientist**

---

## Snowflake-Specific Features

### 1. Zero-Copy Cloning
```sql
-- Clone database instantly (no data copy)
CREATE DATABASE dev_db CLONE prod_db;

-- Clone table
CREATE TABLE test_table CLONE prod_table;
```

### 2. Time Travel
```sql
-- Query historical data (up to 90 days)
SELECT * FROM table AT(OFFSET => -3600);  -- 1 hour ago
SELECT * FROM table BEFORE(STATEMENT => 'query_id');
```

### 3. Result Caching
```sql
-- Automatic result caching (24 hours)
-- Repeated queries return cached results (free)
SELECT * FROM large_table WHERE date = '2024-01-01';
```

### 4. Multi-Cluster Warehouses
```sql
-- Auto-scale for concurrency
CREATE WAREHOUSE analytics_wh
  WITH WAREHOUSE_SIZE = 'MEDIUM'
  MIN_CLUSTER_COUNT = 1
  MAX_CLUSTER_COUNT = 10
  SCALING_POLICY = 'STANDARD';
```

---

## Next Steps

1. **Set up Snowflake account** (30-day trial)
2. **Complete 30 Days of Snowflake** bootcamp (if not done)
3. **Start with Project S1** (Data Sharing)
4. **Progress to S2 and S3** sequentially
5. **Document everything** for your portfolio
6. **Consider certification** to validate skills

These projects complement the Databricks projects and demonstrate platform versatility for your portfolio.
