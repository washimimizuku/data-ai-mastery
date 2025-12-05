# Project D2: Delta Live Tables Medallion Pipeline

## Overview

Build a production-grade data pipeline using Delta Live Tables (DLT), demonstrating medallion architecture, data quality expectations, and liquid clustering.

**What You'll Build**: An end-to-end medallion pipeline (Bronze/Silver/Gold) with DLT, automated data quality checks, incremental processing, and performance optimization.

**What You'll Learn**: Delta Live Tables declarative syntax, medallion architecture patterns, data quality expectations, liquid clustering, and DLT monitoring.

## Time Estimate

**3-4 days (24-32 hours)**

### Day 1: Bronze Layer (8 hours)
- Hours 1-2: DLT pipeline creation
- Hours 3-4: Auto Loader setup
- Hours 5-6: Streaming ingestion
- Hours 7-8: Testing and validation

### Day 2: Silver Layer (8 hours)
- Hours 1-3: Data quality expectations
- Hours 4-5: Transformations and enrichment
- Hours 6-7: Deduplication logic
- Hour 8: Quarantine tables

### Day 3: Gold Layer (8 hours)
- Hours 1-3: Business aggregations
- Hours 4-5: Materialized views
- Hours 6-7: Change Data Feed
- Hour 8: Testing

### Day 4: Optimization (6-8 hours)
- Hours 1-2: Liquid clustering
- Hours 3-4: Pipeline monitoring
- Hours 5-6: Performance tuning
- Hours 7-8: Documentation

## Prerequisites

### Required Knowledge
- [30 Days of Databricks](https://github.com/washimimizuku/30-days-databricks-data-ai) - Days 1-30
  - Days 6-10: Delta Lake basics
  - Days 16-20: Delta Live Tables
  - Days 21-25: Data quality
- [60 Days Advanced](https://github.com/washimimizuku/60-days-advanced-data-ai) - Days 15-24
  - Data orchestration and quality patterns

### Technical Requirements
- Databricks workspace with DLT enabled
- Understanding of medallion architecture
- Python and SQL knowledge
- Delta Lake concepts

### Databricks Setup
- Premium or Enterprise tier workspace
- Cluster with DBR 11.3+ (for liquid clustering)
- Cloud storage configured (S3/ADLS/GCS)
- Sample data source (files or streaming)

## Getting Started

### Step 1: Review Documentation
1. `prd.md` - Product requirements and goals
2. `tech-spec.md` - Technical architecture and code examples
3. `implementation-plan.md` - Day-by-day implementation guide

### Step 2: Create DLT Pipeline (UI)

**Navigate to Workflows:**
1. Click **Workflows** in left sidebar
2. Click **Delta Live Tables** tab
3. Click **Create Pipeline**

**Configure Pipeline:**
1. **Pipeline Name**: `medallion_pipeline`
2. **Product Edition**: Advanced (for expectations)
3. **Pipeline Mode**: Triggered (or Continuous for streaming)
4. **Source Code**: 
   - Click **Add notebook**
   - Select or create notebooks for Bronze/Silver/Gold layers
5. **Storage Location**: `/mnt/dlt/medallion` (or DBFS path)
6. **Target Schema**: `medallion_db`
7. **Cluster Configuration**:
   - Mode: Enhanced Autoscaling
   - Min workers: 1
   - Max workers: 5
   - Photon: Enabled
8. Click **Create**

### Step 3: Prepare Sample Data

```python
# Generate sample orders data
from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from datetime import datetime, timedelta
import random

spark = SparkSession.builder.getOrCreate()

# Generate sample data
data = []
for i in range(10000):
    data.append({
        "order_id": i,
        "customer_id": random.randint(1, 1000),
        "amount": round(random.uniform(10, 1000), 2),
        "status": random.choice(["pending", "completed", "cancelled"]),
        "order_date": (datetime.now() - timedelta(days=random.randint(0, 365))).isoformat(),
        "region": random.choice(["US", "EU", "ASIA"])
    })

df = spark.createDataFrame(data)

# Write to cloud storage
df.write.mode("overwrite").format("json").save("/mnt/raw/orders/")

print("✓ Sample data generated at /mnt/raw/orders/")
```

### Step 4: Create Pipeline Notebooks

Create three notebooks in your workspace:
- `bronze_layer` - Raw data ingestion
- `silver_layer` - Cleaned and validated data
- `gold_layer` - Business aggregations

## Core Implementation

### 1. Bronze Layer - Raw Ingestion

**Create Notebook: `bronze_layer`**

```python
import dlt
from pyspark.sql.functions import *

# Streaming ingestion with Auto Loader
@dlt.table(
    name="bronze_orders",
    comment="Raw orders data ingested from cloud storage",
    table_properties={
        "quality": "bronze",
        "pipelines.autoOptimize.zOrderCols": "order_id"
    }
)
def bronze_orders():
    return (
        spark.readStream
            .format("cloudFiles")
            .option("cloudFiles.format", "json")
            .option("cloudFiles.schemaLocation", "/mnt/schema/orders")
            .option("cloudFiles.inferColumnTypes", "true")
            .load("/mnt/raw/orders/")
            .withColumn("ingestion_timestamp", current_timestamp())
            .withColumn("source_file", input_file_name())
    )

# Batch ingestion for reference data
@dlt.table(
    name="bronze_customers",
    comment="Raw customer reference data"
)
def bronze_customers():
    return (
        spark.read
            .format("parquet")
            .load("/mnt/raw/customers/")
            .withColumn("ingestion_timestamp", current_timestamp())
    )
```

**Add Notebook to Pipeline:**
1. Go to your DLT pipeline
2. Click **Settings**
3. Under **Source Code**, click **Add**
4. Select `bronze_layer` notebook
5. Click **Save**

### 2. Silver Layer - Data Quality

**Create Notebook: `silver_layer`**

```python
import dlt
from pyspark.sql.functions import *

# Silver orders with expectations
@dlt.table(
    name="silver_orders",
    comment="Cleaned and validated orders",
    table_properties={
        "quality": "silver",
        "delta.enableChangeDataFeed": "true"
    }
)
@dlt.expect_or_drop("valid_order_id", "order_id IS NOT NULL")
@dlt.expect_or_drop("valid_amount", "amount > 0 AND amount < 100000")
@dlt.expect_or_fail("valid_status", "status IN ('pending', 'completed', 'cancelled')")
@dlt.expect("valid_customer", "customer_id IS NOT NULL")
def silver_orders():
    return (
        dlt.read_stream("bronze_orders")
            .select(
                col("order_id").cast("long"),
                col("customer_id").cast("long"),
                col("amount").cast("decimal(10,2)"),
                col("status"),
                to_timestamp(col("order_date")).alias("order_date"),
                col("region"),
                col("ingestion_timestamp")
            )
            .dropDuplicates(["order_id"])
    )

# Enriched orders with customer data
@dlt.table(
    name="silver_orders_enriched",
    comment="Orders enriched with customer information"
)
def silver_orders_enriched():
    orders = dlt.read("silver_orders")
    customers = dlt.read("silver_customers")
    
    return (
        orders.join(customers, "customer_id", "left")
            .select(
                orders["*"],
                customers["customer_name"],
                customers["customer_segment"]
            )
    )

# Quarantine table for failed records
@dlt.table(
    name="quarantine_orders",
    comment="Orders that failed validation rules"
)
def quarantine_orders():
    return (
        dlt.read_stream("bronze_orders")
            .where(
                "order_id IS NULL OR " +
                "amount IS NULL OR amount <= 0 OR amount >= 100000 OR " +
                "status NOT IN ('pending', 'completed', 'cancelled')"
            )
            .withColumn("quarantine_reason", 
                when(col("order_id").isNull(), "missing_order_id")
                .when(col("amount").isNull(), "missing_amount")
                .when(col("amount") <= 0, "invalid_amount_negative")
                .when(col("amount") >= 100000, "invalid_amount_too_high")
                .otherwise("invalid_status")
            )
            .withColumn("quarantine_timestamp", current_timestamp())
    )
```

**Understanding Expectations:**
- `@dlt.expect()` - Track violations, don't drop records
- `@dlt.expect_or_drop()` - Drop records that fail
- `@dlt.expect_or_fail()` - Fail entire pipeline if violated

### 3. Gold Layer - Business Aggregations

**Create Notebook: `gold_layer`**

```python
import dlt
from pyspark.sql.functions import *

# Daily sales metrics
@dlt.table(
    name="gold_daily_sales",
    comment="Daily sales metrics by region",
    table_properties={
        "quality": "gold",
        "delta.enableLiquidClustering": "true",
        "delta.clusterBy": "date, region"
    }
)
def gold_daily_sales():
    return (
        dlt.read("silver_orders_enriched")
            .where("status = 'completed'")
            .groupBy(
                to_date("order_date").alias("date"),
                "region"
            )
            .agg(
                count("order_id").alias("order_count"),
                sum("amount").alias("total_sales"),
                avg("amount").alias("avg_order_value"),
                countDistinct("customer_id").alias("unique_customers")
            )
    )

# Customer lifetime value
@dlt.table(
    name="gold_customer_ltv",
    comment="Customer lifetime value metrics"
)
def gold_customer_ltv():
    return (
        dlt.read("silver_orders_enriched")
            .where("status = 'completed'")
            .groupBy("customer_id", "customer_name", "customer_segment")
            .agg(
                count("order_id").alias("total_orders"),
                sum("amount").alias("lifetime_value"),
                min("order_date").alias("first_order_date"),
                max("order_date").alias("last_order_date"),
                avg("amount").alias("avg_order_value"),
                datediff(max("order_date"), min("order_date")).alias("customer_tenure_days")
            )
    )

# Real-time streaming aggregation
@dlt.table(
    name="gold_hourly_metrics",
    comment="Hourly streaming metrics"
)
def gold_hourly_metrics():
    return (
        dlt.read_stream("silver_orders")
            .withWatermark("order_date", "1 hour")
            .groupBy(
                window("order_date", "1 hour").alias("hour_window"),
                "region",
                "status"
            )
            .agg(
                count("order_id").alias("order_count"),
                sum("amount").alias("total_amount")
            )
            .select(
                col("hour_window.start").alias("hour_start"),
                col("hour_window.end").alias("hour_end"),
                "region",
                "status",
                "order_count",
                "total_amount"
            )
    )
```

### 4. Run and Monitor Pipeline

**Start Pipeline (UI):**
1. Go to **Workflows** → **Delta Live Tables**
2. Select your pipeline
3. Click **Start**
4. Watch the pipeline graph build

**Monitor Pipeline:**
1. **Pipeline Graph**: Visual representation of tables and dependencies
2. **Event Log**: Detailed execution logs
3. **Data Quality**: View expectation metrics
4. **Lineage**: See data flow between tables

**View Data Quality Metrics:**
```sql
-- Query expectation metrics
SELECT
    dataset,
    name as expectation_name,
    passed_records,
    failed_records,
    ROUND(passed_records * 100.0 / (passed_records + failed_records), 2) as pass_rate
FROM event_log(TABLE(LIVE.silver_orders))
WHERE details:flow_progress.metrics IS NOT NULL
ORDER BY failed_records DESC;
```

### 5. Liquid Clustering

**Enable Liquid Clustering:**
```python
# In table definition
@dlt.table(
    name="gold_orders_clustered",
    table_properties={
        "delta.enableLiquidClustering": "true",
        "delta.clusterBy": "customer_id, order_date"
    }
)
def gold_orders_clustered():
    return dlt.read("silver_orders_enriched")
```

**Benefits:**
- Automatic data organization
- No manual OPTIMIZE needed
- Better query performance
- Adapts to query patterns

**Compare with Z-Ordering:**
```sql
-- Legacy Z-ordering (manual)
OPTIMIZE medallion_db.silver_orders
ZORDER BY (customer_id, order_date);

-- Liquid clustering (automatic)
-- No manual optimization needed!
```

### 6. Change Data Feed

**Enable CDC:**
```python
@dlt.table(
    name="silver_customers",
    table_properties={
        "delta.enableChangeDataFeed": "true"
    }
)
def silver_customers():
    return dlt.read("bronze_customers")
```

**Consume Changes:**
```python
@dlt.table(name="customer_changes")
def customer_changes():
    return (
        spark.readStream
            .format("delta")
            .option("readChangeFeed", "true")
            .option("startingVersion", 0)
            .table("LIVE.silver_customers")
            .select(
                "*",
                "_change_type",  # insert, update, delete
                "_commit_version",
                "_commit_timestamp"
            )
    )
```

### 7. Pipeline Configuration

**Advanced Settings (UI):**
1. Go to pipeline **Settings**
2. **Configuration**:
   - Add key-value pairs for parameters
   - Example: `source_path=/mnt/raw/orders`
3. **Notifications**:
   - Email on success/failure
   - Webhook integration
4. **Photon Acceleration**: Enable for 2-3x performance
5. **Auto Scaling**: Set min/max workers

**Access Configuration in Code:**
```python
# Use pipeline parameters
source_path = spark.conf.get("source_path", "/mnt/raw/orders")

@dlt.table(name="bronze_orders")
def bronze_orders():
    return spark.readStream.format("cloudFiles").load(source_path)
```

## Success Criteria

- [ ] DLT pipeline created with Bronze/Silver/Gold layers
- [ ] Auto Loader ingesting data from cloud storage
- [ ] Data quality expectations defined (expect, expect_or_drop, expect_or_fail)
- [ ] Quarantine table capturing failed records
- [ ] Enrichment logic joining multiple tables
- [ ] Gold layer aggregations working
- [ ] Liquid clustering enabled on key tables
- [ ] Change Data Feed configured
- [ ] Pipeline monitoring dashboard reviewed
- [ ] Data quality metrics tracked
- [ ] Documentation complete with pipeline graph

## Cost Optimization

### Compute Costs
- Use **Photon** for 2-3x better price/performance
- Set **Auto-stop** to 30 minutes for development
- Use **Triggered** mode instead of Continuous for batch workloads
- Right-size cluster: Start with 1-2 workers, scale as needed

### Storage Costs
- Enable **Auto Optimize** to reduce small files
- Use **Liquid Clustering** instead of manual OPTIMIZE
- Set **VACUUM** retention to 7 days (default is 30)
- Monitor storage with:
  ```sql
  DESCRIBE DETAIL medallion_db.silver_orders;
  ```

### Pipeline Costs
- **Development Mode**: Cheaper, reuses clusters
- **Production Mode**: Dedicated clusters, higher cost
- Monitor DBU usage in **Account Console** → **Usage**

**Estimated Monthly Cost** (for this project):
- DLT Pipeline (Development, 2 workers): ~$100-200
- Storage (500 GB): ~$10-15
- **Total**: ~$110-215/month

## Common Challenges

### Pipeline Fails to Start
Check **Event Log** for errors, verify notebook paths and storage permissions

### Expectations Not Working
Ensure Advanced edition is selected, check expectation syntax

### Slow Performance
Enable Photon, increase workers, check for data skew

### Quarantine Table Empty
Verify expectation logic, check if all records pass validation

## Learning Outcomes

- Build declarative pipelines with Delta Live Tables
- Implement medallion architecture (Bronze/Silver/Gold)
- Define and enforce data quality expectations
- Use liquid clustering for query optimization
- Monitor pipeline health and data quality
- Design incremental processing with Change Data Feed
- Navigate DLT UI for pipeline management

## Next Steps

1. Add to portfolio with pipeline architecture diagram
2. Write blog post: "DLT vs Traditional Spark: A Comparison"
3. Continue to Project D3: Databricks MLOps
4. Extend with CDC integration using Debezium

## Resources

- [Delta Live Tables Docs](https://docs.databricks.com/delta-live-tables/index.html)
- [Expectations Guide](https://docs.databricks.com/delta-live-tables/expectations.html)
- [Liquid Clustering](https://docs.databricks.com/delta/clustering.html)
- [Auto Loader](https://docs.databricks.com/ingestion/auto-loader/index.html)
- [Medallion Architecture](https://www.databricks.com/glossary/medallion-architecture)
