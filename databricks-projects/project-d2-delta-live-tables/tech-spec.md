# Technical Specification: Delta Live Tables Medallion Pipeline

## Architecture Overview

```
Raw Data Sources → Bronze (Raw) → Silver (Cleaned) → Gold (Aggregated)
                       ↓              ↓                  ↓
                   Streaming      Validation        Business Metrics
```

## Technology Stack

- **Platform**: Databricks with Delta Live Tables
- **Storage**: Delta Lake on S3/ADLS/GCS
- **Language**: Python, SQL
- **Orchestration**: DLT pipelines (native)
- **Monitoring**: DLT observability

## Detailed Design

### 1. Bronze Layer - Raw Ingestion

```python
import dlt
from pyspark.sql.functions import *

# Streaming ingestion from cloud storage
@dlt.table(
    name="bronze_orders",
    comment="Raw orders data from source systems",
    table_properties={"quality": "bronze"}
)
def bronze_orders():
    return (
        spark.readStream
            .format("cloudFiles")
            .option("cloudFiles.format", "json")
            .option("cloudFiles.schemaLocation", "/mnt/schema/orders")
            .load("/mnt/raw/orders/")
            .withColumn("ingestion_timestamp", current_timestamp())
            .withColumn("source_file", input_file_name())
    )

# Batch ingestion
@dlt.table(
    name="bronze_customers",
    comment="Raw customer data"
)
def bronze_customers():
    return spark.read.format("parquet").load("/mnt/raw/customers/")
```

### 2. Silver Layer - Cleaned & Validated

```python
# Data quality expectations
@dlt.table(
    name="silver_orders",
    comment="Cleaned and validated orders",
    table_properties={"quality": "silver"}
)
@dlt.expect_or_drop("valid_order_id", "order_id IS NOT NULL")
@dlt.expect_or_drop("valid_amount", "amount > 0")
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
                col("ingestion_timestamp")
            )
            .dropDuplicates(["order_id"])
    )

# Enrichment with dimension data
@dlt.table(
    name="silver_orders_enriched",
    comment="Orders enriched with customer data"
)
def silver_orders_enriched():
    orders = dlt.read("silver_orders")
    customers = dlt.read("silver_customers")
    
    return (
        orders.join(customers, "customer_id", "left")
            .select(
                orders["*"],
                customers["customer_name"],
                customers["customer_segment"],
                customers["region"]
            )
    )
```

### 3. Gold Layer - Business Aggregates

```python
# Materialized view for daily metrics
@dlt.table(
    name="gold_daily_sales",
    comment="Daily sales metrics by region"
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

# Streaming aggregation
@dlt.table(
    name="gold_customer_lifetime_value",
    comment="Customer lifetime value metrics"
)
def gold_customer_lifetime_value():
    return (
        dlt.read("silver_orders_enriched")
            .where("status = 'completed'")
            .groupBy("customer_id", "customer_name", "customer_segment")
            .agg(
                count("order_id").alias("total_orders"),
                sum("amount").alias("lifetime_value"),
                min("order_date").alias("first_order_date"),
                max("order_date").alias("last_order_date"),
                avg("amount").alias("avg_order_value")
            )
    )
```

### 4. Data Quality Framework

```python
# Quarantine table for failed records
@dlt.table(
    name="quarantine_orders",
    comment="Orders that failed validation"
)
@dlt.expect_all_or_drop({
    "has_order_id": "order_id IS NOT NULL",
    "has_amount": "amount IS NOT NULL"
})
def quarantine_orders():
    return (
        dlt.read_stream("bronze_orders")
            .where("order_id IS NULL OR amount IS NULL OR amount <= 0")
    )

# Data quality metrics
@dlt.table(
    name="data_quality_metrics"
)
def data_quality_metrics():
    return spark.sql("""
        SELECT
            'silver_orders' as table_name,
            COUNT(*) as total_records,
            SUM(CASE WHEN order_id IS NULL THEN 1 ELSE 0 END) as null_order_ids,
            SUM(CASE WHEN amount <= 0 THEN 1 ELSE 0 END) as invalid_amounts,
            current_timestamp() as measured_at
        FROM LIVE.silver_orders
    """)
```

### 5. Change Data Feed

```python
# Enable Change Data Feed
@dlt.table(
    name="silver_customers",
    table_properties={
        "delta.enableChangeDataFeed": "true"
    }
)
def silver_customers():
    return (
        dlt.read("bronze_customers")
            .select(
                col("customer_id").cast("long"),
                col("customer_name"),
                col("email"),
                col("customer_segment"),
                col("region")
            )
    )

# Consume changes
@dlt.table(
    name="customer_changes"
)
def customer_changes():
    return (
        spark.readStream
            .format("delta")
            .option("readChangeData", "true")
            .option("startingVersion", 0)
            .table("LIVE.silver_customers")
    )
```

### 6. Performance Optimization

```python
# Liquid clustering for better query performance
@dlt.table(
    name="gold_orders_by_customer",
    table_properties={
        "delta.enableLiquidClustering": "true",
        "delta.clusterBy": "customer_id, order_date"
    }
)
def gold_orders_by_customer():
    return dlt.read("silver_orders_enriched")

# Z-ordering for legacy tables
spark.sql("""
    OPTIMIZE LIVE.silver_orders
    ZORDER BY (customer_id, order_date)
""")
```

### 7. Pipeline Configuration

```json
{
  "id": "medallion_pipeline",
  "name": "Medallion Architecture Pipeline",
  "storage": "/mnt/dlt/medallion",
  "configuration": {
    "pipelines.applyChangesPreviewEnabled": "true"
  },
  "clusters": [
    {
      "label": "default",
      "autoscale": {
        "min_workers": 1,
        "max_workers": 5,
        "mode": "ENHANCED"
      }
    }
  ],
  "libraries": [
    {
      "notebook": {
        "path": "/Pipelines/bronze_layer"
      }
    },
    {
      "notebook": {
        "path": "/Pipelines/silver_layer"
      }
    },
    {
      "notebook": {
        "path": "/Pipelines/gold_layer"
      }
    }
  ],
  "target": "medallion_db",
  "continuous": false,
  "development": true
}
```

## Implementation Phases

### Phase 1: Bronze Layer (Day 1)
- Set up DLT pipeline
- Implement raw data ingestion
- Streaming and batch sources
- Test data flow

### Phase 2: Silver Layer (Day 2)
- Implement data quality expectations
- Create cleaned tables
- Enrichment logic
- Quarantine handling

### Phase 3: Gold Layer (Day 3)
- Business aggregations
- Materialized views
- Performance optimization
- Change Data Feed

### Phase 4: Documentation (Day 4)
- Architecture diagrams
- Migration guide
- Performance benchmarks
- Best practices

## Deliverables

### Code
- DLT pipeline notebooks (Bronze/Silver/Gold)
- Pipeline configuration JSON
- Data quality framework
- Monitoring queries

### Documentation
- Architecture diagram (medallion)
- "Lakehouse Migration Guide: Traditional DW → Databricks"
- DLT vs. Spark comparison
- Data quality framework guide
- Performance benchmarks

### Metrics
- Processing throughput (rows/minute)
- Latency measurements
- Data quality metrics
- Cost per pipeline run
