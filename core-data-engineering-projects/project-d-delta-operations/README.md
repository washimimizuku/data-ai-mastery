# Project D: Delta Lake Operations

## Objective

Build a comprehensive demonstration of Delta Lake operations, showcasing ACID transactions, time travel, merge operations, and optimization techniques (OPTIMIZE, VACUUM, Z-ORDER).

**What You'll Build**: A series of Jupyter notebooks demonstrating all Delta Lake capabilities including CRUD operations, time travel queries, upsert patterns, schema evolution, and performance optimization with real-world examples.

**What You'll Learn**: Delta Lake architecture, ACID guarantees in data lakes, transaction log mechanics, time travel implementation, merge/upsert patterns, and table optimization strategies for production workloads.

## Time Estimate

**1-2 days (8-16 hours)**

- Hours 1-2: Spark + Delta Lake setup and configuration
- Hours 3-4: Basic CRUD operations (Create, Read, Update, Delete)
- Hours 5-6: Time travel and version management
- Hours 7-8: Merge operations and upserts
- Hours 9-11: Optimization (OPTIMIZE, Z-ORDER, VACUUM)
- Hours 12-14: Advanced features (CDF, schema evolution, constraints)
- Hours 15-16: Performance benchmarking and documentation

## Prerequisites

### Required Knowledge
Complete these bootcamp sections first:
- [100 Days Data & AI](https://github.com/washimimizuku/100-days-data-ai) - Days 11-20
  - Days 11-15: Table formats (Delta Lake focus)
  - Days 16-20: ACID transactions and optimization
- [30 Days of Databricks](https://github.com/washimimizuku/30-days-databricks-data-ai) - Days 1-10 (optional)
  - Days 6-10: Delta Lake on Databricks

### Technical Requirements
- Python 3.11+ installed
- PySpark 3.5+ installed
- Understanding of Spark DataFrames
- Basic SQL knowledge
- Familiarity with Parquet format

### Tools Needed
- Python with pyspark, delta-spark
- Jupyter Lab for notebooks
- Git for version control
- 2-4 GB free disk space for data

## Getting Started

### Step 1: Review Documentation
Read the project documents in order:
1. `prd.md` - Understand what you're building
2. `tech-spec.md` - Review technical architecture
3. `implementation-plan.md` - Follow the implementation steps

### Step 2: Set Up Spark with Delta Lake
```bash
# Install dependencies
pip install pyspark==3.5.0 delta-spark==2.4.0 jupyterlab pandas

# Start Jupyter Lab
jupyter lab
```

### Step 3: Configure Spark Session
```python
from pyspark.sql import SparkSession
from delta import *

# Create Spark session with Delta Lake
builder = SparkSession.builder \
    .appName("DeltaLakeOperations") \
    .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension") \
    .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog") \
    .config("spark.jars.packages", "io.delta:delta-core_2.12:2.4.0")

spark = configure_spark_with_delta_pip(builder).getOrCreate()

# Verify Delta Lake is working
print(f"Spark version: {spark.version}")
print("Delta Lake configured successfully!")
```

### Step 4: Create Your First Delta Table
```python
from pyspark.sql.functions import current_timestamp

# Create sample data
data = [
    (1, "Alice", 30, "Engineering"),
    (2, "Bob", 25, "Sales"),
    (3, "Charlie", 35, "Engineering"),
    (4, "Diana", 28, "Marketing")
]

df = spark.createDataFrame(data, ["id", "name", "age", "department"]) \
    .withColumn("created_at", current_timestamp())

# Write as Delta table
df.write.format("delta").mode("overwrite").save("./data/users")

print("✓ Delta table created at ./data/users")

# Read it back
users = spark.read.format("delta").load("./data/users")
users.show()
```

### Step 5: Perform Basic Operations
```python
from delta.tables import DeltaTable

# Load Delta table
delta_table = DeltaTable.forPath(spark, "./data/users")

# UPDATE: Give everyone in Engineering a raise (age += 1 as proxy)
delta_table.update(
    condition="department = 'Engineering'",
    set={"age": "age + 1"}
)

# DELETE: Remove users under 26
delta_table.delete("age < 26")

# Read updated data
spark.read.format("delta").load("./data/users").show()
```

## Key Features to Implement

### 1. CRUD Operations
- **Create**: Write DataFrames as Delta tables
- **Read**: Query Delta tables with Spark SQL
- **Update**: Modify records with conditions
- **Delete**: Remove records based on predicates
- **Insert**: Append new data

### 2. ACID Transactions
- Atomic writes (all or nothing)
- Concurrent reads during writes
- Transaction isolation levels
- Automatic rollback on failures
- Transaction log inspection

### 3. Time Travel
- Query historical versions by version number
- Query by timestamp
- View complete table history
- Restore to previous versions
- Compare data between versions

### 4. Merge Operations (Upsert)
```python
# Prepare updates and new records
updates = spark.createDataFrame([
    (2, "Bob Updated", 26, "Sales"),      # Update existing
    (5, "Eve", 32, "Engineering")         # Insert new
], ["id", "name", "age", "department"])

# Perform merge (upsert)
delta_table.alias("target").merge(
    updates.alias("source"),
    "target.id = source.id"
).whenMatchedUpdate(set={
    "name": "source.name",
    "age": "source.age",
    "department": "source.department"
}).whenNotMatchedInsert(values={
    "id": "source.id",
    "name": "source.name",
    "age": "source.age",
    "department": "source.department"
}).execute()
```

### 5. Optimization Techniques

**OPTIMIZE (Compact Small Files)**:
```python
# Before optimization
spark.sql("DESCRIBE DETAIL delta.`./data/users`").select("numFiles").show()

# Run OPTIMIZE
delta_table.optimize().executeCompaction()

# After optimization
spark.sql("DESCRIBE DETAIL delta.`./data/users`").select("numFiles").show()
```

**Z-ORDER (Data Skipping)**:
```python
# Co-locate data by frequently queried columns
delta_table.optimize().executeZOrderBy("department", "age")

# Queries on these columns will be faster
spark.read.format("delta").load("./data/users") \
    .filter("department = 'Engineering' AND age > 30") \
    .show()
```

**VACUUM (Remove Old Files)**:
```python
# Remove files older than retention period
delta_table.vacuum(retentionHours=168)  # 7 days

# For demo purposes only (removes all old files)
spark.conf.set("spark.databricks.delta.retentionDurationCheck.enabled", "false")
delta_table.vacuum(0)
```

### 6. Schema Evolution
```python
# Add new column
new_data = spark.createDataFrame([
    (6, "Frank", 40, "HR", "frank@example.com")
], ["id", "name", "age", "department", "email"])

new_data.write.format("delta") \
    .mode("append") \
    .option("mergeSchema", "true") \
    .save("./data/users")

# Schema automatically evolved!
spark.read.format("delta").load("./data/users").printSchema()
```

### 7. Change Data Feed (CDC)
```python
# Enable Change Data Feed
spark.sql("""
    ALTER TABLE delta.`./data/users`
    SET TBLPROPERTIES (delta.enableChangeDataFeed = true)
""")

# Read changes between versions
changes = spark.read.format("delta") \
    .option("readChangeFeed", "true") \
    .option("startingVersion", 0) \
    .option("endingVersion", 5) \
    .load("./data/users")

changes.show()
# Shows: _change_type (insert, update_preimage, update_postimage, delete)
```

## Success Criteria

By the end of this project, you should have:

- [ ] Spark + Delta Lake environment configured
- [ ] Delta tables created and queried
- [ ] All CRUD operations working (Create, Read, Update, Delete)
- [ ] Time travel queries functional (by version and timestamp)
- [ ] Table history viewed and analyzed
- [ ] Merge/upsert operations implemented
- [ ] OPTIMIZE executed with before/after metrics
- [ ] Z-ORDER applied with query performance tests
- [ ] VACUUM run with file cleanup verification
- [ ] Schema evolution demonstrated
- [ ] Change Data Feed enabled and queried
- [ ] 4 comprehensive Jupyter notebooks
- [ ] Performance benchmarks documented
- [ ] GitHub repository with all code

## Learning Outcomes

After completing this project, you'll be able to:

- Understand Delta Lake architecture and transaction log
- Implement ACID transactions in data lakes
- Use time travel for auditing and debugging
- Perform efficient upsert operations with MERGE
- Optimize table performance with OPTIMIZE and Z-ORDER
- Manage storage with VACUUM
- Evolve schemas without breaking changes
- Track changes with Change Data Feed
- Explain Delta Lake advantages over Parquet
- Compare Delta Lake vs Iceberg vs Hudi

## Expected Results

**Time Travel Performance**:
```
Version 0: 4 rows (initial data)
Version 1: 5 rows (after append)
Version 2: 4 rows (after update)
Version 3: 3 rows (after delete)

Query time: <1 second for any version
```

**Optimization Impact**:
```
Before OPTIMIZE:
  Files: 47 small files
  Total size: 2.3 MB
  Query time: 1.2s

After OPTIMIZE:
  Files: 3 optimized files
  Total size: 2.1 MB
  Query time: 0.3s (4x faster)
```

**Z-ORDER Benefits**:
```
Without Z-ORDER:
  Files scanned: 3/3 (100%)
  Query time: 0.3s

With Z-ORDER on department:
  Files scanned: 1/3 (33% - data skipping!)
  Query time: 0.1s (3x faster)
```

## Project Structure

```
project-d-delta-operations/
├── notebooks/
│   ├── 01_basic_operations.ipynb      # CRUD operations
│   ├── 02_time_travel.ipynb           # Version queries
│   ├── 03_optimization.ipynb          # OPTIMIZE, Z-ORDER, VACUUM
│   └── 04_advanced_features.ipynb     # Merge, CDF, schema evolution
├── scripts/
│   ├── setup.py                       # Environment setup
│   ├── generate_data.py               # Test data generation
│   └── benchmark.py                   # Performance tests
├── data/
│   └── users/                         # Delta table storage
│       ├── _delta_log/                # Transaction log
│       └── *.parquet                  # Data files
├── results/
│   ├── optimization_metrics.md
│   └── performance_charts.png
├── prd.md
├── tech-spec.md
├── implementation-plan.md
└── README.md
```

## Common Challenges & Solutions

### Challenge 1: Spark Configuration Issues
**Problem**: Delta Lake extensions not loading
**Solution**: Ensure correct package version and configuration order
```python
# Correct order matters!
builder = SparkSession.builder \
    .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension") \
    .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog")
```

### Challenge 2: VACUUM Retention Check
**Problem**: VACUUM fails with retention check error
**Solution**: Understand retention period and disable check only for testing
```python
# Production: Use proper retention (7+ days)
delta_table.vacuum(retentionHours=168)

# Testing only: Disable check
spark.conf.set("spark.databricks.delta.retentionDurationCheck.enabled", "false")
delta_table.vacuum(0)
```

### Challenge 3: Schema Evolution Conflicts
**Problem**: Schema mismatch when appending data
**Solution**: Enable mergeSchema option
```python
df.write.format("delta") \
    .mode("append") \
    .option("mergeSchema", "true") \
    .save("./data/users")
```

### Challenge 4: Time Travel Performance
**Problem**: Querying old versions is slow
**Solution**: Old versions require reading multiple files; use OPTIMIZE regularly

## Next Steps

After completing this project:

1. **Add to Portfolio**: Document on GitHub with performance metrics
2. **Write Blog Post**: "Delta Lake: ACID Transactions for Data Lakes"
3. **Compare Formats**: Build comparison with Iceberg (Project C)
4. **Build Project E**: Continue with Spark Streaming
5. **Production Use**: Apply Delta Lake in real data pipelines
6. **Advanced Topics**: Explore Delta Sharing, Liquid Clustering

## Resources

- [Delta Lake Documentation](https://docs.delta.io/)
- [Delta Lake Python API](https://docs.delta.io/latest/api/python/index.html)
- [Optimization Guide](https://docs.delta.io/latest/optimizations-oss.html)
- [Time Travel Guide](https://docs.delta.io/latest/delta-batch.html#deltatimetravel)
- [Merge Operations](https://docs.delta.io/latest/delta-update.html#upsert-into-a-table-using-merge)
- [Change Data Feed](https://docs.delta.io/latest/delta-change-data-feed.html)

## Questions?

If you get stuck:
1. Review the tech-spec.md for detailed code examples
2. Check Delta Lake documentation for specific operations
3. Search Delta Lake community forums
4. Review the 100 Days bootcamp materials on table formats
5. Compare with Databricks Delta Lake examples
