# Technical Specification: Delta Lake Operations

## Architecture
```
PySpark → Delta Lake Tables → Parquet Files + Transaction Log
              ↓
         Jupyter Notebooks (examples)
```

## Technology Stack
- **Python**: 3.11+
- **PySpark**: 3.5+
- **Delta Lake**: 2.4+
- **Jupyter**: For notebooks
- **Pandas**: For data manipulation

## Spark Configuration

```python
from pyspark.sql import SparkSession

spark = SparkSession.builder \
    .appName("DeltaLakeDemo") \
    .config("spark.jars.packages", "io.delta:delta-core_2.12:2.4.0") \
    .config("spark.sql.extensions", 
            "io.delta.sql.DeltaSparkSessionExtensions") \
    .config("spark.sql.catalog.spark_catalog", 
            "org.apache.spark.sql.delta.catalog.DeltaCatalog") \
    .getOrCreate()
```

## Core Operations

### 1. Create & Insert
```python
# Create Delta table
df.write.format("delta").save("./data/users")

# Or with SQL
spark.sql("""
    CREATE TABLE users (
        id INT,
        name STRING,
        age INT,
        created_at TIMESTAMP
    ) USING DELTA
    LOCATION './data/users'
""")

# Insert data
new_data.write.format("delta").mode("append").save("./data/users")
```

### 2. Update & Delete
```python
from delta.tables import DeltaTable

delta_table = DeltaTable.forPath(spark, "./data/users")

# Update
delta_table.update(
    condition="age < 18",
    set={"age": "18"}
)

# Delete
delta_table.delete("created_at < '2020-01-01'")
```

### 3. Merge (Upsert)
```python
delta_table.alias("target").merge(
    updates.alias("source"),
    "target.id = source.id"
).whenMatchedUpdate(set={
    "name": "source.name",
    "age": "source.age"
}).whenNotMatchedInsert(values={
    "id": "source.id",
    "name": "source.name",
    "age": "source.age"
}).execute()
```

### 4. Time Travel
```python
# Query by version
df = spark.read.format("delta").option("versionAsOf", 0).load("./data/users")

# Query by timestamp
df = spark.read.format("delta") \
    .option("timestampAsOf", "2024-01-01") \
    .load("./data/users")

# View history
delta_table.history().show()

# Restore to version
delta_table.restoreToVersion(5)
```

### 5. Optimization
```python
# Optimize (compact small files)
delta_table.optimize().executeCompaction()

# Z-ORDER (co-locate related data)
delta_table.optimize().executeZOrderBy("age", "created_at")

# Vacuum (remove old files)
delta_table.vacuum(retentionHours=168)  # 7 days

# Analyze for statistics
spark.sql("ANALYZE TABLE users COMPUTE STATISTICS")
```

### 6. Change Data Feed
```python
# Enable CDF
spark.sql("""
    ALTER TABLE users 
    SET TBLPROPERTIES (delta.enableChangeDataFeed = true)
""")

# Read changes
changes = spark.read.format("delta") \
    .option("readChangeFeed", "true") \
    .option("startingVersion", 5) \
    .option("endingVersion", 10) \
    .load("./data/users")
```

## Notebook Structure

### Notebook 1: Basic Operations
1. Setup and configuration
2. Create Delta table
3. Insert data
4. Read data
5. Update records
6. Delete records

### Notebook 2: Time Travel
1. Insert multiple versions
2. Query historical versions
3. View table history
4. Restore to previous version
5. Compare versions

### Notebook 3: Optimization
1. Create table with small files
2. Show file statistics
3. Run OPTIMIZE
4. Compare before/after
5. Demonstrate Z-ORDER
6. Run VACUUM

### Notebook 4: Advanced Features
1. MERGE operations (upsert)
2. Schema evolution
3. Change Data Feed
4. Partition management
5. Constraints and checks

## Example Workflow

```python
# 1. Create table with initial data
df1 = spark.createDataFrame([
    (1, "Alice", 30),
    (2, "Bob", 25)
], ["id", "name", "age"])

df1.write.format("delta").save("./data/users")

# 2. Append more data (version 1)
df2 = spark.createDataFrame([
    (3, "Charlie", 35)
], ["id", "name", "age"])

df2.write.format("delta").mode("append").save("./data/users")

# 3. Update data (version 2)
delta_table = DeltaTable.forPath(spark, "./data/users")
delta_table.update(
    condition="name = 'Bob'",
    set={"age": "26"}
)

# 4. Query current version
current = spark.read.format("delta").load("./data/users")
current.show()

# 5. Time travel to version 0
v0 = spark.read.format("delta").option("versionAsOf", 0).load("./data/users")
v0.show()  # Only Alice and Bob

# 6. View history
delta_table.history().select("version", "timestamp", "operation").show()

# 7. Optimize
delta_table.optimize().executeCompaction()

# 8. Vacuum old files
delta_table.vacuum(0)  # For demo only, use higher retention in production
```

## Project Structure
```
project-d-delta-operations/
├── notebooks/
│   ├── 01_basic_operations.ipynb
│   ├── 02_time_travel.ipynb
│   ├── 03_optimization.ipynb
│   └── 04_advanced_features.ipynb
├── scripts/
│   ├── setup.py
│   └── generate_data.py
├── data/
│   └── delta_tables/
└── README.md
```

## Testing Strategy
- Run all notebook cells
- Verify data integrity after operations
- Test time travel accuracy
- Validate optimization improvements
- Check error handling

## Performance Metrics
- File count before/after OPTIMIZE
- Query time improvements with Z-ORDER
- Storage savings after VACUUM
