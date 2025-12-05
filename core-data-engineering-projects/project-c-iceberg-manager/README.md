# Project C: Iceberg Table Manager

## Objective

Build a comprehensive tool for managing Apache Iceberg tables, demonstrating time travel, snapshots, schema evolution, and table maintenance operations.

**What You'll Build**: A CLI tool and Python library for creating, querying, and maintaining Iceberg tables with full support for ACID transactions, time travel, schema evolution, and snapshot management.

**What You'll Learn**: Apache Iceberg table format, ACID transactions in data lakes, time travel queries, schema evolution patterns, and table maintenance best practices.

## Time Estimate

**2 days (16 hours)**

- Hours 1-4: Iceberg setup and basic table operations
- Hours 5-8: Time travel and snapshot management
- Hours 9-12: Schema evolution and maintenance
- Hours 13-16: CLI tool and comprehensive examples

## Prerequisites

### Required Knowledge
Complete these bootcamp sections first:
- [100 Days Data & AI](https://github.com/washimimizuku/100-days-data-ai) - Days 11-20
  - Days 11-15: Table formats (Iceberg focus)
  - Days 16-20: ACID transactions and time travel
- [60 Days Advanced](https://github.com/washimimizuku/60-days-advanced-data-ai) - Days 1-5
  - Days 1-5: Advanced table format patterns

### Technical Requirements
- Python 3.11+
- Understanding of table formats
- Basic SQL knowledge
- Familiarity with Parquet

### Tools Needed
- Python with pyiceberg, duckdb, pandas
- Git for version control
- Sample datasets

## Getting Started

### Step 1: Review Documentation
Read the project documents in order:
1. `prd.md` - Understand what you're building
2. `tech-spec.md` - Review technical architecture
3. `implementation-plan.md` - Follow the implementation steps

### Step 2: Set Up Environment
```bash
# Install dependencies
pip install pyiceberg duckdb pandas pyarrow

# Create project structure
mkdir -p iceberg-manager/{data,catalog,checkpoints}
cd iceberg-manager
```

### Step 3: Initialize Iceberg Catalog
```python
from pyiceberg.catalog import load_catalog

# Create local catalog
catalog = load_catalog(
    "local",
    **{
        "type": "sql",
        "uri": "sqlite:///catalog.db",
        "warehouse": "file://./data"
    }
)

# Create namespace
catalog.create_namespace("demo")
```

### Step 4: Create Your First Table
```python
from pyiceberg.schema import Schema
from pyiceberg.types import NestedField, StringType, IntegerType, TimestampType

# Define schema
schema = Schema(
    NestedField(1, "id", IntegerType(), required=True),
    NestedField(2, "name", StringType(), required=True),
    NestedField(3, "age", IntegerType()),
    NestedField(4, "created_at", TimestampType(), required=True),
)

# Create table
table = catalog.create_table(
    "demo.users",
    schema=schema,
)

print(f"Created table: {table}")
```

### Step 5: Insert and Query Data
```python
import pandas as pd
from datetime import datetime

# Create sample data
data = pd.DataFrame({
    'id': [1, 2, 3],
    'name': ['Alice', 'Bob', 'Charlie'],
    'age': [30, 25, 35],
    'created_at': [datetime.now()] * 3
})

# Insert data
table.append(data)

# Query data
df = table.scan().to_pandas()
print(df)
```

## Key Features to Implement

### 1. Table Operations
- Create tables with schema definition
- Insert data (append mode)
- Update records
- Delete records
- Upsert (merge) operations

### 2. Time Travel
- Query table as of specific timestamp
- Query specific snapshot by ID
- List all snapshots with metadata
- Compare data between snapshots

### 3. Snapshot Management
- Create named snapshots
- Rollback to previous snapshot
- Expire old snapshots
- Snapshot retention policies

### 4. Schema Evolution
- Add columns (non-breaking)
- Rename columns
- Drop columns
- Change column types (compatible changes)
- Reorder columns

### 5. Table Maintenance
- Compact small files
- Rewrite manifests
- Expire snapshots
- Remove orphan files
- Optimize file layout

### 6. CLI Interface
```bash
# Create table
iceberg-manager create demo.users --schema schema.json

# Insert data
iceberg-manager insert demo.users --data users.csv

# Time travel query
iceberg-manager query demo.users --snapshot-id 12345

# List snapshots
iceberg-manager snapshots demo.users

# Compact files
iceberg-manager compact demo.users
```

## Success Criteria

By the end of this project, you should have:

- [ ] Iceberg catalog configured (local or cloud)
- [ ] Tables created with proper schemas
- [ ] CRUD operations working (insert, update, delete)
- [ ] Time travel queries functional
- [ ] Snapshot management implemented
- [ ] Schema evolution examples
- [ ] Table maintenance operations
- [ ] CLI tool with clean interface
- [ ] Comprehensive documentation
- [ ] Test suite validating operations
- [ ] GitHub repository with examples

## Learning Outcomes

After completing this project, you'll be able to:

- Understand Apache Iceberg architecture
- Implement ACID transactions in data lakes
- Use time travel for historical queries
- Evolve schemas without breaking changes
- Manage snapshots and table history
- Perform table maintenance operations
- Explain Iceberg advantages over Hive tables
- Compare Iceberg vs Delta Lake vs Hudi

## Expected Capabilities

**Time Travel Example**:
```python
# Query current data
current = table.scan().to_pandas()

# Query as of 1 hour ago
historical = table.scan(
    snapshot_id=table.history()[0].snapshot_id
).to_pandas()

# Compare changes
diff = current.merge(historical, on='id', how='outer', 
                     suffixes=('_now', '_then'))
```

**Schema Evolution Example**:
```python
# Add new column (non-breaking)
table.update_schema().add_column("email", StringType()).commit()

# Rename column
table.update_schema().rename_column("name", "full_name").commit()

# Old queries still work!
df = table.scan().to_pandas()
```

## Project Structure

```
project-c-iceberg-manager/
├── src/
│   ├── manager.py           # Main Iceberg manager class
│   ├── operations.py        # CRUD operations
│   ├── time_travel.py       # Time travel queries
│   ├── schema_evolution.py  # Schema changes
│   ├── maintenance.py       # Table maintenance
│   └── cli.py               # CLI interface
├── tests/
│   ├── test_operations.py
│   ├── test_time_travel.py
│   └── test_schema.py
├── examples/
│   ├── basic_usage.py
│   ├── time_travel_demo.py
│   ├── schema_evolution_demo.py
│   └── maintenance_demo.py
├── data/                    # Iceberg warehouse
├── catalog.db               # SQLite catalog
├── prd.md
├── tech-spec.md
├── implementation-plan.md
└── README.md
```

## Advanced Features

### 1. Partition Evolution
```python
# Start with daily partitions
table = catalog.create_table(
    "demo.events",
    schema=schema,
    partition_spec=PartitionSpec(
        PartitionField(source_id=4, field_id=1000, 
                      transform=DayTransform(), name="event_day")
    )
)

# Later, evolve to hourly partitions (non-breaking!)
table.update_spec().add_field("event_hour", HourTransform(), "created_at").commit()

# Old queries still work, new data uses new partitioning
```

### 2. Hidden Partitioning
```python
# Iceberg handles partitioning automatically
# No need to specify partition in queries!

# Query without partition filter (Iceberg optimizes)
df = table.scan().filter("created_at > '2024-01-01'").to_pandas()

# Iceberg automatically prunes partitions
```

### 3. Metadata Tables
```python
# Query table history
history = table.scan(table_name="demo.users.history").to_pandas()

# Query snapshots metadata
snapshots = table.scan(table_name="demo.users.snapshots").to_pandas()

# Query files metadata
files = table.scan(table_name="demo.users.files").to_pandas()

# Query manifests
manifests = table.scan(table_name="demo.users.manifests").to_pandas()
```

### 4. Incremental Reads
```python
# Read only new data since last snapshot
last_snapshot_id = 12345
new_data = table.scan(
    snapshot_id=table.current_snapshot().snapshot_id
).filter(
    f"snapshot_id > {last_snapshot_id}"
).to_pandas()
```

## Common Challenges & Solutions

### Challenge 1: Catalog Configuration
**Problem**: Setting up Iceberg catalog correctly
**Solution**: Start with SQLite catalog for local development, use Hive or Glue for production
```python
# Local development (SQLite)
catalog = load_catalog(
    "local",
    **{
        "type": "sql",
        "uri": "sqlite:///catalog.db",
        "warehouse": "file://./data"
    }
)

# Production (AWS Glue)
catalog = load_catalog(
    "glue",
    **{
        "type": "glue",
        "warehouse": "s3://my-bucket/warehouse"
    }
)
```

### Challenge 2: Schema Evolution Compatibility
**Problem**: Some schema changes break compatibility
**Solution**: Follow Iceberg's schema evolution rules, test changes before production
```python
# Safe changes (non-breaking)
table.update_schema().add_column("new_field", StringType()).commit()
table.update_schema().rename_column("old_name", "new_name").commit()

# Unsafe changes (breaking)
# Don't change column types incompatibly
# Don't drop required columns without migration

# Test schema changes
def test_schema_evolution():
    # Create test table
    # Apply schema change
    # Verify old queries still work
    # Verify new queries work with new schema
```

### Challenge 3: Snapshot Retention
**Problem**: Too many snapshots consuming storage
**Solution**: Implement retention policies, expire old snapshots regularly
```python
from datetime import datetime, timedelta

def expire_old_snapshots(table, retention_days=7):
    """Expire snapshots older than retention period"""
    cutoff = datetime.now() - timedelta(days=retention_days)
    
    table.expire_snapshots(
        older_than=cutoff,
        retain_last=5  # Always keep at least 5 snapshots
    )
```

### Challenge 4: Small Files Problem
**Problem**: Many small files degrading query performance
**Solution**: Regular compaction and file size tuning
```python
# Compact small files
from pyiceberg.table import Table

def compact_table(table: Table, target_file_size_mb=128):
    """Compact small files into larger ones"""
    table.rewrite_data_files(
        target_file_size_bytes=target_file_size_mb * 1024 * 1024
    )
```

## Next Steps

After completing this project:

1. **Add to Portfolio**: Document on GitHub with time travel examples
2. **Write Blog Post**: "Apache Iceberg: Time Travel for Data Lakes"
3. **Extend Features**: Add partition evolution, hidden partitioning
4. **Build Project D**: Continue with Delta Lake Operations
5. **Production Use**: Integrate with Spark or Trino for queries

## Troubleshooting

### Installation Issues

**Issue**: PyIceberg installation fails
```bash
# Solution: Install with specific version
pip install "pyiceberg[pyarrow,duckdb]==0.5.0"
```

**Issue**: Catalog connection errors
```python
# Solution: Verify catalog configuration
from pyiceberg.catalog import load_catalog

try:
    catalog = load_catalog("local", **config)
    print("Catalog connected successfully")
except Exception as e:
    print(f"Catalog error: {e}")
    # Check warehouse path exists
    # Verify SQLite database permissions
```

### Runtime Issues

**Issue**: "Table not found" error
```python
# Solution: List available tables
namespaces = catalog.list_namespaces()
for ns in namespaces:
    tables = catalog.list_tables(ns)
    print(f"Namespace {ns}: {tables}")
```

**Issue**: Schema evolution fails
```python
# Solution: Check schema compatibility
def validate_schema_change(table, new_column_name, new_column_type):
    current_schema = table.schema()
    
    # Check if column already exists
    if new_column_name in [field.name for field in current_schema.fields]:
        raise ValueError(f"Column {new_column_name} already exists")
    
    # Apply change
    table.update_schema().add_column(new_column_name, new_column_type).commit()
```

**Issue**: Time travel query returns empty results
```python
# Solution: Verify snapshot exists and has data
snapshots = table.history()
for snapshot in snapshots:
    print(f"Snapshot {snapshot.snapshot_id}: {snapshot.summary}")
    
# Query with valid snapshot ID
df = table.scan(snapshot_id=valid_snapshot_id).to_pandas()
```

### Performance Issues

**Issue**: Slow queries on large tables
```python
# Solution: Use partition pruning and column projection
df = table.scan(
    row_filter="date >= '2024-01-01'",  # Partition pruning
    selected_fields=["id", "name"]       # Column projection
).to_pandas()
```

**Issue**: Too many small files
```bash
# Solution: Run regular compaction
python -c "
from pyiceberg.catalog import load_catalog
catalog = load_catalog('local', **config)
table = catalog.load_table('demo.users')
table.rewrite_data_files(target_file_size_bytes=128*1024*1024)
"
```

## Real-World Use Cases

### Use Case 1: Data Versioning for ML
**Scenario**: Track training data versions for ML models
**Solution**:
```python
# Create snapshot before training
table.append(training_data)
snapshot_id = table.current_snapshot().snapshot_id

# Train model
model = train_model(training_data)

# Store snapshot ID with model metadata
model_metadata = {
    'model_version': '1.0',
    'training_snapshot': snapshot_id,
    'timestamp': datetime.now()
}

# Later, reproduce exact training data
training_data = table.scan(snapshot_id=snapshot_id).to_pandas()
```

### Use Case 2: Regulatory Compliance
**Scenario**: Maintain audit trail for financial data
**Solution**:
```python
# All changes are tracked in snapshots
history = table.history()

# Query data as of specific date for audit
audit_date = datetime(2024, 1, 1)
audit_data = table.scan(
    as_of_timestamp=audit_date
).to_pandas()

# Generate compliance report
report = generate_audit_report(audit_data, audit_date)
```

### Use Case 3: A/B Testing Analysis
**Scenario**: Compare metrics before and after feature launch
**Solution**:
```python
# Snapshot before feature launch
pre_launch_snapshot = table.current_snapshot().snapshot_id

# Feature launches, data continues to flow
time.sleep(3600)  # 1 hour later

# Snapshot after feature launch
post_launch_snapshot = table.current_snapshot().snapshot_id

# Compare metrics
pre_metrics = calculate_metrics(
    table.scan(snapshot_id=pre_launch_snapshot).to_pandas()
)
post_metrics = calculate_metrics(
    table.scan(snapshot_id=post_launch_snapshot).to_pandas()
)

# Analyze impact
impact = compare_metrics(pre_metrics, post_metrics)
```

### Use Case 4: Data Recovery
**Scenario**: Recover from accidental data deletion
**Solution**:
```python
# Accidental deletion
table.delete("status = 'deleted'")  # Oops, wrong condition!

# List recent snapshots
snapshots = table.history()[-5:]  # Last 5 snapshots

# Rollback to before deletion
table.rollback_to_snapshot(snapshots[-2].snapshot_id)

# Data is recovered!
```

## Comparison with Other Table Formats

### vs Delta Lake
| Feature | Iceberg | Delta Lake |
|---------|---------|------------|
| **Partition Evolution** | ✅ Yes | ❌ No |
| **Hidden Partitioning** | ✅ Yes | ❌ No |
| **Time Travel** | ✅ Yes | ✅ Yes |
| **Schema Evolution** | ✅ Yes | ✅ Yes |
| **Multi-Engine Support** | ✅ Excellent | ⚠️ Limited |
| **Metadata Tables** | ✅ Yes | ⚠️ Limited |

### vs Apache Hudi
| Feature | Iceberg | Hudi |
|---------|---------|------|
| **Simplicity** | ✅ Simple | ⚠️ Complex |
| **Time Travel** | ✅ Yes | ✅ Yes |
| **Upserts** | ✅ Yes | ✅ Yes (better) |
| **Streaming** | ✅ Yes | ✅ Yes (better) |
| **Adoption** | ✅ Growing | ⚠️ Moderate |

### When to Use Iceberg
- Need partition evolution
- Multi-engine access (Spark, Trino, Flink)
- Strong schema evolution requirements
- Cloud-native architecture

## Resources

### Documentation
- [Apache Iceberg Documentation](https://iceberg.apache.org/) - Official docs
- [PyIceberg Documentation](https://py.iceberg.apache.org/) - Python library
- [Iceberg Table Spec](https://iceberg.apache.org/docs/latest/spec/) - Format specification
- [Schema Evolution Guide](https://iceberg.apache.org/docs/latest/evolution/) - Best practices

### Tutorials
- [Iceberg: The Definitive Guide](https://www.dremio.com/resources/guides/apache-iceberg-an-architectural-look-under-the-covers/) - Architecture deep dive
- [Getting Started with Iceberg](https://iceberg.apache.org/docs/latest/getting-started/) - Official tutorial
- [Iceberg vs Delta vs Hudi](https://www.onehouse.ai/blog/apache-hudi-vs-delta-lake-vs-apache-iceberg-lakehouse-feature-comparison) - Comparison

### Community
- [Iceberg Slack](https://apache-iceberg.slack.com/) - Community support
- [GitHub Discussions](https://github.com/apache/iceberg/discussions) - Q&A
- [Iceberg Meetups](https://www.meetup.com/topics/apache-iceberg/) - Local groups

## Questions?

If you get stuck:
1. Review the `tech-spec.md` for detailed architecture
2. Check PyIceberg documentation and examples
3. Search Iceberg community Slack
4. Review the 100 Days bootcamp materials on table formats
5. Test with small datasets first
6. Compare behavior with Delta Lake (Project D)

## Related Projects

After completing this project, consider:
- **Project D**: Delta Operations - Compare with Delta Lake
- **Project E**: Spark Streaming - Stream data into Iceberg tables
- **Project H**: Format Benchmark - Compare table format performance
- Build a data lakehouse using Iceberg with Trino or Spark
