# Technical Specification: Iceberg Table Manager

## Architecture
```
CLI → Iceberg Manager → Local Catalog → Parquet Files
                            ↓
                       DuckDB (queries)
```

## Technology Stack
- **Rust**: 1.75+
- **Python**: 3.11+
- **Libraries**:
  - iceberg-rust (Rust)
  - pyiceberg (Python)
  - duckdb (Python)
  - arrow-rs (Rust)

## Core Components

### Iceberg Manager (Python)
```python
class IcebergManager:
    def __init__(self, catalog_path: str)
    
    def create_table(self, name: str, schema: Schema) -> Table
    def insert_data(self, table: str, data: DataFrame)
    def update_data(self, table: str, updates: dict)
    def delete_data(self, table: str, condition: str)
    
    def query_snapshot(self, table: str, snapshot_id: int) -> DataFrame
    def list_snapshots(self, table: str) -> List[Snapshot]
    def rollback(self, table: str, snapshot_id: int)
    
    def evolve_schema(self, table: str, changes: SchemaChange)
```

### Rust Utilities
```rust
// Fast data generation and processing
pub fn generate_test_data(rows: usize) -> Vec<Record>
pub fn write_parquet_batch(data: &[Record], path: &str)
```

## Iceberg Features Demonstrated

### 1. ACID Transactions
```python
# Atomic insert
manager.insert_data("users", new_users_df)

# Concurrent reads work during writes
df = manager.query_table("users")
```

### 2. Time Travel
```python
# Query as of timestamp
df = manager.query_as_of("users", timestamp="2024-01-01 10:00:00")

# Query specific snapshot
df = manager.query_snapshot("users", snapshot_id=12345)

# List all snapshots
snapshots = manager.list_snapshots("users")
```

### 3. Schema Evolution
```python
# Add column (non-breaking)
manager.add_column("users", "email", StringType())

# Rename column
manager.rename_column("users", "name", "full_name")

# Drop column
manager.drop_column("users", "deprecated_field")
```

### 4. Snapshot Management
```python
# Expire old snapshots
manager.expire_snapshots("users", older_than_days=30)

# Rollback to previous version
manager.rollback("users", snapshot_id=12340)
```

## DuckDB Integration

```python
import duckdb

# Query Iceberg table with DuckDB
con = duckdb.connect()
con.install_extension("iceberg")
con.load_extension("iceberg")

# Query current version
df = con.execute("""
    SELECT * FROM iceberg_scan('data/warehouse/users')
    WHERE age > 30
""").df()

# Query historical snapshot
df = con.execute("""
    SELECT * FROM iceberg_scan('data/warehouse/users', 
                                snapshot_id=12345)
""").df()
```

## CLI Interface

```bash
# Create table
iceberg-manager create users --schema schema.json

# Insert data
iceberg-manager insert users --data users.csv

# Query table
iceberg-manager query users --where "age > 30"

# Time travel
iceberg-manager query users --snapshot 12345
iceberg-manager query users --timestamp "2024-01-01 10:00:00"

# Schema evolution
iceberg-manager add-column users email string
iceberg-manager rename-column users name full_name

# Snapshot management
iceberg-manager list-snapshots users
iceberg-manager rollback users --snapshot 12340
iceberg-manager expire-snapshots users --days 30
```

## Example Workflow

```python
# 1. Create table
manager = IcebergManager("./data/warehouse")
manager.create_table("events", schema)

# 2. Insert initial data
manager.insert_data("events", df1)
snapshot1 = manager.get_current_snapshot("events")

# 3. Insert more data
manager.insert_data("events", df2)
snapshot2 = manager.get_current_snapshot("events")

# 4. Query current state
current = manager.query_table("events")

# 5. Time travel to snapshot1
historical = manager.query_snapshot("events", snapshot1.id)

# 6. Evolve schema
manager.add_column("events", "user_agent", StringType())

# 7. Insert with new schema
manager.insert_data("events", df3)  # includes user_agent

# 8. Query still works on old snapshots
old_data = manager.query_snapshot("events", snapshot1.id)
```

## Project Structure
```
project-c-iceberg-manager/
├── rust/
│   ├── src/
│   │   └── data_gen.rs
│   └── Cargo.toml
├── python/
│   ├── iceberg_manager/
│   │   ├── __init__.py
│   │   ├── manager.py
│   │   └── cli.py
│   ├── examples/
│   │   ├── basic_operations.py
│   │   ├── time_travel.py
│   │   └── schema_evolution.py
│   └── tests/
├── data/
│   └── warehouse/
└── README.md
```

## Testing Strategy
- Test CRUD operations
- Verify time travel accuracy
- Test schema evolution scenarios
- Validate snapshot management
- Integration tests with DuckDB

## Performance Targets
- Create table: < 1 second
- Insert 1M rows: < 10 seconds
- Time travel query: < 2 seconds
- Schema evolution: < 1 second
