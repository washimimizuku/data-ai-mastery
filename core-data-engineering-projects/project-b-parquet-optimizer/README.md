# Project B: Parquet Optimizer

## Objective

Build a tool to analyze and optimize Parquet file layouts, compression algorithms, and encoding strategies for better query performance and storage efficiency.

**What You'll Build**: A Rust CLI tool that analyzes Parquet files, tests different compression algorithms and row group sizes, and provides data-driven recommendations for optimal configurations.

**What You'll Learn**: Parquet file format internals, compression algorithms (Snappy, ZSTD, LZ4, Gzip), columnar storage optimization, and performance benchmarking.

## Time Estimate

**1-2 days (8-16 hours)**

- Hours 1-4: Parquet analyzer implementation
- Hours 5-8: Optimizer with compression testing
- Hours 9-12: Row group size optimization
- Hours 13-16: CLI interface and reporting

## Prerequisites

### Required Knowledge
Complete these bootcamp sections first:
- [100 Days Data & AI](https://github.com/washimimizuku/100-days-data-ai) - Days 1-10
  - Days 1-5: Data formats (Parquet focus)
  - Days 6-10: Compression and optimization
- [30 Days of Rust](https://github.com/washimimizuku/30-days-rust-data-ai) - Days 1-20 (optional, for Rust implementation)
- [30 Days of Python](https://github.com/washimimizuku/30-days-python-data-ai) - Days 1-15 (for Python implementation)

### Technical Requirements
- Rust 1.75+ or Python 3.11+
- Understanding of columnar storage
- Basic knowledge of compression algorithms

### Tools Needed
- Rust toolchain or Python with PyArrow
- Sample Parquet files (or generate them)
- Git for version control

## Getting Started

### Step 1: Review Documentation
Read the project documents in order:
1. `prd.md` - Understand what you're building
2. `tech-spec.md` - Review technical architecture
3. `implementation-plan.md` - Follow the implementation steps

### Step 2: Set Up Environment

**Option A: Rust Implementation**
```bash
# Create new Rust project
cargo new parquet-optimizer
cd parquet-optimizer

# Add dependencies to Cargo.toml
[dependencies]
parquet = "49.0"
arrow = "49.0"
clap = { version = "4.4", features = ["derive"] }
serde = { version = "1.0", features = ["derive"] }
serde_json = "1.0"
```

**Option B: Python Implementation**
```bash
# Install dependencies
pip install pyarrow pandas
```

### Step 3: Build Parquet Analyzer

**Rust Example**:
```rust
use parquet::file::reader::{FileReader, SerializedFileReader};
use std::fs::File;

pub struct ParquetAnalyzer {
    file_path: String,
}

impl ParquetAnalyzer {
    pub fn analyze(&self) -> Result<FileMetadata, Error> {
        let file = File::open(&self.file_path)?;
        let reader = SerializedFileReader::new(file)?;
        
        let metadata = reader.metadata();
        let num_rows = metadata.file_metadata().num_rows();
        let num_row_groups = metadata.num_row_groups();
        
        Ok(FileMetadata {
            num_rows,
            num_row_groups,
            schema: metadata.file_metadata().schema().clone(),
            compression: self.get_compression_info(metadata),
        })
    }
}
```

**Python Example**:
```python
import pyarrow.parquet as pq

class ParquetAnalyzer:
    def __init__(self, file_path):
        self.file_path = file_path
        
    def analyze(self):
        parquet_file = pq.ParquetFile(self.file_path)
        metadata = parquet_file.metadata
        
        return {
            'num_rows': metadata.num_rows,
            'num_row_groups': metadata.num_row_groups,
            'schema': parquet_file.schema,
            'file_size': os.path.getsize(self.file_path),
        }
```

### Step 4: Implement Optimizer
```python
# Test different compressions
compressions = ['SNAPPY', 'ZSTD', 'LZ4', 'GZIP', 'NONE']

for compression in compressions:
    start = time.time()
    pq.write_table(table, f'output_{compression}.parquet', 
                   compression=compression)
    write_time = time.time() - start
    
    file_size = os.path.getsize(f'output_{compression}.parquet')
    print(f"{compression}: {file_size / 1024 / 1024:.2f} MB, {write_time:.2f}s")
```

### Step 5: Create CLI Interface
```bash
# Analyze a Parquet file
parquet-optimizer analyze data.parquet

# Test all compressions
parquet-optimizer optimize data.parquet --test-all

# Generate report
parquet-optimizer report data.parquet --output report.json
```

## Key Features to Implement

### 1. Parquet Analyzer
- Read and parse Parquet metadata
- Extract schema information
- Analyze row group sizes
- Identify current compression
- Calculate file statistics

### 2. Compression Testing
- Test multiple algorithms: Snappy, ZSTD, LZ4, Gzip, Uncompressed
- Measure file size for each
- Measure write time
- Measure read time
- Calculate compression ratios

### 3. Row Group Optimization
- Test different row group sizes (128MB, 256MB, 512MB, 1GB)
- Measure impact on query performance
- Analyze memory usage
- Provide recommendations

### 4. Performance Benchmarking
- Write performance (time to write)
- Read performance (time to read)
- Query performance (filter, aggregate)
- Memory usage during operations

### 5. Reporting
- Detailed analysis report
- Comparison tables
- Recommendations based on use case
- JSON/Markdown output formats

## Success Criteria

By the end of this project, you should have:

- [ ] Parquet analyzer reading metadata correctly
- [ ] Compression testing for 5+ algorithms
- [ ] Row group size optimization tests
- [ ] Performance benchmarks (write, read, query)
- [ ] Comprehensive report generation
- [ ] CLI tool with clean interface
- [ ] Documentation with usage examples
- [ ] Test suite validating optimizations
- [ ] GitHub repository with sample files
- [ ] Performance comparison charts

## Learning Outcomes

After completing this project, you'll be able to:

- Understand Parquet file format internals
- Explain compression algorithm trade-offs
- Optimize Parquet files for different use cases
- Benchmark storage and query performance
- Choose appropriate row group sizes
- Explain columnar storage advantages
- Make data-driven optimization decisions

## Expected Results

Based on typical optimizations:

**Compression Comparison (10M rows, 15 columns)**:
- Uncompressed: 3.5 GB (baseline)
- Snappy: 1.2 GB (2.9x, fast read/write)
- LZ4: 1.4 GB (2.5x, fastest)
- ZSTD: 0.8 GB (4.4x, best compression)
- Gzip: 0.9 GB (3.9x, good compression)

**Use Case Recommendations**:
- **Storage-optimized**: ZSTD (4.4x compression)
- **Speed-optimized**: LZ4 (fastest read/write)
- **Balanced**: Snappy (default, good balance)

## Project Structure

```
project-b-parquet-optimizer/
├── src/
│   ├── main.rs              # CLI entry point
│   ├── analyzer.rs          # Parquet analysis
│   ├── optimizer.rs         # Optimization tests
│   ├── benchmarks.rs        # Performance benchmarks
│   └── report.rs            # Report generation
├── tests/
│   ├── test_analyzer.rs
│   └── test_optimizer.rs
├── examples/
│   ├── generate_sample.py   # Generate test data
│   └── sample_data.parquet
├── results/
│   ├── compression_comparison.md
│   └── optimization_report.json
├── prd.md
├── tech-spec.md
├── implementation-plan.md
└── README.md
```

## Advanced Optimization Techniques

### 1. Dictionary Encoding
```rust
// Check if dictionary encoding is beneficial
fn should_use_dictionary(column: &Column) -> bool {
    let unique_values = column.unique_count();
    let total_values = column.len();
    
    // Use dictionary if < 50% unique values
    (unique_values as f64 / total_values as f64) < 0.5
}
```

### 2. Predicate Pushdown Analysis
```python
# Analyze which columns benefit from statistics
def analyze_filter_columns(parquet_file):
    metadata = parquet_file.metadata
    
    for rg in range(metadata.num_row_groups):
        row_group = metadata.row_group(rg)
        for col in range(row_group.num_columns):
            column = row_group.column(col)
            stats = column.statistics
            
            if stats:
                print(f"Column {col}: min={stats.min}, max={stats.max}")
                # Columns with good min/max stats benefit from pushdown
```

### 3. Bloom Filter Optimization
```rust
// Enable bloom filters for high-cardinality columns
use parquet::file::properties::WriterProperties;

let props = WriterProperties::builder()
    .set_bloom_filter_enabled(true)
    .set_bloom_filter_fpp(0.01)  // 1% false positive rate
    .build();
```

### 4. Nested Data Optimization
```python
# Optimize nested structures
def optimize_nested_schema(table):
    # Flatten nested columns if frequently accessed together
    # Keep nested if accessed as a unit
    
    if should_flatten(table.schema):
        return flatten_schema(table)
    return table
```

## Common Challenges & Solutions

### Challenge 1: Large File Memory Usage
**Problem**: Loading large Parquet files into memory
**Solution**: Use streaming readers, process row groups individually
```rust
// Stream row groups instead of loading entire file
let file = File::open(path)?;
let reader = SerializedFileReader::new(file)?;

for i in 0..reader.metadata().num_row_groups() {
    let row_group = reader.get_row_group(i)?;
    // Process one row group at a time
    process_row_group(row_group);
}
```

### Challenge 2: Accurate Benchmarking
**Problem**: Inconsistent performance measurements
**Solution**: Run multiple iterations, warm up caches, use median values
```python
import time
import statistics

def benchmark_operation(operation, iterations=5):
    times = []
    
    # Warm-up run
    operation()
    
    # Actual measurements
    for _ in range(iterations):
        start = time.perf_counter()
        operation()
        times.append(time.perf_counter() - start)
    
    return {
        'median': statistics.median(times),
        'mean': statistics.mean(times),
        'stdev': statistics.stdev(times)
    }
```

### Challenge 3: Compression Trade-offs
**Problem**: Choosing the right compression for use case
**Solution**: Provide multiple recommendations based on priorities
```python
def recommend_compression(results, priority='balanced'):
    if priority == 'storage':
        # Best compression ratio
        return min(results, key=lambda r: r.file_size)
    elif priority == 'speed':
        # Fastest read time
        return min(results, key=lambda r: r.read_time)
    elif priority == 'write_speed':
        # Fastest write time
        return min(results, key=lambda r: r.write_time)
    else:  # balanced
        # Score based on multiple factors
        return calculate_balanced_score(results)
```

### Challenge 4: Schema Evolution
**Problem**: Optimizing files with evolving schemas
**Solution**: Analyze schema compatibility and suggest migration strategies
```rust
fn check_schema_compatibility(old_schema: &Schema, new_schema: &Schema) -> bool {
    // Check if schemas are compatible for merging
    // Allow adding columns, but not removing or changing types
    for field in old_schema.fields() {
        if !new_schema.contains(field.name()) {
            return false;
        }
    }
    true
}
```

## Next Steps

After completing this project:

1. **Add to Portfolio**: Document on GitHub with benchmark results
2. **Write Blog Post**: "Optimizing Parquet Files: A Data-Driven Approach"
3. **Extend Features**: Add encoding optimization, predicate pushdown analysis
4. **Build Project C**: Continue with Iceberg Table Manager
5. **Create Tool**: Package as reusable CLI tool for teams

## Troubleshooting

### Build Issues

**Issue**: Parquet crate compilation errors
```bash
# Solution: Update to latest version
cargo update
cargo build --release
```

**Issue**: Arrow version conflicts
```toml
# Solution: Ensure matching versions in Cargo.toml
[dependencies]
parquet = "49.0"
arrow = "49.0"  # Must match parquet version
```

### Runtime Issues

**Issue**: "Invalid Parquet file" error
```rust
// Solution: Validate file before processing
use parquet::file::reader::FileReader;

fn validate_parquet(path: &str) -> Result<(), Error> {
    let file = File::open(path)?;
    let reader = SerializedFileReader::new(file)?;
    let metadata = reader.metadata();
    
    if metadata.num_row_groups() == 0 {
        return Err(Error::new("Empty Parquet file"));
    }
    Ok(())
}
```

**Issue**: Out of memory when processing large files
```python
# Solution: Process in chunks
import pyarrow.parquet as pq

# Read in batches
parquet_file = pq.ParquetFile('large_file.parquet')
for batch in parquet_file.iter_batches(batch_size=100000):
    process_batch(batch)
```

### Performance Issues

**Issue**: Slow compression testing
```rust
// Solution: Use parallel processing
use rayon::prelude::*;

let compressions = vec!["SNAPPY", "ZSTD", "LZ4", "GZIP"];
let results: Vec<_> = compressions
    .par_iter()
    .map(|comp| test_compression(comp))
    .collect();
```

**Issue**: Inaccurate benchmarks
```python
# Solution: Clear OS cache between tests (Linux)
import os
import subprocess

def clear_cache():
    subprocess.run(['sync'])
    with open('/proc/sys/vm/drop_caches', 'w') as f:
        f.write('3')  # Requires sudo
```

## Real-World Use Cases

### Use Case 1: Data Lake Optimization
**Scenario**: Optimize 10TB data lake with daily ingestion
**Solution**:
- Use ZSTD for cold storage (4x compression)
- Use Snappy for hot data (fast queries)
- 256MB row groups for optimal query performance
- Enable bloom filters on join keys

### Use Case 2: Real-Time Analytics
**Scenario**: Low-latency queries on streaming data
**Solution**:
- Use LZ4 compression (fastest read)
- Smaller row groups (128MB) for faster first-byte
- Dictionary encoding for categorical columns
- Partition by time for efficient pruning

### Use Case 3: Archive Storage
**Scenario**: Long-term storage with infrequent access
**Solution**:
- Use ZSTD level 9 (maximum compression)
- Large row groups (1GB) to reduce metadata
- Disable bloom filters to save space
- Coalesce small files to reduce overhead

### Use Case 4: Machine Learning Training
**Scenario**: Fast data loading for model training
**Solution**:
- Use Snappy (balanced performance)
- Partition by training/validation split
- Store features in optimal order
- Use dictionary encoding for labels

## Comparison with Other Tools

### vs Parquet-tools
- **Parquet-tools**: Java-based, basic analysis
- **This Tool**: Rust-based, optimization focus, performance testing
- **Advantage**: Automated recommendations, compression testing

### vs DuckDB COPY
- **DuckDB**: SQL interface, good for queries
- **This Tool**: Specialized optimization, detailed analysis
- **Use Together**: Use this for optimization, DuckDB for querying

### vs Spark
- **Spark**: Distributed processing, production workloads
- **This Tool**: Single-node optimization, development/testing
- **Advantage**: Faster iteration, detailed metrics

## Deployment Options

### 1. Standalone CLI Tool
```bash
# Install globally
cargo install --path .

# Use anywhere
parquet-optimizer analyze /data/file.parquet
```

### 2. Docker Container
```dockerfile
FROM rust:1.75 as builder
WORKDIR /app
COPY . .
RUN cargo build --release

FROM debian:bookworm-slim
COPY --from=builder /app/target/release/parquet-optimizer /usr/local/bin/
ENTRYPOINT ["parquet-optimizer"]
```

### 3. CI/CD Integration
```yaml
# GitHub Actions
name: Parquet Optimization Check
on: [pull_request]
jobs:
  optimize:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Analyze Parquet files
        run: |
          parquet-optimizer analyze data/*.parquet
          # Fail if files are not optimally compressed
```

### 4. Python Library
```python
# Create Python bindings with PyO3
import parquet_optimizer_rs

results = parquet_optimizer_rs.analyze("data.parquet")
recommendations = parquet_optimizer_rs.recommend(results)
```

## Resources

### Documentation
- [Apache Parquet Documentation](https://parquet.apache.org/docs/) - Official format spec
- [PyArrow Parquet](https://arrow.apache.org/docs/python/parquet.html) - Python API
- [Parquet Rust Crate](https://docs.rs/parquet/) - Rust implementation
- [Compression Algorithms Comparison](https://github.com/facebook/zstd#benchmarks) - Performance data

### Tutorials
- [Parquet Performance Tuning](https://www.databricks.com/blog/2022/08/22/parquet-performance-tuning.html) - Databricks guide
- [Understanding Parquet](https://blog.twitter.com/engineering/en_us/a/2013/dremel-made-simple-with-parquet) - Twitter engineering
- [Columnar Storage](https://www.youtube.com/watch?v=1j8SdS7s_NY) - Video explanation

### Tools
- [parquet-tools](https://github.com/apache/parquet-mr/tree/master/parquet-tools) - Java CLI
- [parquet-cli](https://github.com/chhantyal/parquet-cli) - Python CLI
- [DuckDB](https://duckdb.org/) - Query Parquet files

## Questions?

If you get stuck:
1. Review the `tech-spec.md` for detailed architecture
2. Check Parquet documentation for format details
3. Search for compression algorithm benchmarks
4. Review the 100 Days bootcamp materials on data formats
5. Test with small files first before large datasets
6. Compare results with parquet-tools for validation

## Related Projects

After completing this project, consider:
- **Project C**: Iceberg Manager - Work with table formats
- **Project D**: Delta Operations - Optimize Delta Lake tables
- **Project H**: Format Benchmark - Compare all formats
- Build a data lake optimization pipeline using your tool
