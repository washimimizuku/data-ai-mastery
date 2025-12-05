# Project H: Data Format Performance Benchmark

## Overview

Build a comprehensive benchmarking tool comparing CSV, JSON, Parquet, Avro, Arrow, Iceberg, and Delta Lake formats across read/write performance, compression ratios, and query speed in Python and Rust.

**What You'll Build**: A dual-language benchmarking suite that measures performance metrics for all major data formats and produces comparison reports with visualizations.

**What You'll Learn**: Data format trade-offs, columnar vs row-based storage, compression algorithms, performance benchmarking methodology, and data-driven format selection.

## Time Estimate

**1-2 days (8-16 hours)**

### Day 1 (8 hours)
- Hour 1: Test data generation (10M rows)
- Hours 2-3: Python benchmarks (CSV, JSON, Parquet)
- Hours 4-5: Rust benchmarks (CSV, JSON, Parquet)
- Hours 6-7: Extended formats (Avro, Arrow, Iceberg, Delta)
- Hour 8: Initial results collection

### Day 2 (8 hours)
- Hours 1-2: Advanced operations (partial read, filters)
- Hours 3-4: Visualizations (charts and graphs)
- Hours 5-6: CLI tool
- Hours 7-8: Report generation and documentation

## Prerequisites

### Required Knowledge
- [100 Days Data & AI](https://github.com/washimimizuku/100-days-data-ai) - Days 1-20
  - Days 1-10: Data formats (CSV, JSON, Parquet, Avro)
  - Days 11-20: Table formats (Iceberg, Delta Lake)
- [30 Days of Rust](https://github.com/washimimizuku/30-days-rust-data-ai) - Days 1-20 (optional)

### Technical Requirements
- Python 3.11+ installed
- Rust 1.75+ (optional for Rust benchmarks)
- Understanding of data formats
- Basic statistics knowledge

### Tools
- Python: pandas, pyarrow, fastavro, deltalake
- Rust: csv, parquet, arrow crates (optional)
- 5-10 GB free disk space
- Git for version control

## Getting Started

### Step 1: Review Documentation
1. `prd.md` - Product requirements and goals
2. `tech-spec.md` - Technical architecture and design
3. `implementation-plan.md` - Step-by-step build guide

### Step 2: Set Up Environment
```bash
# Python dependencies
pip install pandas pyarrow fastavro deltalake pyiceberg matplotlib plotly

# Optional: Rust setup
cargo new format-benchmark
```

Add to `Cargo.toml` (optional):
```toml
[dependencies]
csv = "1.3"
serde = { version = "1.0", features = ["derive"] }
serde_json = "1.0"
parquet = "49.0"
arrow = "49.0"
```

### Step 3: Follow Implementation Plan
Build components in order:
1. **Data Generator** (Hour 1): Create 10M row test dataset
2. **Python Benchmarks** (Hours 2-3): CSV, JSON, Parquet
3. **Rust Benchmarks** (Hours 4-5): Same formats in Rust
4. **Extended Formats** (Hours 6-7): Avro, Arrow, Iceberg, Delta
5. **Visualization** (Day 2): Charts and reports

## Core Components

### 1. Data Generator
```python
# src/data_generator.py
import pandas as pd
import numpy as np

def generate_test_data(num_rows=10_000_000):
    """Generate realistic test dataset"""
    data = {
        'transaction_id': range(num_rows),
        'user_id': np.random.randint(1, 100_000, num_rows),
        'timestamp': [...],  # datetime values
        'product_id': np.random.randint(1, 10_000, num_rows),
        'quantity': np.random.randint(1, 10, num_rows),
        'price': np.random.uniform(10, 1000, num_rows),
        'category': np.random.choice(['Electronics', 'Clothing', ...], num_rows),
        'region': np.random.choice(['US', 'EU', 'ASIA', 'LATAM'], num_rows),
        'payment_method': np.random.choice(['Credit Card', ...], num_rows),
        'discount': np.random.uniform(0, 0.3, num_rows),
    }
    return pd.DataFrame(data)
```

### 2. Benchmark Runner
```python
# src/python/benchmarks.py
import time
import os

@dataclass
class BenchmarkResult:
    format: str
    operation: str
    time_seconds: float
    file_size_mb: float
    rows: int
    language: str = "Python"

class FormatBenchmark:
    def benchmark_csv(self):
        # Write
        start = time.time()
        self.df.to_csv('output.csv', index=False)
        write_time = time.time() - start
        
        # Read
        start = time.time()
        df_read = pd.read_csv('output.csv')
        read_time = time.time() - start
        
        return results
    
    def benchmark_parquet(self):
        # Write, Read, Partial Read
    
    def benchmark_json(self):
        # Write, Read
    
    def benchmark_avro(self):
        # Write, Read
```

### 3. Rust Benchmarks (Optional)
```rust
// src/rust/src/benchmarks/csv.rs
use std::time::Instant;

pub fn benchmark_csv_read(path: &str) -> BenchmarkResult {
    let start = Instant::now();
    
    let mut reader = csv::Reader::from_path(path)?;
    let mut count = 0;
    for result in reader.records() {
        count += 1;
    }
    
    BenchmarkResult {
        time_ms: start.elapsed().as_millis(),
        rows: count,
    }
}
```

### 4. Visualizations
```python
# src/visualizer.py
import matplotlib.pyplot as plt

def create_comparison_charts(results):
    # 1. Write performance bar chart
    # 2. Read performance bar chart
    # 3. File size comparison
    # 4. Compression ratio vs CSV
    
    plt.savefig('results/benchmark_comparison.png')
```

## Key Features

### 1. Format Support
- CSV (baseline)
- JSON (API standard)
- Parquet (analytics standard)
- Avro (streaming standard)
- Arrow (in-memory)
- Iceberg (modern table format)
- Delta Lake (modern table format)

### 2. Operations Benchmarked
- Write: Serialize and write to disk
- Read: Load entire dataset
- Partial Read: Read specific columns (columnar formats)
- Filter: Read with predicate pushdown
- Compression: File size comparison

### 3. Language Implementations
- Python: pandas, pyarrow, delta-rs, pyiceberg
- Rust: csv, serde_json, parquet, arrow-rs (optional)

### 4. Metrics Collected
- Execution time (milliseconds)
- Memory usage (MB)
- File size (MB)
- Compression ratio
- Speedup factor (Rust vs Python)

### 5. Visualizations
- Bar charts for performance comparison
- File size comparison
- Rust vs Python speedup charts
- Operation breakdown

### 6. CLI Interface
```bash
# Run all benchmarks
format-benchmark run --all

# Specific format
format-benchmark run --format parquet

# Compare languages
format-benchmark compare --formats parquet,csv --languages python,rust

# Generate report
format-benchmark report --output results/
```

## Success Criteria

- [ ] Test data generator (10M rows)
- [ ] All 7 formats benchmarked in Python
- [ ] Rust implementations (4+ formats, optional)
- [ ] Write, read, and partial read operations measured
- [ ] Performance metrics collected (time, size, memory)
- [ ] Compression ratios calculated
- [ ] Visualizations generated (4+ charts)
- [ ] Rust 15-20x faster demonstrated
- [ ] Comprehensive report with recommendations
- [ ] CLI interface working
- [ ] Code < 800 lines

## Learning Outcomes

- Understand data format trade-offs
- Benchmark performance accurately
- Choose appropriate formats for use cases
- Explain columnar vs row-based storage
- Compare compression algorithms
- Measure and optimize data pipelines

## Performance Targets

**Write Performance** (10M rows, from tech-spec):
| Format | Python | Rust | File Size | Compression |
|--------|--------|------|-----------|-------------|
| CSV | 45s | 2.1s | 1.2 GB | 1.0x |
| JSON | 52s | 3.2s | 1.8 GB | 0.67x |
| Parquet | 18s | 1.2s | 280 MB | 4.3x |
| Avro | 22s | 1.5s | 320 MB | 3.8x |
| Arrow | 5s | 0.3s | N/A | N/A |

**Read Performance** (10M rows):
| Format | Python | Rust | Speedup |
|--------|--------|------|---------|
| CSV | 38s | 1.8s | 21x |
| Parquet | 8s | 0.4s | 20x |
| Avro | 12s | 0.6s | 20x |

**Partial Read** (3 columns):
- CSV: 38s (no benefit)
- Parquet: 2.5s (15x faster)
- Avro: 12s (no benefit)

## Project Structure

```
project-h-format-benchmark/
├── src/
│   ├── data_generator.py
│   ├── python/
│   │   ├── benchmarks.py
│   │   └── formats/
│   │       ├── csv_bench.py
│   │       ├── parquet_bench.py
│   │       └── avro_bench.py
│   ├── rust/
│   │   └── src/
│   │       ├── main.rs
│   │       └── benchmarks/
│   └── visualizer.py
├── data/
│   ├── test_data.csv
│   └── outputs/
├── results/
│   ├── benchmark_results.json
│   ├── charts/
│   └── REPORT.md
├── cli.py
├── prd.md
├── tech-spec.md
├── implementation-plan.md
└── README.md
```

## Common Challenges

### Inconsistent Benchmarks
Run multiple iterations, use median, warm up caches before measuring

### Memory Constraints
Use smaller test sets if needed, stream processing for large files

### Fair Comparison
Benchmark common operations, note format-specific advantages (e.g., columnar partial reads)

## Next Steps

1. Add to portfolio with performance charts
2. Write blog post: "Choosing the Right Data Format: A Benchmark Study"
3. Extend with more formats (ORC, Feather)
4. Apply learnings to production pipelines

## Resources

- [Apache Parquet](https://parquet.apache.org/)
- [Apache Avro](https://avro.apache.org/)
- [Apache Arrow](https://arrow.apache.org/)
- [Delta Lake](https://delta.io/)
- [Apache Iceberg](https://iceberg.apache.org/)
