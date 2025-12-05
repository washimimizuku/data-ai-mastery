# Project G: Data Quality Profiler

## Overview

Build a high-performance data profiling tool in Rust with Python bindings that generates comprehensive statistical summaries, detects patterns, calculates quality metrics, and creates HTML/JSON reports.

**What You'll Build**: A Rust library with PyO3 bindings that profiles CSV/Parquet files 20x faster than pandas-profiling, with automatic pattern detection and quality analysis.

**What You'll Learn**: Statistical analysis in Rust, pattern detection with regex, PyO3 for Python interop, HTML report generation, and performance optimization for data processing.

## Time Estimate

**1-2 days (8-16 hours)**

### Day 1 (8 hours)
- Hours 1-2: Core profiler setup and data loading
- Hours 3-4: Statistics calculator (mean, median, percentiles)
- Hours 5-6: Pattern detection (email, phone, UUID)
- Hours 7-8: Quality metrics (completeness, uniqueness, validity)

### Day 2 (8 hours)
- Hours 1-2: HTML report generation
- Hours 3-4: PyO3 bindings and Python API
- Hours 5-6: CLI interface
- Hours 7-8: Benchmarking and documentation

## Prerequisites

### Required Knowledge
- [30 Days of Rust](https://github.com/washimimizuku/30-days-rust-data-ai) - Days 1-25
  - Days 1-15: Rust basics and data structures
  - Days 16-20: File I/O and parsing
  - Days 21-25: Performance optimization
- [100 Days Data & AI](https://github.com/washimimizuku/100-days-data-ai) - Days 41-50
  - Data quality concepts and Great Expectations

### Technical Requirements
- Rust 1.75+ installed
- Python 3.11+ installed
- Understanding of statistics (mean, median, std dev, percentiles)
- Basic HTML/CSS knowledge

### Tools
- Rust toolchain with maturin
- Python with pandas (for comparison)
- Git for version control

## Getting Started

### Step 1: Review Documentation
1. `prd.md` - Product requirements and goals
2. `tech-spec.md` - Technical architecture and design
3. `implementation-plan.md` - Step-by-step build guide

### Step 2: Initialize Project
```bash
cargo new --lib data-profiler
cd data-profiler
pip install maturin
```

Add to `Cargo.toml`:
```toml
[dependencies]
pyo3 = { version = "0.20", features = ["extension-module"] }
polars = { version = "0.36", features = ["lazy", "parquet"] }
regex = "1.10"
serde = { version = "1.0", features = ["derive"] }
serde_json = "1.0"
tera = "1.19"
rayon = "1.8"

[lib]
name = "data_profiler_rs"
crate-type = ["cdylib"]
```

### Step 3: Follow Implementation Plan
Build components in order:
1. **Core Profiler** (Hours 1-2): Data loading and structures
2. **Statistics** (Hours 3-4): Calculate metrics
3. **Pattern Detection** (Hours 5-6): Regex-based detection
4. **Quality Metrics** (Hours 7-8): Completeness, uniqueness, validity
5. **Reports** (Day 2): HTML/JSON generation and PyO3 bindings

## Core Components

### 1. Data Structures
```rust
// src/lib.rs
#[pyclass]
pub struct DataProfile {
    pub row_count: usize,
    pub column_count: usize,
    pub columns: Vec<ColumnProfile>,
}

#[pyclass]
pub struct ColumnProfile {
    pub name: String,
    pub data_type: String,
    pub statistics: Statistics,
    pub quality: QualityMetrics,
    pub patterns: Vec<String>,
}

#[pyclass]
pub struct Statistics {
    pub count: usize,
    pub null_count: usize,
    pub unique_count: usize,
    pub mean: Option<f64>,
    pub median: Option<f64>,
    pub std: Option<f64>,
    // ... percentiles
}

#[pyclass]
pub struct QualityMetrics {
    pub completeness: f64,  // % non-null
    pub uniqueness: f64,    // % unique
    pub validity: f64,      // % matching pattern
}
```

### 2. Profiler Functions
```rust
#[pyfunction]
fn profile_csv(path: String) -> PyResult<DataProfile> {
    // Load CSV with Polars
    // Profile each column
}

#[pyfunction]
fn profile_parquet(path: String) -> PyResult<DataProfile> {
    // Load Parquet with Polars
    // Profile each column
}

fn calculate_statistics(series: &Series) -> Statistics {
    // Calculate count, nulls, unique
    // For numeric: mean, median, std, min, max, percentiles
}

fn detect_patterns(series: &Series) -> Vec<String> {
    // Regex matching for email, phone, UUID, URL
    // Return patterns with >80% match rate
}

fn calculate_quality_metrics(series: &Series) -> QualityMetrics {
    // Completeness: % non-null
    // Uniqueness: % unique
    // Validity: % matching expected pattern
}
```

### 3. Python API
```python
# python/data_profiler/__init__.py
import data_profiler_rs

class DataProfiler:
    def __init__(self, file_path, format='csv'):
        if format == 'csv':
            self.profile = data_profiler_rs.profile_csv(file_path)
        else:
            self.profile = data_profiler_rs.profile_parquet(file_path)
    
    def summary(self):
        # Print overview and column details
    
    def to_html(self, output_path):
        # Generate HTML report
    
    def to_json(self, output_path):
        # Export JSON report
```

### 4. Build and Test
```bash
# Build Rust extension
maturin develop --release

# Test in Python
python -c "
import data_profiler
profiler = data_profiler.DataProfiler('data.csv')
profiler.summary()
profiler.to_html('report.html')
"
```

## Key Features

### 1. Statistical Profiling
- Column-level statistics (mean, median, std, min, max)
- Percentiles (p25, p50, p75, p95, p99)
- Cardinality and uniqueness
- Null/missing value analysis
- Data type inference

### 2. Pattern Detection
- Email addresses
- Phone numbers
- Dates and timestamps
- URLs
- UUIDs and numeric IDs
- Custom regex patterns

### 3. Data Quality Checks
- Completeness (% non-null)
- Validity (format compliance)
- Consistency (type consistency)
- Uniqueness (duplicate detection)
- Distribution analysis

### 4. Report Generation
- HTML reports with styling
- JSON export for programmatic use
- Summary statistics table
- Column-by-column analysis
- Recommendations

### 5. Python Integration
- PyO3 bindings
- Simple Python API
- Easy installation with maturin

### 6. CLI Interface
```bash
# Profile CSV
data-profiler profile data.csv

# Generate HTML report
data-profiler profile data.csv --output report.html

# Generate JSON report
data-profiler profile data.csv --output report.json --format json

# Profile specific columns
data-profiler profile data.csv --columns age,email,salary
```

## Success Criteria

- [ ] Calculate all statistics (mean, median, std, percentiles)
- [ ] Detect patterns (email, phone, UUID, URL)
- [ ] Calculate quality metrics (completeness, uniqueness, validity)
- [ ] Generate HTML reports with styling
- [ ] Export JSON reports
- [ ] PyO3 bindings working from Python
- [ ] CLI with profile command
- [ ] 20x+ faster than pandas-profiling
- [ ] Profile 10M rows in < 30 seconds
- [ ] Code < 600 lines
- [ ] Unit tests passing

## Learning Outcomes

- Calculate statistical metrics in Rust with Polars
- Detect patterns with regex efficiently
- Build PyO3 Python bindings
- Generate HTML reports programmatically
- Optimize data processing performance
- Profile large datasets quickly
- Understand data quality dimensions

## Performance Targets

**Profiling Speed** (from PRD):
- 10M+ rows profiled in seconds
- 10-50x faster than pandas-profiling
- Profile 10M rows in < 30 seconds

**Benchmarks**:
```
Rust Data Profiler:
  1M rows, 10 cols: 3 seconds
  10M rows, 10 cols: 28 seconds

pandas-profiling:
  1M rows: 65 seconds
  10M rows: 580 seconds

Speedup: 20-21x faster
Memory: 83% less than pandas-profiling
```

## Project Structure

```
project-g-data-profiler/
├── rust/
│   ├── src/
│   │   ├── lib.rs         # PyO3 module
│   │   ├── profiler.rs    # Core profiling
│   │   ├── statistics.rs  # Stats calculation
│   │   ├── patterns.rs    # Pattern detection
│   │   └── quality.rs     # Quality metrics
│   ├── templates/
│   │   └── report.html.tera
│   ├── Cargo.toml
│   └── tests/
├── python/
│   ├── data_profiler/
│   │   └── __init__.py    # Python API
│   ├── examples/
│   │   └── usage.py
│   └── tests/
├── benches/
│   └── profiler_bench.rs
├── prd.md
├── tech-spec.md
├── implementation-plan.md
└── README.md
```

## Common Challenges

### Large Dataset Memory
Use sampling for pattern detection (1000 rows), stream processing for statistics

### Pattern Detection Accuracy
Require high match threshold (80%+), test with sample data first

### PyO3 Type Conversions
Use PyO3's built-in conversions, avoid unnecessary clones, leverage `#[pyo3(get)]`

## Next Steps

1. Add to portfolio with performance benchmarks
2. Write blog post: "Building a 20x Faster Data Profiler in Rust"
3. Extend with correlation analysis and visualizations
4. Continue to Project H: Format Benchmark

## Resources

- [Polars Docs](https://pola-rs.github.io/polars-book/)
- [PyO3 Guide](https://pyo3.rs/)
- [Regex in Rust](https://docs.rs/regex/)
- [Tera Templates](https://tera.netlify.app/)
