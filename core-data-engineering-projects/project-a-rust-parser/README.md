# Project A: Rust Data Parser Benchmark

## Objective

Build a high-performance data parser in Rust with Python bindings (PyO3), demonstrating 10-50x performance improvements over pure Python/Pandas implementations.

**What You'll Build**: A Rust library that parses CSV/JSON files at blazing speed, with Python bindings that make it easy to use from Python code, plus comprehensive benchmarks proving the performance gains.

**What You'll Learn**: Rust for data processing, PyO3 for Python interop, performance optimization, memory management, and how to create production-ready Rust libraries for Python.

## Time Estimate

**1 day (8 hours)**

- Hours 1-2: Rust parser implementation (CSV/JSON)
- Hours 3-4: PyO3 bindings and Python API
- Hours 5-6: Benchmark suite and data generation
- Hours 7-8: Performance analysis and documentation

## Prerequisites

### Required Knowledge
Complete these bootcamp sections first:
- [30 Days of Rust](https://github.com/washimimizuku/30-days-rust-data-ai) - Days 1-20
  - Days 1-10: Rust basics
  - Days 11-15: Data structures and error handling
  - Days 16-20: File I/O and parsing
- [30 Days of Python](https://github.com/washimimizuku/30-days-python-data-ai) - Days 1-15
  - Days 1-10: Python basics
  - Days 11-15: Pandas fundamentals

### Technical Requirements
- Rust 1.75+ installed
- Python 3.11+ installed
- Understanding of CSV/JSON formats
- Basic knowledge of performance benchmarking

### Tools Needed
- Rust toolchain (rustc, cargo)
- maturin (for building Python wheels)
- Python with pandas, matplotlib
- Git for version control

## Getting Started

### Step 1: Review Documentation
Read the project documents in order:
1. `prd.md` - Understand what you're building
2. `tech-spec.md` - Review technical architecture
3. `implementation-plan.md` - Follow the implementation steps

### Step 2: Set Up Rust + PyO3 Environment
```bash
# Install Rust (if not already installed)
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# Create new Rust project
cargo new --lib rust_parser
cd rust_parser

# Add dependencies to Cargo.toml
[dependencies]
pyo3 = { version = "0.20", features = ["extension-module"] }
csv = "1.3"
serde = { version = "1.0", features = ["derive"] }
serde_json = "1.0"

[lib]
name = "data_parser_rs"
crate-type = ["cdylib"]

# Install maturin for building Python wheels
pip install maturin
```

### Step 3: Build Rust Parser
```rust
// src/lib.rs
use pyo3::prelude::*;
use csv::ReaderBuilder;
use std::fs::File;

#[pyfunction]
fn parse_csv(path: &str) -> PyResult<Vec<Vec<String>>> {
    let file = File::open(path)?;
    let mut reader = ReaderBuilder::new().from_reader(file);
    
    let mut data = Vec::new();
    for result in reader.records() {
        let record = result?;
        let row: Vec<String> = record.iter().map(|s| s.to_string()).collect();
        data.push(row);
    }
    Ok(data)
}

#[pymodule]
fn data_parser_rs(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(parse_csv, m)?)?;
    Ok(())
}
```

### Step 4: Build and Test
```bash
# Build the Python wheel
maturin develop

# Test in Python
python -c "import data_parser_rs; print(data_parser_rs.parse_csv('test.csv'))"
```

### Step 5: Create Benchmark Suite
```python
# benchmarks/run_benchmarks.py
import time
import pandas as pd
import data_parser_rs

def benchmark_csv_parsing(file_path, rows):
    # Rust parser
    start = time.time()
    rust_data = data_parser_rs.parse_csv(file_path)
    rust_time = time.time() - start
    
    # Pandas parser
    start = time.time()
    pandas_data = pd.read_csv(file_path)
    pandas_time = time.time() - start
    
    speedup = pandas_time / rust_time
    print(f"CSV Parsing ({rows:,} rows):")
    print(f"  Rust:   {rust_time:.2f}s")
    print(f"  Pandas: {pandas_time:.2f}s")
    print(f"  Speedup: {speedup:.1f}x")
```

## Key Features to Implement

### 1. CSV Parser
- Fast CSV reading with configurable delimiter
- Header detection and handling
- Type inference for columns
- Memory-efficient streaming for large files

### 2. JSON Parser
- Parse nested JSON structures
- Handle arrays and objects
- Streaming parser for large files
- Error handling for malformed JSON

### 3. Basic Operations
- Filter rows based on conditions
- Aggregate data (sum, count, average)
- Group by operations
- Sort operations

### 4. PyO3 Python Bindings
- Clean Python API
- Pandas-compatible return types
- Error handling with Python exceptions
- Documentation strings

### 5. Benchmark Suite
- Multiple dataset sizes (1M, 10M, 50M rows)
- CSV and JSON parsing benchmarks
- Operation benchmarks (filter, aggregate)
- Memory usage tracking
- Performance visualization

## Success Criteria

By the end of this project, you should have:

- [ ] Rust parser handling CSV and JSON files
- [ ] PyO3 bindings with clean Python API
- [ ] Installable Python package (pip install)
- [ ] Benchmark suite comparing Rust vs Pandas
- [ ] 10-50x speedup demonstrated
- [ ] Memory usage comparison documented
- [ ] Performance charts generated
- [ ] Comprehensive README with usage examples
- [ ] GitHub repository with clean code
- [ ] Unit tests for Rust code

## Learning Outcomes

After completing this project, you'll be able to:

- Write high-performance Rust code for data processing
- Create Python bindings using PyO3
- Build and distribute Rust-based Python packages
- Benchmark and profile code performance
- Optimize memory usage in data processing
- Explain when to use Rust vs Python for data tasks
- Package Rust libraries for Python consumption

## Expected Performance

Based on typical results:

**CSV Parsing (10M rows, 10 columns)**:
- Rust: 2-3 seconds
- Pandas: 40-50 seconds
- Speedup: 15-20x

**JSON Parsing (1M nested objects)**:
- Rust: 1-2 seconds
- Pandas: 25-30 seconds
- Speedup: 20-25x

**Memory Usage**:
- Rust: 40-60% less memory than Pandas
- Streaming capability for files larger than RAM

## Project Structure

```
project-a-rust-parser/
├── rust/
│   ├── src/
│   │   ├── lib.rs           # PyO3 module definition
│   │   ├── parser.rs        # CSV/JSON parsers
│   │   └── operations.rs    # Filter, aggregate, etc.
│   ├── Cargo.toml
│   └── tests/
│       └── parser_tests.rs
├── python/
│   ├── benchmarks/
│   │   ├── run_benchmarks.py
│   │   ├── generate_data.py
│   │   └── visualize.py
│   ├── examples/
│   │   └── usage_example.py
│   └── tests/
│       └── test_bindings.py
├── results/
│   ├── charts/
│   │   ├── parsing_time.png
│   │   └── memory_usage.png
│   └── benchmark_results.md
├── prd.md
├── tech-spec.md
├── implementation-plan.md
└── README.md
```

## Performance Optimization Tips

### 1. Use Zero-Copy Parsing
```rust
// Instead of allocating new strings
let value: &str = record.get(0).unwrap();

// Use string slices when possible
fn process_field(field: &str) -> Result<i32, ParseError> {
    field.parse::<i32>()
}
```

### 2. Batch Processing
```rust
// Process in chunks for better cache locality
const BATCH_SIZE: usize = 10_000;

for chunk in data.chunks(BATCH_SIZE) {
    process_batch(chunk);
}
```

### 3. Parallel Processing
```rust
use rayon::prelude::*;

// Parallel CSV parsing
let results: Vec<_> = files
    .par_iter()
    .map(|file| parse_csv(file))
    .collect();
```

### 4. Memory-Mapped Files
```rust
use memmap2::Mmap;

// For very large files
let file = File::open(path)?;
let mmap = unsafe { Mmap::map(&file)? };
let data = &mmap[..];
```

## Common Challenges & Solutions

### Challenge 1: PyO3 Type Conversions
**Problem**: Converting Rust types to Python types efficiently
**Solution**: Use PyO3's built-in conversions, avoid unnecessary copies
```rust
// Efficient: Return Vec directly
#[pyfunction]
fn parse_csv(path: &str) -> PyResult<Vec<Vec<String>>> {
    // PyO3 handles conversion automatically
}

// For large data, consider returning iterators
#[pyfunction]
fn parse_csv_iter(path: &str) -> PyResult<CsvIterator> {
    // Return custom iterator type
}
```

### Challenge 2: Memory Management
**Problem**: Large files consuming too much memory
**Solution**: Implement streaming parsers, process in chunks
```rust
#[pyfunction]
fn parse_csv_streaming(path: &str, chunk_size: usize) -> PyResult<Vec<Vec<String>>> {
    let mut reader = ReaderBuilder::new().from_path(path)?;
    let mut chunks = Vec::new();
    
    for chunk in reader.records().chunks(chunk_size) {
        // Process chunk
        chunks.push(process_chunk(chunk));
    }
    Ok(chunks)
}
```

### Challenge 3: Error Handling
**Problem**: Rust errors need to map to Python exceptions
**Solution**: Use PyO3's error conversion, create custom exception types
```rust
use pyo3::exceptions::PyValueError;

#[pyfunction]
fn parse_csv(path: &str) -> PyResult<Vec<Vec<String>>> {
    let file = File::open(path)
        .map_err(|e| PyValueError::new_err(format!("Cannot open file: {}", e)))?;
    // ...
}
```

### Challenge 4: String Encoding
**Problem**: Handling different character encodings
**Solution**: Use encoding_rs crate for robust encoding detection
```rust
use encoding_rs::*;

fn detect_and_decode(bytes: &[u8]) -> String {
    let (decoded, _, _) = UTF_8.decode(bytes);
    decoded.into_owned()
}
```

## Troubleshooting

### Build Errors

**Issue**: `maturin develop` fails with "Rust compiler not found"
```bash
# Solution: Ensure Rust is in PATH
source $HOME/.cargo/env
rustc --version
```

**Issue**: PyO3 version mismatch
```bash
# Solution: Update dependencies
cargo update
maturin develop --release
```

### Runtime Errors

**Issue**: `ImportError: cannot import name 'data_parser_rs'`
```bash
# Solution: Rebuild the module
maturin develop
# Or install in editable mode
pip install -e .
```

**Issue**: Segmentation fault when parsing large files
```rust
// Solution: Increase stack size or use heap allocation
// In Cargo.toml:
[profile.release]
opt-level = 3
lto = true
```

### Performance Issues

**Issue**: Rust parser not significantly faster than Pandas
```rust
// Solution: Enable release mode optimizations
maturin develop --release

// Check compiler optimizations
[profile.release]
opt-level = 3
lto = true
codegen-units = 1
```

**Issue**: High memory usage
```rust
// Solution: Use streaming instead of loading all data
// Process in chunks, don't collect all results at once
```

## Next Steps

After completing this project:

1. **Add to Portfolio**: Document on GitHub with performance charts
2. **Write Blog Post**: "Why Rust is 20x Faster than Python for Data Parsing"
3. **Extend Features**: Add Parquet support, more operations
4. **Build Project B**: Continue with Parquet Optimizer
5. **Publish Package**: Release on PyPI for others to use

## Advanced Features (Optional Extensions)

### 1. Type Inference
```rust
// Automatically detect column types
pub enum ColumnType {
    Integer,
    Float,
    String,
    DateTime,
}

fn infer_type(samples: &[&str]) -> ColumnType {
    // Try parsing as different types
    if samples.iter().all(|s| s.parse::<i64>().is_ok()) {
        ColumnType::Integer
    } else if samples.iter().all(|s| s.parse::<f64>().is_ok()) {
        ColumnType::Float
    } else {
        ColumnType::String
    }
}
```

### 2. Parallel File Processing
```rust
use rayon::prelude::*;

#[pyfunction]
fn parse_multiple_files(paths: Vec<String>) -> PyResult<Vec<Vec<Vec<String>>>> {
    let results: Vec<_> = paths
        .par_iter()
        .map(|path| parse_csv(path))
        .collect::<Result<Vec<_>, _>>()?;
    Ok(results)
}
```

### 3. Custom Delimiters and Formats
```rust
#[pyfunction]
fn parse_csv_custom(
    path: &str,
    delimiter: char,
    has_header: bool,
    skip_rows: usize
) -> PyResult<Vec<Vec<String>>> {
    let mut reader = ReaderBuilder::new()
        .delimiter(delimiter as u8)
        .has_headers(has_header)
        .from_path(path)?;
    
    // Skip rows
    for _ in 0..skip_rows {
        reader.records().next();
    }
    
    // Parse remaining
    // ...
}
```

### 4. Data Validation
```rust
pub struct ValidationRule {
    column: String,
    rule_type: RuleType,
}

pub enum RuleType {
    NotNull,
    Range(f64, f64),
    Regex(String),
}

fn validate_data(data: &[Vec<String>], rules: &[ValidationRule]) -> Vec<ValidationError> {
    // Check each rule
    // Return list of violations
}
```

## Deployment Options

### 1. PyPI Package
```bash
# Build wheel
maturin build --release

# Upload to PyPI
maturin publish
```

### 2. Docker Container
```dockerfile
FROM rust:1.75 as builder
WORKDIR /app
COPY . .
RUN cargo build --release

FROM python:3.11-slim
COPY --from=builder /app/target/release/libdata_parser_rs.so /usr/local/lib/
RUN pip install maturin
CMD ["python"]
```

### 3. GitHub Actions CI/CD
```yaml
name: Build and Test
on: [push]
jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - uses: actions-rs/toolchain@v1
      - run: maturin build --release
      - run: pytest tests/
```

## Comparison with Other Tools

### vs Pandas
- **Speed**: 10-50x faster for parsing
- **Memory**: 40-60% less memory usage
- **Features**: Pandas has more data manipulation features
- **Use Case**: Use Rust parser for ETL, Pandas for analysis

### vs Polars
- **Speed**: Similar performance (both Rust-based)
- **Features**: Polars is more feature-complete
- **Learning**: This project teaches Rust + PyO3 fundamentals
- **Use Case**: Use Polars for production, this for learning

### vs DuckDB
- **Speed**: DuckDB faster for SQL queries
- **Features**: DuckDB has SQL interface
- **Simplicity**: This parser is simpler, more focused
- **Use Case**: DuckDB for analytics, this for custom parsing

## Resources

### Documentation
- [PyO3 Documentation](https://pyo3.rs/) - Python bindings for Rust
- [Rust CSV crate](https://docs.rs/csv/) - CSV parsing in Rust
- [Rust serde_json](https://docs.rs/serde_json/) - JSON serialization
- [maturin Guide](https://www.maturin.rs/) - Build Python packages in Rust
- [Rust Performance Book](https://nnethercote.github.io/perf-book/) - Optimization techniques

### Tutorials
- [PyO3 User Guide](https://pyo3.rs/v0.20.0/getting_started) - Getting started
- [Rust by Example](https://doc.rust-lang.org/rust-by-example/) - Learn Rust
- [Writing Fast Rust](https://www.youtube.com/watch?v=rDoqT-a6UFg) - Performance tips

### Community
- [PyO3 Discord](https://discord.gg/rust-lang) - Get help
- [Rust Users Forum](https://users.rust-lang.org/) - Ask questions
- [r/rust](https://www.reddit.com/r/rust/) - Community discussions

## Questions?

If you get stuck:
1. Review the `tech-spec.md` for detailed architecture
2. Check PyO3 documentation for binding examples
3. Search Rust forums for parsing questions
4. Review the 30 Days Rust bootcamp materials
5. Compare your code with the reference implementation
6. Run benchmarks to verify performance improvements

## Related Projects

After completing this project, consider:
- **Project B**: Parquet Optimizer - Optimize Parquet file layouts
- **Project C**: Iceberg Manager - Work with Apache Iceberg tables
- **Project H**: Format Benchmark - Compare all data formats
- Build a complete ETL pipeline using your Rust parser
