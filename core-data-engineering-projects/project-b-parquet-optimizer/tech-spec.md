# Technical Specification: Parquet Optimizer

## Architecture
```
Input Parquet → Analyzer → Optimizer → Test Configurations → Report
                                ↓
                         Optimized Files + Recommendations
```

## Technology Stack
- **Rust**: 1.75+
- **Libraries**:
  - parquet (Rust)
  - arrow-rs (Rust)
  - clap (CLI)
  - serde (serialization)

## Core Components

### Parquet Analyzer
```rust
pub struct ParquetAnalyzer {
    file_path: String,
}

impl ParquetAnalyzer {
    pub fn analyze(&self) -> FileMetadata
    pub fn get_schema(&self) -> Schema
    pub fn get_statistics(&self) -> Statistics
    pub fn get_compression_info(&self) -> CompressionInfo
}
```

### Optimizer
```rust
pub struct ParquetOptimizer {
    input_path: String,
    output_dir: String,
}

impl ParquetOptimizer {
    pub fn test_compressions(&self) -> Vec<CompressionResult>
    pub fn test_row_group_sizes(&self, sizes: Vec<usize>) -> Vec<Result>
    pub fn recommend(&self, results: &[Result]) -> Recommendation
}
```

## Optimization Tests

### Compression Algorithms
- Snappy (default, balanced)
- ZSTD (best compression)
- LZ4 (fastest)
- Gzip (good compression)
- Uncompressed (baseline)

### Row Group Sizes
- 128 MB (default)
- 256 MB
- 512 MB
- 1 GB

### Metrics Collected
- File size (bytes)
- Compression ratio
- Write time (seconds)
- Read time (seconds)
- Memory usage

## CLI Interface

```bash
# Analyze file
parquet-optimizer analyze input.parquet

# Test all compressions
parquet-optimizer optimize input.parquet --test-all

# Test specific compression
parquet-optimizer optimize input.parquet --compression zstd

# Full report
parquet-optimizer report input.parquet --output report.json
```

## Output Report

```
Parquet File Analysis
=====================
File: data.parquet
Size: 1.2 GB
Rows: 10,000,000
Columns: 15
Current Compression: Snappy

Optimization Results
====================
Compression    Size      Ratio   Write    Read
-------------------------------------------------
Uncompressed   3.5 GB    1.0x    12.3s    8.1s
Snappy         1.2 GB    2.9x    15.4s    9.2s
LZ4            1.4 GB    2.5x    13.8s    8.7s
ZSTD           0.8 GB    4.4x    22.1s    10.5s
Gzip           0.9 GB    3.9x    28.3s    11.2s

Recommendation
==============
Best for storage: ZSTD (4.4x compression, 0.8 GB)
Best for speed: LZ4 (2.5x compression, fastest read)
Balanced: Snappy (current, good balance)
```

## Project Structure
```
project-b-parquet-optimizer/
├── src/
│   ├── main.rs
│   ├── analyzer.rs
│   ├── optimizer.rs
│   └── cli.rs
├── tests/
├── examples/
│   └── sample_data.parquet
├── results/
└── README.md
```

## Testing Strategy
- Unit tests for analyzer
- Integration tests for optimizer
- Test with various file sizes
- Validate output files

## Performance Targets
- Analyze 1GB file in < 5 seconds
- Test all compressions in < 2 minutes
- Generate report in < 1 second
