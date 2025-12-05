# Technical Specification: Data Quality Profiler

## Architecture
```
CSV/Parquet → Rust Profiler → Statistics → Report Generator
                    ↓                           ↓
              PyO3 Bindings                HTML/JSON Reports
```

## Technology Stack
- **Rust**: 1.75+
- **Python**: 3.11+
- **Libraries**:
  - polars (Rust, data reading)
  - regex (pattern detection)
  - serde (serialization)
  - PyO3 (Python bindings)
  - tera (HTML templating)

## Core Components

### Profiler
```rust
pub struct DataProfiler {
    data: DataFrame,
}

impl DataProfiler {
    pub fn from_csv(path: &str) -> Result<Self, Error>
    pub fn from_parquet(path: &str) -> Result<Self, Error>
    
    pub fn profile(&self) -> DataProfile
    pub fn profile_column(&self, column: &str) -> ColumnProfile
}

pub struct DataProfile {
    pub row_count: usize,
    pub column_count: usize,
    pub columns: Vec<ColumnProfile>,
    pub correlations: Option<Vec<Vec<f64>>>,
}

pub struct ColumnProfile {
    pub name: String,
    pub data_type: DataType,
    pub statistics: Statistics,
    pub quality: QualityMetrics,
    pub patterns: Vec<Pattern>,
}
```

### Statistics Calculator
```rust
pub struct Statistics {
    pub count: usize,
    pub null_count: usize,
    pub unique_count: usize,
    pub mean: Option<f64>,
    pub median: Option<f64>,
    pub std: Option<f64>,
    pub min: Option<f64>,
    pub max: Option<f64>,
    pub percentiles: Percentiles,
}

impl Statistics {
    pub fn calculate(column: &Series) -> Self
}
```

### Quality Metrics
```rust
pub struct QualityMetrics {
    pub completeness: f64,      // % non-null
    pub uniqueness: f64,         // % unique
    pub validity: f64,           // % matching expected pattern
    pub consistency: f64,        // % matching inferred type
}
```

### Pattern Detector
```rust
pub enum Pattern {
    Email,
    Phone,
    Date,
    Url,
    Uuid,
    NumericId,
    Custom(String),
}

pub struct PatternDetector;

impl PatternDetector {
    pub fn detect(values: &[String]) -> Vec<Pattern>
    pub fn match_rate(values: &[String], pattern: &Pattern) -> f64
}
```

### Report Generator
```rust
pub struct ReportGenerator {
    profile: DataProfile,
}

impl ReportGenerator {
    pub fn generate_html(&self) -> String
    pub fn generate_json(&self) -> String
    pub fn save_report(&self, path: &str) -> Result<(), Error>
}
```

## Python API

```python
import data_profiler_rs

# Profile CSV
profile = data_profiler_rs.profile_csv("data.csv")

# Profile Parquet
profile = data_profiler_rs.profile_parquet("data.parquet")

# Get statistics
print(f"Rows: {profile.row_count}")
print(f"Columns: {profile.column_count}")

# Column-level stats
for col in profile.columns:
    print(f"{col.name}: {col.statistics.mean}")

# Generate report
profile.save_html_report("report.html")
profile.save_json_report("report.json")
```

## HTML Report Structure

```html
<!DOCTYPE html>
<html>
<head>
    <title>Data Profile Report</title>
    <style>/* Styling */</style>
</head>
<body>
    <h1>Data Profile Report</h1>
    
    <section id="overview">
        <h2>Overview</h2>
        <table>
            <tr><td>Rows</td><td>10,000,000</td></tr>
            <tr><td>Columns</td><td>15</td></tr>
            <tr><td>Memory Size</td><td>1.2 GB</td></tr>
        </table>
    </section>
    
    <section id="columns">
        <h2>Column Analysis</h2>
        
        <div class="column">
            <h3>age (numeric)</h3>
            <table>
                <tr><td>Count</td><td>9,950,000</td></tr>
                <tr><td>Null</td><td>50,000 (0.5%)</td></tr>
                <tr><td>Unique</td><td>100</td></tr>
                <tr><td>Mean</td><td>35.2</td></tr>
                <tr><td>Median</td><td>34.0</td></tr>
                <tr><td>Std</td><td>12.5</td></tr>
                <tr><td>Min</td><td>18</td></tr>
                <tr><td>Max</td><td>95</td></tr>
            </table>
            
            <h4>Quality Metrics</h4>
            <ul>
                <li>Completeness: 99.5%</li>
                <li>Uniqueness: 0.001%</li>
            </ul>
            
            <div class="histogram">
                <!-- Distribution chart -->
            </div>
        </div>
        
        <div class="column">
            <h3>email (string)</h3>
            <table>
                <tr><td>Count</td><td>10,000,000</td></tr>
                <tr><td>Null</td><td>0 (0%)</td></tr>
                <tr><td>Unique</td><td>9,998,500</td></tr>
            </table>
            
            <h4>Detected Patterns</h4>
            <ul>
                <li>Email: 99.8% match</li>
            </ul>
            
            <h4>Quality Metrics</h4>
            <ul>
                <li>Completeness: 100%</li>
                <li>Uniqueness: 99.99%</li>
                <li>Validity: 99.8%</li>
            </ul>
        </div>
    </section>
    
    <section id="correlations">
        <h2>Correlations</h2>
        <!-- Correlation matrix heatmap -->
    </section>
    
    <section id="recommendations">
        <h2>Recommendations</h2>
        <ul>
            <li>Column 'age' has 0.5% null values - consider imputation</li>
            <li>Column 'email' has 0.2% invalid formats - review data quality</li>
            <li>Column 'id' is 100% unique - good candidate for primary key</li>
        </ul>
    </section>
</body>
</html>
```

## CLI Interface

```bash
# Profile CSV
data-profiler profile data.csv

# Profile Parquet
data-profiler profile data.parquet

# Generate HTML report
data-profiler profile data.csv --output report.html

# Generate JSON report
data-profiler profile data.csv --output report.json --format json

# Profile specific columns
data-profiler profile data.csv --columns age,email,salary

# Compare two datasets
data-profiler compare data1.csv data2.csv
```

## Benchmarks

### Target Performance
| Dataset Size | Rust Profiler | pandas-profiling | Speedup |
|--------------|---------------|------------------|---------|
| 1M rows | 3s | 65s | 21x |
| 10M rows | 28s | 580s | 20x |
| 50M rows | 145s | 3000s+ | 20x+ |

## Project Structure
```
project-g-data-profiler/
├── rust/
│   ├── src/
│   │   ├── lib.rs
│   │   ├── profiler.rs
│   │   ├── statistics.rs
│   │   ├── patterns.rs
│   │   ├── quality.rs
│   │   └── report.rs
│   ├── templates/
│   │   └── report.html.tera
│   └── Cargo.toml
├── python/
│   ├── data_profiler/
│   │   └── __init__.py
│   ├── examples/
│   │   └── usage.py
│   └── tests/
├── benches/
│   └── profiler_bench.rs
└── README.md
```

## Testing Strategy
- Unit tests for statistics calculations
- Test pattern detection accuracy
- Validate quality metrics
- Test with various data types
- Benchmark against pandas-profiling
- Test edge cases (all nulls, all unique, etc.)

## Performance Optimizations
- Use Polars for fast data reading
- Parallel column processing with rayon
- Efficient string pattern matching
- Minimize memory allocations
- Batch processing for large datasets
