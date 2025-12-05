# Technical Specification: Log Processor

## Architecture
```
Log Files → Parser → Aggregator → Terminal Dashboard
                         ↓
                   Parquet Writer
```

## Technology Stack
- **Rust**: 1.75+
- **Libraries**:
  - regex (parsing)
  - serde_json (JSON logs)
  - arrow-rs (Arrow format)
  - parquet (Parquet writing)
  - clap (CLI)
  - crossterm/ratatui (terminal UI)

## Core Components

### Log Parser
```rust
pub enum LogFormat {
    Apache,
    Nginx,
    Json,
}

pub struct LogParser {
    format: LogFormat,
}

impl LogParser {
    pub fn parse_line(&self, line: &str) -> Result<LogEntry, ParseError>
    pub fn parse_file(&self, path: &str) -> Vec<LogEntry>
}

pub struct LogEntry {
    timestamp: DateTime<Utc>,
    ip: String,
    method: String,
    path: String,
    status: u16,
    response_time: f64,
    user_agent: String,
}
```

### Aggregator
```rust
pub struct LogAggregator {
    entries: Vec<LogEntry>,
}

impl LogAggregator {
    pub fn count_by_endpoint(&self) -> HashMap<String, usize>
    pub fn status_distribution(&self) -> HashMap<u16, usize>
    pub fn percentiles(&self, field: &str) -> Percentiles
    pub fn error_rate(&self) -> f64
    pub fn top_ips(&self, n: usize) -> Vec<(String, usize)>
}

pub struct Percentiles {
    p50: f64,
    p95: f64,
    p99: f64,
}
```

### Parquet Writer
```rust
pub struct ParquetWriter {
    output_dir: String,
}

impl ParquetWriter {
    pub fn write_batch(&self, entries: &[LogEntry]) -> Result<(), Error>
    pub fn partition_by_time(&self, entries: &[LogEntry]) -> HashMap<String, Vec<LogEntry>>
}
```

### Terminal Dashboard
```rust
pub struct Dashboard {
    aggregator: LogAggregator,
}

impl Dashboard {
    pub fn render(&self)
    pub fn update(&mut self, new_entries: Vec<LogEntry>)
}
```

## Log Format Examples

### Apache Common Log Format
```
127.0.0.1 - - [01/Jan/2024:12:00:00 +0000] "GET /api/users HTTP/1.1" 200 1234
```

### JSON Log Format
```json
{
  "timestamp": "2024-01-01T12:00:00Z",
  "ip": "127.0.0.1",
  "method": "GET",
  "path": "/api/users",
  "status": 200,
  "response_time": 0.123,
  "user_agent": "Mozilla/5.0..."
}
```

## CLI Interface

```bash
# Process single file
log-processor process access.log

# Watch directory
log-processor watch ./logs --format apache

# Tail mode (follow file)
log-processor tail access.log --dashboard

# Export aggregations
log-processor aggregate access.log --output stats.json

# Write to Parquet
log-processor convert access.log --output data.parquet
```

## Terminal Dashboard Output

```
Log Processor Dashboard
=======================
Total Requests: 1,234,567
Time Range: 2024-01-01 00:00:00 - 2024-01-01 23:59:59

Status Codes
------------
200: ████████████████████ 85.2% (1,051,234)
404: ███░░░░░░░░░░░░░░░░░  8.1% (100,000)
500: ██░░░░░░░░░░░░░░░░░░  6.7% (82,333)

Response Time Percentiles
-------------------------
p50: 0.123s
p95: 0.456s
p99: 1.234s

Top Endpoints
-------------
1. /api/users        234,567 requests
2. /api/products     123,456 requests
3. /api/orders        98,765 requests

Top IPs
-------
1. 192.168.1.100     12,345 requests
2. 192.168.1.101      9,876 requests
3. 192.168.1.102      8,765 requests

Error Rate: 6.7%
Processing Rate: 125,432 logs/sec
```

## Performance Optimizations

### Parallel Processing
```rust
use rayon::prelude::*;

let entries: Vec<LogEntry> = lines
    .par_iter()
    .filter_map(|line| parser.parse_line(line).ok())
    .collect();
```

### Efficient String Handling
```rust
// Use string slices instead of owned strings where possible
pub fn parse_status(line: &str) -> Option<u16> {
    // Extract status code without allocating
}
```

### Batch Writing
```rust
// Write in batches to Parquet
const BATCH_SIZE: usize = 10_000;

for chunk in entries.chunks(BATCH_SIZE) {
    writer.write_batch(chunk)?;
}
```

## Project Structure
```
project-f-log-processor/
├── src/
│   ├── main.rs
│   ├── parser.rs
│   ├── aggregator.rs
│   ├── writer.rs
│   ├── dashboard.rs
│   └── cli.rs
├── tests/
│   ├── test_parser.rs
│   └── test_aggregator.rs
├── examples/
│   └── sample_logs.txt
├── benches/
│   └── parser_bench.rs
└── README.md
```

## Benchmarks

### Target Performance
| Operation | Target | Comparison |
|-----------|--------|------------|
| Parse 1M Apache logs | < 10s | Python: ~120s (12x) |
| Parse 1M JSON logs | < 8s | Python: ~90s (11x) |
| Aggregate 1M entries | < 2s | Python: ~25s (12x) |
| Write to Parquet | < 5s | Python: ~45s (9x) |

## Testing Strategy
- Unit tests for parser (various formats)
- Test malformed log lines
- Test aggregation accuracy
- Benchmark against Python
- Test Parquet output validity

## Error Handling
- Skip malformed lines (with warning)
- Handle missing files gracefully
- Validate Parquet writes
- Report parsing errors
