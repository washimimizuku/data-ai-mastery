# Project F: Log Processor

## Overview

Build a high-performance log processing tool in Rust that parses various log formats (Apache, Nginx, JSON), aggregates metrics in real-time, and displays statistics in a terminal dashboard.

**What You'll Build**: A Rust CLI tool that processes 100K+ logs/second, calculates aggregations, writes to Parquet, and displays live metrics in a terminal UI.

**What You'll Learn**: High-performance log parsing with regex, real-time aggregations, terminal UI development with ratatui, Parquet writing with Arrow, and Rust optimization techniques.

## Time Estimate

**1-2 days (8-16 hours)**

### Day 1 (8 hours)
- Hours 1-2: Parser implementation (Apache, JSON formats)
- Hours 3-4: Aggregation engine (counts, percentiles, distributions)
- Hours 5-6: Parquet writer with Arrow schema
- Hours 7-8: CLI interface with clap

### Day 2 (Optional, 4-6 hours)
- Hours 1-3: Terminal dashboard with ratatui
- Hours 4-5: Benchmarking vs Python
- Hour 6: Documentation and examples

## Prerequisites

### Required Knowledge
- [30 Days of Rust](https://github.com/washimimizuku/30-days-rust-data-ai) - Days 1-25
  - Days 1-10: Rust basics (ownership, types, structs)
  - Days 11-15: Error handling and regex
  - Days 16-20: File I/O and parsing
  - Days 21-25: Performance optimization

### Technical Requirements
- Rust 1.75+ installed
- Basic understanding of log formats
- Regex fundamentals
- Terminal/CLI experience

### Tools
- Rust toolchain (rustc, cargo)
- Text editor or IDE
- Git for version control

## Getting Started

### Step 1: Review Documentation
1. `prd.md` - Product requirements and goals
2. `tech-spec.md` - Technical architecture and design
3. `implementation-plan.md` - Step-by-step build guide

### Step 2: Initialize Project
```bash
cargo new log-processor
cd log-processor
```

Add to `Cargo.toml`:
```toml
[dependencies]
regex = "1.10"
serde = { version = "1.0", features = ["derive"] }
serde_json = "1.0"
chrono = "0.4"
clap = { version = "4.4", features = ["derive"] }
arrow = "49.0"
parquet = "49.0"
crossterm = "0.27"
ratatui = "0.25"
rayon = "1.8"
```

### Step 3: Follow Implementation Plan
Build components in order:
1. **Parser** (Hours 1-2): Apache and JSON log parsing
2. **Aggregator** (Hours 3-4): Statistics and metrics
3. **Parquet Writer** (Hours 5-6): Arrow schema and batch writing
4. **CLI** (Hours 7-8): Commands with clap
5. **Dashboard** (Optional): Terminal UI with ratatui

## Core Components

### 1. Log Parser
```rust
// src/parser.rs
use regex::Regex;
use chrono::{DateTime, Utc};

#[derive(Debug, Clone)]
pub struct LogEntry {
    pub timestamp: DateTime<Utc>,
    pub ip: String,
    pub method: String,
    pub path: String,
    pub status: u16,
    pub response_time: f64,
}

pub struct LogParser {
    apache_regex: Regex,
}

impl LogParser {
    pub fn parse_apache(&self, line: &str) -> Option<LogEntry> {
        // Parse Apache Common Log Format
        // Example: 127.0.0.1 - - [01/Jan/2024:12:00:00 +0000] "GET /api HTTP/1.1" 200 1234
    }
    
    pub fn parse_json(&self, line: &str) -> Option<LogEntry> {
        // Parse JSON log format
    }
}
```

### 2. Aggregator
```rust
// src/aggregator.rs
use std::collections::HashMap;

pub struct LogAggregator {
    entries: Vec<LogEntry>,
}

impl LogAggregator {
    pub fn status_distribution(&self) -> HashMap<u16, usize> {
        // Count by status code
    }
    
    pub fn top_endpoints(&self, n: usize) -> Vec<(String, usize)> {
        // Top N endpoints by request count
    }
    
    pub fn response_time_percentiles(&self) -> Percentiles {
        // Calculate p50, p95, p99
    }
    
    pub fn error_rate(&self) -> f64 {
        // Percentage of 4xx/5xx responses
    }
}
```

### 3. Parquet Writer
```rust
// src/writer.rs
use arrow::array::*;
use parquet::arrow::ArrowWriter;

pub fn write_to_parquet(entries: &[LogEntry], path: &str) -> Result<()> {
    // Convert LogEntry to Arrow arrays
    // Create RecordBatch
    // Write with Snappy compression
}
```

### 4. Terminal Dashboard
```rust
// src/dashboard.rs
use ratatui::{Terminal, widgets::*};

pub fn run_dashboard(aggregator: LogAggregator) -> Result<()> {
    // Display real-time metrics
    // Bar charts for status codes
    // Top endpoints and IPs
    // Press 'q' to quit
}
```

## Key Features

### 1. Log Parsing
- Apache Common Log Format
- JSON structured logs
- Nginx Combined Log Format (optional)
- Handle malformed lines gracefully

### 2. Real-Time Aggregations
- Request count by endpoint
- Status code distribution
- Response time percentiles (p50, p95, p99)
- Error rate tracking (4xx/5xx)
- Top IPs and user agents

### 3. Parquet Storage
- Write to columnar format
- Snappy compression
- Arrow schema definition
- Queryable with DuckDB/Polars

### 4. Terminal Dashboard (Optional)
- Real-time metrics display
- Bar charts for distributions
- Color-coded status codes
- Keyboard navigation (q to quit)

### 5. CLI Interface
```bash
# Process single file
log-processor process access.log --format apache

# Show dashboard
log-processor dashboard access.log

# Export to Parquet
log-processor export access.log --output stats.parquet

# Aggregate and report
log-processor aggregate access.log --output report.json
```

## Success Criteria

- [ ] Parse Apache and JSON log formats
- [ ] Process 100K+ logs per second
- [ ] Calculate all aggregations (status, endpoints, percentiles, error rate)
- [ ] Write to Parquet with Snappy compression
- [ ] CLI with process, aggregate, export commands
- [ ] Terminal dashboard (optional)
- [ ] 10x+ faster than Python equivalent
- [ ] Process 1M logs in < 10 seconds
- [ ] Code < 500 lines
- [ ] Unit tests for parser and aggregator

## Learning Outcomes

- Parse complex text formats with regex in Rust
- Build high-performance text processing tools
- Create terminal UIs with ratatui
- Write data to Parquet with Arrow
- Optimize Rust code for throughput
- Handle large files efficiently
- Build production-ready CLI tools

## Performance Targets

**Processing Speed** (from PRD):
- 100K+ logs/second throughput
- 1M logs processed in < 10 seconds
- 10x+ faster than Python alternatives

**Benchmarks**:
```
Rust Log Processor:
  1M Apache logs: ~8 seconds
  Throughput: 125,000 logs/second

Python Equivalent:
  1M logs: ~95 seconds
  Throughput: 10,500 logs/second

Speedup: 12x faster
Memory: 82% less than Python
```

## Project Structure

```
project-f-log-processor/
├── src/
│   ├── main.rs           # CLI entry point
│   ├── parser.rs         # Log parsing (Apache, JSON)
│   ├── aggregator.rs     # Statistics and metrics
│   ├── writer.rs         # Parquet export
│   ├── dashboard.rs      # Terminal UI (optional)
│   └── cli.rs            # CLI commands
├── tests/
│   ├── test_parser.rs
│   └── test_aggregator.rs
├── examples/
│   └── sample_logs.txt
├── benches/
│   └── parser_bench.rs
├── Cargo.toml
├── prd.md
├── tech-spec.md
├── implementation-plan.md
└── README.md
```

## Common Challenges

### Regex Performance
Compile regex once at initialization, use string slices instead of owned strings

### Memory Usage
Stream processing instead of loading all logs into memory, process in batches

### Terminal UI Flickering
Use double buffering, limit refresh rate to 10-20 FPS

## Next Steps

1. Add to portfolio with performance benchmarks
2. Write blog post: "Building a 12x Faster Log Processor in Rust"
3. Extend with anomaly detection or alerting
4. Continue to Project G: Data Quality Profiler

## Resources

- [Rust Regex Docs](https://docs.rs/regex/)
- [Ratatui Tutorial](https://ratatui.rs/)
- [Arrow Rust](https://docs.rs/arrow/)
- [Parquet Rust](https://docs.rs/parquet/)
