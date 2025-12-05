# Product Requirements Document: Log Processor

## Overview
Build a high-performance log processing tool in Rust that parses, aggregates, and analyzes log files in real-time.

## Goals
- Demonstrate Rust performance for log processing
- Show real-time aggregations
- Create practical CLI tool
- Efficient storage with Parquet

## Core Features

### 1. Log Parsing
- Parse Apache/Nginx log format
- Parse JSON logs
- Custom log format support
- Handle malformed lines

### 2. Real-Time Aggregations
- Request count by endpoint
- Status code distribution
- Response time percentiles (p50, p95, p99)
- Error rate tracking
- Top IPs/User agents

### 3. Storage
- Write to Parquet files
- Time-based partitioning
- Efficient compression
- Queryable output

### 4. Terminal Dashboard
- Real-time metrics display
- Update frequency configurable
- Color-coded output
- Progress indicators

### 5. CLI Interface
- Process single file
- Watch directory for new files
- Tail mode (follow file)
- Export aggregations

## Technical Requirements

### Performance
- Process 100K+ lines/second
- Low memory footprint
- Efficient string parsing
- Parallel processing

### Usability
- Simple CLI interface
- Clear error messages
- Helpful documentation

### Quality
- Unit tests
- Handle edge cases
- Graceful error handling

## Success Metrics
- 10x+ faster than Python alternatives
- Process 1M logs in < 10 seconds
- Real-time dashboard working
- < 500 lines of code

## Timeline
1-2 days implementation
