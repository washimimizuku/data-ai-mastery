# Product Requirements Document: Parquet Optimizer

## Overview
Build a tool to analyze and optimize Parquet files by testing different compression algorithms, row group sizes, and encoding strategies.

## Goals
- Demonstrate Parquet format expertise
- Show optimization techniques
- Provide actionable recommendations
- Create practical CLI tool

## Core Features

### 1. Parquet Analysis
- Read Parquet file metadata
- Analyze schema and statistics
- Report file size and structure
- Show compression ratios

### 2. Optimization Testing
- Test compression algorithms (Snappy, ZSTD, LZ4, Gzip)
- Vary row group sizes
- Test different encodings
- Measure read/write performance

### 3. Recommendations
- Suggest optimal compression
- Recommend row group size
- Identify optimization opportunities
- Cost/benefit analysis

### 4. CLI Tool
- Simple command interface
- Progress indicators
- Comparison reports
- Export results

## Technical Requirements

### Performance
- Fast file analysis
- Efficient compression testing
- Parallel processing where applicable

### Usability
- Clear CLI interface
- Helpful error messages
- Detailed reports

### Quality
- Unit tests
- Handle edge cases
- Validate Parquet files

## Success Metrics
- Demonstrate 2-5x file size reduction
- Show read performance trade-offs
- Clear optimization recommendations
- < 400 lines of code

## Timeline
1-2 days implementation
