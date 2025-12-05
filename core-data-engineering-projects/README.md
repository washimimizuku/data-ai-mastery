# Data Engineering Projects

Focused, standalone data engineering projects showcasing Rust performance, modern table formats (Iceberg, Delta Lake), and distributed processing with Spark. All projects run locally without cloud dependencies.

## Projects Overview

### Project A: Rust Data Parser Benchmark
**Time**: 1 day | **Focus**: Raw performance comparison

Parse large CSV/JSON files with Rust vs Python/Pandas. Demonstrates 10-50x performance improvements with PyO3 bindings.

**Stack**: Rust, Python, Pandas, PyO3

---

### Project B: Parquet Optimizer
**Time**: 1-2 days | **Focus**: File format optimization

Analyze and optimize Parquet files with different compression algorithms and configurations.

**Stack**: Rust, Python, Parquet, Arrow

---

### Project C: Iceberg Table Manager
**Time**: 2 days | **Focus**: Iceberg fundamentals

Local Iceberg table operations including time travel, schema evolution, and snapshot management.

**Stack**: Rust, Python, Iceberg, DuckDB

---

### Project D: Delta Lake Operations
**Time**: 1-2 days | **Focus**: Delta Lake basics

Delta Lake CRUD operations, time travel, optimize, Z-order, and vacuum demonstrations.

**Stack**: Python, PySpark, Delta Lake

---

### Project E: Spark Streaming Demo
**Time**: 2 days | **Focus**: Structured streaming basics

Real-time data processing with Spark Structured Streaming writing to Delta Lake.

**Stack**: Python, PySpark, Delta Lake

---

### Project F: Log Processor
**Time**: 1-2 days | **Focus**: Real-time processing with Rust

High-performance log parsing and aggregation with terminal dashboard.

**Stack**: Rust, Arrow, Parquet

---

### Project G: Data Quality Profiler
**Time**: 1-2 days | **Focus**: Fast profiling with Rust

Statistical data profiling tool with Rust core and Python wrapper.

**Stack**: Rust, Python, Polars

---

### Project H: Data Format Performance Benchmark
**Time**: 1-2 days | **Focus**: Format comparison and performance

Comprehensive benchmark comparing ETL performance of all major data formats (CSV, JSON, Parquet, Avro, Arrow, Iceberg, Delta Lake) using both Python and Rust.

**Stack**: Python, Rust, Parquet, Avro, Arrow, Iceberg, Delta Lake

---

## Recommended Learning Paths

### Path 1: Balanced (1 week)
1. Project A - Rust benchmark (1 day)
2. Project C - Iceberg basics (2 days)
3. Project D - Delta basics (1 day)
4. Project E - Spark streaming (2 days)

**Covers**: Rust, Iceberg, Delta Lake, PySpark

---

### Path 2: Rust-Focused (1 week)
1. Project A - Parser benchmark (1 day)
2. Project B - Parquet optimizer (1 day)
3. Project F - Log processor (2 days)
4. Project G - Data profiler (1 day)

**Covers**: Rust performance, PyO3, Arrow, Parquet, practical tools

---

### Path 3: Table Formats Focus (1 week)
1. Project C - Iceberg (2 days)
2. Project D - Delta (1 day)
3. Project E - Spark streaming (2 days)

**Covers**: Modern lakehouse patterns, PySpark, table formats

---

## Technology Coverage

| Project | Rust | PySpark | Iceberg | Delta | Arrow | Parquet | Avro |
|---------|------|---------|---------|-------|-------|---------|------|
| A. Parser | ✅✅ | - | - | - | - | - | - |
| B. Parquet | ✅✅ | - | - | - | ✅ | ✅✅ | - |
| C. Iceberg | ✅ | - | ✅✅ | - | ✅ | ✅ | - |
| D. Delta | - | ✅✅ | - | ✅✅ | - | - | - |
| E. Streaming | - | ✅✅ | - | ✅ | - | - | - |
| F. Logs | ✅✅ | - | - | - | ✅✅ | ✅ | - |
| G. Profiler | ✅✅ | - | - | - | - | - | - |
| H. Benchmark | ✅✅ | - | ✅ | ✅ | ✅✅ | ✅✅ | ✅✅ |

---

## Key Features

**All projects include**:
- Working code (< 500 lines)
- README with setup and examples
- Benchmark results with charts
- Basic tests
- Demo screenshots/GIFs

**All projects are**:
- Runnable locally (no cloud required)
- Completable in 1-2 days
- Focused on specific skills
- Production-quality code
- Well-documented

---

## Data Sources

**See [DATA_SOURCES.md](./DATA_SOURCES.md) for comprehensive data source guide.**

### Quick Reference

#### Synthetic Data (Recommended)
Use **Faker** to generate custom datasets - see DATA_SOURCES.md for complete scripts.

#### Public Datasets
- **NYC Taxi**: Large scale data
- **E-commerce**: Kaggle datasets
- **Logs**: Web server logs
- **GitHub Events**: Real-time streaming

### By Project

| Project | Recommended Source | Size | Notes |
|---------|-------------------|------|-------|
| A. Parser | Faker-generated CSV/JSON | 10M rows | E-commerce transactions |
| B. Parquet | NYC Taxi or Faker | 1GB+ | Need large files |
| C. Iceberg | Faker events | 5M+ rows | Multiple versions |
| D. Delta | Kaggle e-commerce | 1M+ rows | Need updates/deletes |
| E. Streaming | Faker real-time | Continuous | IoT or clickstream |
| F. Logs | Faker or Kaggle logs | 10M lines | Apache/Nginx format |
| G. Profiler | Multiple Kaggle datasets | Various | Test data quality |
| H. Benchmark | Faker-generated | 10M rows | Mixed data types |

---

## Prerequisites

### For Rust Projects (A, B, F, G)
- Rust 1.75+
- Python 3.11+
- Cargo

### For Spark Projects (D, E)
- Python 3.11+
- PySpark 3.5+
- Java 11+

### For Iceberg Project (C)
- Rust 1.75+
- Python 3.11+
- DuckDB

---

## Why These Projects?

1. **No cloud dependency** - Run everything locally
2. **Quick to complete** - 1-2 days each
3. **Focused scope** - One skill per project
4. **Modern stack** - Rust, Iceberg, Delta, Spark
5. **Practical tools** - Solve real problems
6. **Performance focus** - Clear benchmarks
7. **Portfolio ready** - Professional quality

---

## Getting Started

Each project folder contains:
- `prd.md` - Product requirements
- `tech-spec.md` - Technical specification
- `implementation-plan.md` - Step-by-step guide

Start with any project based on your learning goals or follow one of the recommended paths above.
