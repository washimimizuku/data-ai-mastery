# Data Sources for Data Engineering Projects

## Quick Start: Synthetic Data Generation

### Faker (Recommended for All Projects)

```python
from faker import Faker
import pandas as pd
import random
from datetime import datetime, timedelta

fake = Faker()

def generate_ecommerce_data(num_rows=10_000_000):
    """Generate realistic e-commerce transaction data"""
    data = {
        'transaction_id': range(num_rows),
        'user_id': [random.randint(1, 100000) for _ in range(num_rows)],
        'timestamp': [fake.date_time_between(start_date='-1y', end_date='now') for _ in range(num_rows)],
        'product_id': [random.randint(1, 10000) for _ in range(num_rows)],
        'product_name': [fake.word() for _ in range(num_rows)],
        'category': [fake.random_element(['Electronics', 'Clothing', 'Food', 'Books', 'Home']) for _ in range(num_rows)],
        'quantity': [random.randint(1, 10) for _ in range(num_rows)],
        'price': [round(random.uniform(10, 1000), 2) for _ in range(num_rows)],
        'region': [fake.random_element(['US', 'EU', 'ASIA', 'LATAM']) for _ in range(num_rows)],
        'payment_method': [fake.random_element(['Credit Card', 'PayPal', 'Crypto', 'Bank Transfer']) for _ in range(num_rows)],
        'discount': [round(random.uniform(0, 0.3), 2) for _ in range(num_rows)]
    }
    
    return pd.DataFrame(data)

# Generate and save in multiple formats
df = generate_ecommerce_data(10_000_000)
df.to_csv('data/ecommerce.csv', index=False)
df.to_json('data/ecommerce.json', orient='records', lines=True)
df.to_parquet('data/ecommerce.parquet', index=False)
```

### Log Data Generation

```python
def generate_log_data(num_lines=10_000_000):
    """Generate Apache-style log entries"""
    logs = []
    for _ in range(num_lines):
        ip = fake.ipv4()
        timestamp = fake.date_time_this_year().strftime('%d/%b/%Y:%H:%M:%S +0000')
        method = fake.random_element(['GET', 'POST', 'PUT', 'DELETE'])
        path = fake.uri_path()
        status = fake.random_element([200, 200, 200, 404, 500, 301])
        size = random.randint(100, 50000)
        
        log = f'{ip} - - [{timestamp}] "{method} {path} HTTP/1.1" {status} {size}'
        logs.append(log)
    
    with open('data/access.log', 'w') as f:
        f.write('\n'.join(logs))
```

---

## Public Datasets

### Large-Scale Datasets

#### 1. NYC Taxi Dataset
- **Source**: [Kaggle](https://www.kaggle.com/datasets/elemento/nyc-yellow-taxi-trip-data)
- **Size**: 1-10GB per month
- **Format**: CSV, Parquet
- **Use for**: Projects B, C, D, H
- **Schema**: pickup/dropoff times, locations, fares, distances

#### 2. Brazilian E-Commerce
- **Source**: [Kaggle](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce)
- **Size**: ~100MB
- **Format**: CSV
- **Use for**: Projects A, D, G
- **Schema**: Orders, customers, products, reviews

#### 3. GitHub Archive
- **Source**: [GH Archive](https://www.gharchive.org/)
- **Size**: Continuous stream
- **Format**: JSON
- **Use for**: Project E (Streaming)
- **Schema**: GitHub events (commits, PRs, issues)

#### 4. Web Server Logs
- **Source**: [Kaggle](https://www.kaggle.com/datasets/eliasdabbas/web-server-access-logs)
- **Size**: 100MB-1GB
- **Format**: Log files
- **Use for**: Project F
- **Schema**: Apache/Nginx log format

---

## Project-Specific Recommendations

### Project A: Rust Data Parser Benchmark
**Recommended**: Faker-generated CSV/JSON (10M rows)
```bash
python generate_data.py --format csv --rows 10000000
python generate_data.py --format json --rows 10000000
```

### Project B: Parquet Optimizer
**Recommended**: NYC Taxi or Faker (1GB+)
- Need large files to show optimization benefits
- Multiple columns with different data types

### Project C: Iceberg Table Manager
**Recommended**: Faker events (5M+ rows)
- Generate multiple batches for time travel demo
- Include updates and deletes

### Project D: Delta Lake Operations
**Recommended**: Kaggle e-commerce or Faker
- Need realistic update/delete scenarios
- MERGE operations require matching keys

### Project E: Spark Streaming Demo
**Recommended**: Faker real-time generation
```python
# Continuous generation
while True:
    batch = generate_ecommerce_data(1000)
    batch.to_json(f'input/batch_{timestamp}.json', orient='records', lines=True)
    time.sleep(5)
```

### Project F: Log Processor
**Recommended**: Faker log generation or Kaggle logs
- Apache/Nginx format
- 10M+ lines for performance testing

### Project G: Data Quality Profiler
**Recommended**: Multiple Kaggle datasets
- Test with various data quality issues
- Include nulls, duplicates, outliers

### Project H: Data Format Performance Benchmark
**Recommended**: Faker-generated (10M rows)
- Mixed data types (int, float, string, date)
- Consistent schema across all formats

---

## Installation

```bash
# Install Faker
pip install faker pandas pyarrow

# Generate sample data
python -c "
from faker import Faker
import pandas as pd
import random

fake = Faker()
data = {
    'id': range(1000000),
    'name': [fake.name() for _ in range(1000000)],
    'email': [fake.email() for _ in range(1000000)],
    'created_at': [fake.date_time_this_year() for _ in range(1000000)]
}
df = pd.DataFrame(data)
df.to_csv('sample_data.csv', index=False)
print('Generated 1M rows')
"
```

---

## Data Storage Structure

```
data-engineering/
├── data/
│   ├── raw/
│   │   ├── ecommerce.csv
│   │   ├── ecommerce.json
│   │   └── access.log
│   ├── processed/
│   │   ├── ecommerce.parquet
│   │   ├── ecommerce.avro
│   │   └── ecommerce.arrow
│   ├── iceberg/
│   │   └── warehouse/
│   ├── delta/
│   │   └── tables/
│   └── scripts/
│       ├── generate_ecommerce.py
│       ├── generate_logs.py
│       └── download_kaggle.py
```

---

## Tips

1. **Generate once, reuse**: Create datasets and use across multiple projects
2. **Start small**: Test with 100K rows, then scale to 10M
3. **Use Parquet**: Faster to read/write than CSV for large datasets
4. **Version control**: Don't commit large datasets, use `.gitignore`
5. **Document schema**: Always document your data schema

---

## Kaggle Setup

```bash
# Install Kaggle CLI
pip install kaggle

# Setup credentials (~/.kaggle/kaggle.json)
# Download dataset
kaggle datasets download -d elemento/nyc-yellow-taxi-trip-data
unzip nyc-yellow-taxi-trip-data.zip -d data/raw/
```
