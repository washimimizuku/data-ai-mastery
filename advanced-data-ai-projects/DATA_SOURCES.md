# Data Sources for Advanced Projects

Comprehensive guide to data sources for all 9 advanced projects, including setup scripts, download instructions, and data generation tools.

## Quick Reference

| Project | Primary Source | Size | Type | Setup Time |
|---------|---------------|------|------|------------|
| 1. Kafka | Synthetic (Faker) | Streaming | Real-time | 10 min |
| 2. Rust Pipeline | NYC Taxi + Faker | 100M+ rows | Batch + Stream | 30 min |
| 3. Lakehouse | Multi-source | 1TB+ | Mixed | 1 hour |
| 4. MLOps | Kaggle ML datasets | 10-100GB | ML training | 30 min |
| 5. Deep Learning | ImageNet, COCO | 100GB+ | Images | 2 hours |
| 6. RAG | Wikipedia, arXiv | 10K-1M docs | Documents | 1 hour |
| 7. Agents | Synthetic + APIs | N/A | Task-based | 10 min |
| 8. LLM Fine-Tune | Alpaca, Dolly | 10K-100K | Instructions | 30 min |
| 9. Full Platform | All above | Multi-TB | Everything | 3 hours |

---

## Project 1: Kafka Streaming Platform

### Data Sources

#### 1. IoT Sensor Data (Recommended)
```python
from faker import Faker
import json
import time
from kafka import KafkaProducer

fake = Faker()

producer = KafkaProducer(
    bootstrap_servers=['localhost:9092'],
    value_serializer=lambda v: json.dumps(v).encode('utf-8')
)

def generate_iot_event():
    return {
        'sensor_id': fake.uuid4(),
        'device_type': fake.random_element(['temperature', 'humidity', 'pressure']),
        'value': fake.random_int(0, 100),
        'timestamp': fake.iso8601(),
        'location': {'lat': fake.latitude(), 'lon': fake.longitude()}
    }

# Stream continuously
while True:
    event = generate_iot_event()
    producer.send('iot-sensors', event)
    time.sleep(0.1)  # 10 events/sec
```

#### 2. Clickstream Data
```python
def generate_clickstream():
    return {
        'user_id': fake.uuid4(),
        'session_id': fake.uuid4(),
        'event_type': fake.random_element(['page_view', 'click', 'purchase']),
        'page_url': fake.url(),
        'timestamp': fake.iso8601(),
        'user_agent': fake.user_agent()
    }
```

#### 3. Real-Time Logs
```python
def generate_log_entry():
    return {
        'timestamp': fake.iso8601(),
        'level': fake.random_element(['INFO', 'WARN', 'ERROR']),
        'service': fake.random_element(['api', 'web', 'worker']),
        'message': fake.sentence(),
        'trace_id': fake.uuid4()
    }
```

### Public Datasets (Alternative)
- **GitHub Events**: https://www.gharchive.org/ (real-time)
- **Twitter Stream**: https://developer.twitter.com/en/docs/twitter-api
- **Stock Market**: https://www.alphavantage.co/ (real-time quotes)

---

## Project 2: Rust Pipeline

### Data Sources

#### 1. NYC Taxi Dataset (Recommended)
```bash
# Download NYC Taxi data (Parquet format)
wget https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2023-01.parquet
wget https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2023-02.parquet
# ... download more months as needed

# Total size: ~500MB per month, 6GB for full year
```

#### 2. E-Commerce Transactions (Faker)
```python
from faker import Faker
import pandas as pd

fake = Faker()

def generate_transactions(n=10_000_000):
    return pd.DataFrame({
        'transaction_id': [fake.uuid4() for _ in range(n)],
        'user_id': [fake.uuid4() for _ in range(n)],
        'product_id': [fake.random_int(1, 10000) for _ in range(n)],
        'amount': [fake.random_int(10, 1000) for _ in range(n)],
        'timestamp': [fake.date_time_this_year() for _ in range(n)],
        'country': [fake.country_code() for _ in range(n)]
    })

# Generate and save
df = generate_transactions(10_000_000)
df.to_parquet('transactions.parquet', compression='snappy')
```

---

## Project 3: Lakehouse Architecture

### Data Sources (Multi-Source)

#### 1. Bronze Layer - Raw Data
```python
# CSV files
wget https://www.kaggle.com/datasets/retailrocket/ecommerce-dataset

# JSON API data
import requests
response = requests.get('https://api.github.com/events')
events = response.json()

# Database dumps
pg_dump -h localhost -U user -d database > dump.sql
```

#### 2. Silver Layer - Cleaned Data
```python
# Transform bronze to silver
df = spark.read.parquet('bronze/raw_data')
df_clean = df.dropna().drop_duplicates()
df_clean.write.format('delta').save('silver/cleaned_data')
```

#### 3. Gold Layer - Aggregated
```python
# Business-level aggregations
df_agg = spark.sql("""
    SELECT 
        date_trunc('day', timestamp) as date,
        country,
        SUM(amount) as total_sales,
        COUNT(*) as transaction_count
    FROM silver.transactions
    GROUP BY 1, 2
""")
df_agg.write.format('delta').save('gold/daily_sales')
```

### Recommended Datasets
- **E-commerce**: https://www.kaggle.com/datasets/carrie1/ecommerce-data
- **Retail**: https://www.kaggle.com/datasets/retailrocket/ecommerce-dataset
- **Logs**: https://www.kaggle.com/datasets/eliasdabbas/web-server-access-logs

---

## Project 4: MLOps Platform

### Data Sources

#### 1. Tabular Data (Classification/Regression)
```python
from sklearn.datasets import make_classification, make_regression

# Classification
X, y = make_classification(n_samples=100000, n_features=20, n_classes=2)

# Or use Kaggle datasets
# Titanic: https://www.kaggle.com/c/titanic
# House Prices: https://www.kaggle.com/c/house-prices-advanced-regression-techniques
# Credit Card Fraud: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
```

#### 2. Image Data (Computer Vision)
```python
from torchvision import datasets

# MNIST
mnist = datasets.MNIST(root='./data', download=True)

# CIFAR-10
cifar = datasets.CIFAR10(root='./data', download=True)

# Or download from Kaggle
# Dogs vs Cats: https://www.kaggle.com/c/dogs-vs-cats
```

#### 3. Time Series
```python
# Stock prices
import yfinance as yf
data = yf.download('AAPL', start='2020-01-01', end='2024-01-01')

# Or use
# Energy consumption: https://www.kaggle.com/datasets/robikscube/hourly-energy-consumption
```

---

## Project 5: Deep Learning Pipeline

### Data Sources

#### 1. ImageNet (Large-Scale)
```bash
# Download ImageNet subset (50GB)
wget https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_train.tar
wget https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_val.tar

# Or use smaller subset
# Tiny ImageNet: https://www.kaggle.com/c/tiny-imagenet
```

#### 2. COCO Dataset (Object Detection)
```bash
# Download COCO 2017 (25GB)
wget http://images.cocodataset.org/zips/train2017.zip
wget http://images.cocodataset.org/zips/val2017.zip
wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip
```

#### 3. Custom Dataset Generation
```python
from torchvision import datasets, transforms

# Use existing datasets
dataset = datasets.ImageFolder(
    root='./data/custom',
    transform=transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor()
    ])
)
```

---

## Project 6: Production RAG System

### Data Sources

#### 1. Wikipedia Dump (Recommended)
```bash
# Download Wikipedia dump (20GB compressed)
wget https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-pages-articles.xml.bz2

# Or use pre-processed
pip install wikipedia
```

```python
import wikipedia

# Download articles
topics = ['Machine Learning', 'Data Engineering', 'Python']
for topic in topics:
    content = wikipedia.page(topic).content
    with open(f'{topic}.txt', 'w') as f:
        f.write(content)
```

#### 2. arXiv Papers
```python
import arxiv

# Download papers
search = arxiv.Search(
    query="machine learning",
    max_results=1000,
    sort_by=arxiv.SortCriterion.SubmittedDate
)

for result in search.results():
    result.download_pdf(filename=f"{result.entry_id}.pdf")
```

#### 3. Technical Documentation
```bash
# Scrape documentation
wget --recursive --no-parent https://docs.python.org/3/

# Or use existing datasets
# Stack Overflow: https://www.kaggle.com/datasets/stackoverflow/stackoverflow
# GitHub README: https://www.kaggle.com/datasets/github/github-repos
```

---

## Project 7: Multi-Agent AI System

### Data Sources

#### 1. Task Descriptions (Synthetic)
```python
from faker import Faker

fake = Faker()

def generate_task():
    return {
        'task_id': fake.uuid4(),
        'description': fake.sentence(),
        'priority': fake.random_element(['low', 'medium', 'high']),
        'category': fake.random_element(['research', 'analysis', 'coding']),
        'deadline': fake.future_date()
    }

tasks = [generate_task() for _ in range(1000)]
```

#### 2. API Data Sources
```python
# Weather API
import requests
weather = requests.get('https://api.openweathermap.org/data/2.5/weather?q=London')

# News API
news = requests.get('https://newsapi.org/v2/top-headlines?country=us')

# Stock API
stocks = requests.get('https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol=IBM')
```

#### 3. Knowledge Base
```python
# Use existing datasets
# MMLU: https://huggingface.co/datasets/cais/mmlu
# HotpotQA: https://huggingface.co/datasets/hotpot_qa
```

---

## Project 8: LLM Fine-Tuning Platform

### Data Sources

#### 1. Alpaca Dataset (Recommended)
```python
from datasets import load_dataset

# Stanford Alpaca (52K instructions)
dataset = load_dataset("tatsu-lab/alpaca")

# Format: instruction, input, output
print(dataset['train'][0])
```

#### 2. Dolly 15k
```python
# Databricks Dolly (15K high-quality)
dataset = load_dataset("databricks/databricks-dolly-15k")
```

#### 3. Domain-Specific Datasets
```python
# SQL generation
dataset = load_dataset("b-mc2/sql-create-context")

# Code generation
dataset = load_dataset("iamtarun/python_code_instructions_18k_alpaca")

# Math reasoning
dataset = load_dataset("gsm8k", "main")
```

#### 4. Custom Dataset Creation
```python
# Create instruction dataset
instructions = [
    {
        "instruction": "Explain machine learning",
        "input": "",
        "output": "Machine learning is..."
    },
    # ... more examples
]

# Save as JSONL
import json
with open('instructions.jsonl', 'w') as f:
    for item in instructions:
        f.write(json.dumps(item) + '\n')
```

---

## Project 9: Full Platform (All Sources)

This project uses data from all previous projects:
- Streaming data (Project 1)
- Batch data (Project 2)
- Multi-source lakehouse (Project 3)
- ML datasets (Project 4, 5)
- Documents for RAG (Project 6)
- Task data for agents (Project 7)
- Instruction data for fine-tuning (Project 8)

### Setup Script
```bash
#!/bin/bash

# Create data directory
mkdir -p data/{streaming,batch,lakehouse,ml,documents,tasks,instructions}

# Download key datasets
cd data/batch
wget https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2023-01.parquet

cd ../ml
kaggle datasets download -d mlg-ulb/creditcardfraud

cd ../documents
python -c "import wikipedia; open('ml.txt', 'w').write(wikipedia.page('Machine Learning').content)"

cd ../instructions
wget https://huggingface.co/datasets/tatsu-lab/alpaca/resolve/main/data/train-00000-of-00001-a09b74b3ef9c3b56.parquet

echo "Data setup complete!"
```

---

## Data Generation Tools

### 1. Faker (Synthetic Data)
```python
from faker import Faker
import pandas as pd

fake = Faker()

# Generate any type of data
df = pd.DataFrame({
    'name': [fake.name() for _ in range(1000)],
    'email': [fake.email() for _ in range(1000)],
    'address': [fake.address() for _ in range(1000)],
    'date': [fake.date_this_year() for _ in range(1000)]
})
```

### 2. SDV (Synthetic Data Vault)
```python
from sdv.tabular import GaussianCopula

# Learn from real data
model = GaussianCopula()
model.fit(real_data)

# Generate synthetic data
synthetic_data = model.sample(num_rows=10000)
```

### 3. Mockaroo (Web-based)
- Visit: https://www.mockaroo.com/
- Configure schema
- Download CSV/JSON

---

## Storage Recommendations

### Local Development
- **SSD**: 500GB+ recommended
- **HDD**: 1TB+ for large datasets
- **External**: Consider external drive for large datasets

### Cloud Storage
- **AWS S3**: $0.023/GB/month
- **GCP Cloud Storage**: $0.020/GB/month
- **Azure Blob**: $0.018/GB/month

### Cost Optimization
- Use compression (Parquet with Snappy)
- Lifecycle policies (move to cold storage)
- Delete intermediate data
- Use free tiers where available

---

## Data Privacy & Compliance

### Best Practices
1. **Use synthetic data** for development
2. **Anonymize** real data (remove PII)
3. **Encrypt** sensitive data at rest
4. **Access control** for production data
5. **Audit logs** for data access

### Tools
- **Faker**: Generate synthetic data
- **Presidio**: PII detection and anonymization
- **SDV**: Synthetic data generation
- **Great Expectations**: Data validation

---

## Quick Start Commands

```bash
# Install data tools
pip install faker pandas pyarrow kaggle datasets wikipedia arxiv

# Set up Kaggle API
mkdir -p ~/.kaggle
cp kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json

# Download sample datasets
kaggle datasets download -d retailrocket/ecommerce-dataset
kaggle datasets download -d mlg-ulb/creditcardfraud

# Generate synthetic data
python -c "
from faker import Faker
import pandas as pd
fake = Faker()
df = pd.DataFrame({
    'id': range(1000000),
    'name': [fake.name() for _ in range(1000000)],
    'email': [fake.email() for _ in range(1000000)]
})
df.to_parquet('synthetic_data.parquet')
print('Generated 1M rows')
"
```

---

## Troubleshooting

### Large Downloads
- Use `wget -c` for resumable downloads
- Use `aria2c` for parallel downloads
- Consider torrents for large datasets

### Storage Issues
- Compress data with Parquet + Snappy
- Use external drives
- Cloud storage for large datasets
- Delete intermediate files

### API Rate Limits
- Use caching
- Implement exponential backoff
- Consider paid tiers
- Use multiple API keys

---

## Additional Resources

- **Kaggle Datasets**: https://www.kaggle.com/datasets
- **Hugging Face Datasets**: https://huggingface.co/datasets
- **AWS Open Data**: https://registry.opendata.aws/
- **Google Dataset Search**: https://datasetsearch.research.google.com/
- **Papers with Code**: https://paperswithcode.com/datasets
