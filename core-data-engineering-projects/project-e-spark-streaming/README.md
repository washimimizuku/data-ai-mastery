# Project E: Spark Structured Streaming Demo

## Objective

Build real-time streaming applications using Spark Structured Streaming, demonstrating windowing operations, watermarking for late data, stateful processing, and integration with Delta Lake.

**What You'll Build**: Multiple streaming applications showcasing tumbling/sliding windows, late data handling with watermarks, stream-static joins, and real-time aggregations writing to Delta Lake with exactly-once semantics.

**What You'll Learn**: Structured Streaming architecture, windowing operations (tumbling, sliding, session), watermarking strategies, stateful processing, checkpoint management, and production streaming patterns.

## Time Estimate

**2 days (16 hours)**

- Hours 1-2: Streaming setup and data generator
- Hours 3-4: Simple pass-through streaming
- Hours 5-7: Windowed aggregations (tumbling, sliding)
- Hours 8-10: Watermarking and late data handling
- Hours 11-13: Stateful processing and stream joins
- Hours 14-16: Monitoring, optimization, and documentation

## Prerequisites

### Required Knowledge
Complete these bootcamp sections first:
- [100 Days Data & AI](https://github.com/washimimizuku/100-days-data-ai) - Days 21-30
  - Days 21-25: Spark basics
  - Days 26-30: Structured Streaming
- [60 Days Advanced](https://github.com/washimimizuku/60-days-advanced-data-ai) - Days 1-10
  - Days 6-10: Advanced streaming patterns

### Technical Requirements
- Python 3.11+ installed
- PySpark 3.5+ installed
- Understanding of Spark DataFrames
- Basic knowledge of streaming concepts
- Completed Project D (Delta Lake) recommended

### Tools Needed
- Python with pyspark, delta-spark
- Jupyter Lab for notebooks
- Git for version control
- 2-4 GB free disk space

## Getting Started

### Step 1: Review Documentation
Read the project documents in order:
1. `prd.md` - Understand what you're building
2. `tech-spec.md` - Review technical architecture
3. `implementation-plan.md` - Follow the implementation steps

### Step 2: Set Up Streaming Environment
```bash
# Install dependencies
pip install pyspark==3.5.0 delta-spark==2.4.0 jupyterlab

# Create project structure
mkdir -p spark-streaming/{input,data,checkpoints,src}
cd spark-streaming
```

### Step 3: Configure Spark for Streaming
```python
from pyspark.sql import SparkSession
from delta import *

builder = SparkSession.builder \
    .appName("StructuredStreamingDemo") \
    .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension") \
    .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog") \
    .config("spark.jars.packages", "io.delta:delta-core_2.12:2.4.0") \
    .config("spark.sql.streaming.checkpointLocation", "./checkpoints")

spark = configure_spark_with_delta_pip(builder).getOrCreate()

print("✓ Spark Structured Streaming configured")
```

### Step 4: Create Data Generator
```python
# src/data_generator.py
import time
import json
from datetime import datetime, timedelta
import random
import os

def generate_events(output_dir="./input", events_per_batch=100, delay_seconds=5):
    """Generate streaming events to simulate real-time data"""
    os.makedirs(output_dir, exist_ok=True)
    batch = 0
    
    print(f"Starting event generator...")
    print(f"  Output: {output_dir}")
    print(f"  Events per batch: {events_per_batch}")
    print(f"  Delay: {delay_seconds}s")
    
    try:
        while True:
            events = []
            base_time = datetime.now()
            
            for i in range(events_per_batch):
                # Simulate some late-arriving events (10% of events)
                if random.random() < 0.1:
                    timestamp = base_time - timedelta(seconds=random.randint(60, 600))
                else:
                    timestamp = base_time - timedelta(seconds=random.randint(0, 30))
                
                event = {
                    "event_id": f"{batch}_{i}",
                    "timestamp": timestamp.isoformat(),
                    "user_id": random.randint(1, 1000),
                    "event_type": random.choice(["click", "view", "purchase", "add_to_cart"]),
                    "value": round(random.uniform(10, 1000), 2),
                    "session_id": f"session_{random.randint(1, 100)}"
                }
                events.append(event)
            
            # Write batch as JSON lines
            filename = f"{output_dir}/batch_{batch:06d}.json"
            with open(filename, "w") as f:
                for event in events:
                    f.write(json.dumps(event) + "\n")
            
            print(f"✓ Generated batch {batch} ({events_per_batch} events) -> {filename}")
            batch += 1
            time.sleep(delay_seconds)
            
    except KeyboardInterrupt:
        print(f"\n✓ Generator stopped. Generated {batch} batches.")

if __name__ == "__main__":
    generate_events()
```

### Step 5: Build Your First Streaming Query
```python
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, DoubleType, TimestampType
from pyspark.sql.functions import col, from_json

# Define schema for incoming events
schema = StructType([
    StructField("event_id", StringType(), False),
    StructField("timestamp", TimestampType(), False),
    StructField("user_id", IntegerType(), False),
    StructField("event_type", StringType(), False),
    StructField("value", DoubleType(), False),
    StructField("session_id", StringType(), False)
])

# Read streaming data
stream_df = spark.readStream \
    .format("json") \
    .schema(schema) \
    .option("maxFilesPerTrigger", 1) \
    .load("./input/")

print("✓ Streaming source configured")

# Write to Delta Lake (pass-through)
query = stream_df.writeStream \
    .format("delta") \
    .outputMode("append") \
    .option("checkpointLocation", "./checkpoints/raw_events") \
    .start("./data/events")

print("✓ Streaming query started")
print(f"  Query ID: {query.id}")
print(f"  Status: {query.status}")

# Let it run for a bit, then check
time.sleep(30)
query.stop()

# Query the Delta table
events = spark.read.format("delta").load("./data/events")
print(f"\n✓ Processed {events.count()} events")
events.show(10)
```

## Key Features to Implement

### 1. Basic Streaming (Pass-Through)
- Read from file stream
- Parse JSON events
- Write to Delta Lake
- Checkpoint management
- Exactly-once semantics

### 2. Windowed Aggregations

**Tumbling Windows** (non-overlapping):
```python
from pyspark.sql.functions import window, count, avg, sum

# 5-minute tumbling windows
tumbling = stream_df \
    .groupBy(
        window(col("timestamp"), "5 minutes"),
        col("event_type")
    ) \
    .agg(
        count("*").alias("event_count"),
        avg("value").alias("avg_value"),
        sum("value").alias("total_value")
    )

query_tumbling = tumbling.writeStream \
    .format("delta") \
    .outputMode("update") \
    .option("checkpointLocation", "./checkpoints/tumbling") \
    .start("./data/tumbling_aggregates")
```

**Sliding Windows** (overlapping):
```python
# 5-minute windows, sliding every 1 minute
sliding = stream_df \
    .groupBy(
        window(col("timestamp"), "5 minutes", "1 minute"),
        col("event_type")
    ) \
    .agg(
        count("*").alias("event_count"),
        avg("value").alias("avg_value")
    )

query_sliding = sliding.writeStream \
    .format("delta") \
    .outputMode("update") \
    .option("checkpointLocation", "./checkpoints/sliding") \
    .start("./data/sliding_aggregates")
```

### 3. Watermarking for Late Data
```python
from pyspark.sql.functions import window, col

# Allow late data up to 10 minutes
windowed_with_watermark = stream_df \
    .withWatermark("timestamp", "10 minutes") \
    .groupBy(
        window(col("timestamp"), "5 minutes"),
        col("event_type")
    ) \
    .agg(
        count("*").alias("event_count"),
        avg("value").alias("avg_value")
    )

# Late data within 10 minutes will update the window
# Data later than 10 minutes will be dropped
query_watermark = windowed_with_watermark.writeStream \
    .format("delta") \
    .outputMode("update") \
    .option("checkpointLocation", "./checkpoints/watermark") \
    .start("./data/watermarked_aggregates")
```

### 4. Stateful Processing (Deduplication)
```python
from pyspark.sql.functions import expr

# Deduplicate events within a 1-hour window
deduplicated = stream_df \
    .withWatermark("timestamp", "1 hour") \
    .dropDuplicates(["event_id", "timestamp"])

query_dedup = deduplicated.writeStream \
    .format("delta") \
    .outputMode("append") \
    .option("checkpointLocation", "./checkpoints/dedup") \
    .start("./data/deduplicated_events")
```

### 5. Stream-Static Join (Enrichment)
```python
# Load static dimension table
users_df = spark.createDataFrame([
    (1, "Alice", "Premium"),
    (2, "Bob", "Standard"),
    (3, "Charlie", "Premium")
], ["user_id", "name", "tier"])

users_df.write.format("delta").mode("overwrite").save("./data/users")

# Join stream with static data
users_static = spark.read.format("delta").load("./data/users")

enriched = stream_df.join(users_static, "user_id")

query_enriched = enriched.writeStream \
    .format("delta") \
    .outputMode("append") \
    .option("checkpointLocation", "./checkpoints/enriched") \
    .start("./data/enriched_events")
```

### 6. Session Windows (Gap-Based)
```python
from pyspark.sql.functions import session_window

# Group events into sessions with 30-minute inactivity gap
sessions = stream_df \
    .withWatermark("timestamp", "1 hour") \
    .groupBy(
        session_window(col("timestamp"), "30 minutes"),
        col("user_id")
    ) \
    .agg(
        count("*").alias("events_in_session"),
        sum("value").alias("session_value")
    )

query_sessions = sessions.writeStream \
    .format("delta") \
    .outputMode("update") \
    .option("checkpointLocation", "./checkpoints/sessions") \
    .start("./data/user_sessions")
```

## Success Criteria

By the end of this project, you should have:

- [ ] Spark Structured Streaming environment configured
- [ ] Data generator producing streaming events
- [ ] Pass-through streaming query working
- [ ] Tumbling window aggregations implemented
- [ ] Sliding window aggregations implemented
- [ ] Watermarking configured for late data
- [ ] Stateful deduplication working
- [ ] Stream-static join enriching events
- [ ] Session windows tracking user activity
- [ ] All queries writing to Delta Lake
- [ ] Checkpoint management understood
- [ ] Monitoring and query status tracking
- [ ] Comprehensive documentation
- [ ] GitHub repository with all code

## Learning Outcomes

After completing this project, you'll be able to:

- Understand Structured Streaming architecture
- Implement windowing operations (tumbling, sliding, session)
- Handle late-arriving data with watermarks
- Build stateful streaming applications
- Join streams with static data
- Manage checkpoints for fault tolerance
- Monitor streaming queries
- Optimize streaming performance
- Explain exactly-once semantics
- Compare Structured Streaming vs Kafka Streams

## Expected Results

**Streaming Performance**:
```
Input rate: 100 events/batch, 1 batch/5s = 20 events/second
Processing time: <1 second per batch
Latency: <2 seconds end-to-end
Throughput: Handles 1000+ events/second
```

**Windowing Results**:
```
Tumbling (5-min windows):
  Window 1: 6000 events (20 events/s * 300s)
  Window 2: 6000 events
  No overlap between windows

Sliding (5-min window, 1-min slide):
  Window 1: 6000 events
  Window 2: 4800 events (overlap with Window 1)
  More granular view of data
```

**Late Data Handling**:
```
Watermark: 10 minutes
Late event at T+5min: ✓ Accepted (within watermark)
Late event at T+15min: ✗ Dropped (beyond watermark)
```

## Project Structure

```
project-e-spark-streaming/
├── src/
│   ├── data_generator.py          # Event generator
│   ├── stream_passthrough.py      # Basic streaming
│   ├── stream_tumbling.py         # Tumbling windows
│   ├── stream_sliding.py          # Sliding windows
│   ├── stream_watermark.py        # Late data handling
│   ├── stream_dedup.py            # Deduplication
│   ├── stream_join.py             # Stream-static join
│   └── stream_sessions.py         # Session windows
├── notebooks/
│   └── streaming_demo.ipynb       # Interactive demo
├── input/                         # Streaming source files
│   └── batch_*.json
├── data/                          # Delta Lake tables
│   ├── events/
│   ├── tumbling_aggregates/
│   ├── sliding_aggregates/
│   ├── watermarked_aggregates/
│   ├── deduplicated_events/
│   ├── enriched_events/
│   └── user_sessions/
├── checkpoints/                   # Streaming checkpoints
│   ├── raw_events/
│   ├── tumbling/
│   ├── sliding/
│   └── ...
├── results/
│   ├── streaming_metrics.md
│   └── performance_charts.png
├── prd.md
├── tech-spec.md
├── implementation-plan.md
└── README.md
```

## Common Challenges & Solutions

### Challenge 1: Checkpoint Location Conflicts
**Problem**: Multiple queries using same checkpoint location
**Solution**: Use unique checkpoint location per query
```python
.option("checkpointLocation", "./checkpoints/unique_query_name")
```

### Challenge 2: State Size Growth
**Problem**: Stateful operations consuming too much memory
**Solution**: Use watermarks to bound state size
```python
.withWatermark("timestamp", "1 hour")  # Drop state older than 1 hour
```

### Challenge 3: Late Data Tuning
**Problem**: Too much late data dropped or too much state retained
**Solution**: Tune watermark based on data characteristics
```python
# Analyze your data latency first
# Set watermark slightly above P99 latency
.withWatermark("timestamp", "15 minutes")
```

### Challenge 4: Output Mode Selection
**Problem**: Wrong output mode causing errors
**Solution**: Choose based on operation type
```
- Append: Non-aggregations, aggregations with watermark
- Update: Aggregations with watermark
- Complete: Aggregations without watermark (small result sets only)
```

## Next Steps

After completing this project:

1. **Add to Portfolio**: Document on GitHub with streaming metrics
2. **Write Blog Post**: "Spark Structured Streaming: Windowing and Watermarks"
3. **Extend Features**: Add Kafka source, multiple sinks
4. **Build Project F**: Continue with Log Processor
5. **Production Use**: Apply streaming in real pipelines
6. **Advanced Topics**: Explore arbitrary stateful operations, stream-stream joins

## Resources

- [Structured Streaming Guide](https://spark.apache.org/docs/latest/structured-streaming-programming-guide.html)
- [Windowing Operations](https://spark.apache.org/docs/latest/structured-streaming-programming-guide.html#window-operations-on-event-time)
- [Watermarking](https://spark.apache.org/docs/latest/structured-streaming-programming-guide.html#handling-late-data-and-watermarking)
- [Output Modes](https://spark.apache.org/docs/latest/structured-streaming-programming-guide.html#output-modes)
- [Monitoring](https://spark.apache.org/docs/latest/structured-streaming-programming-guide.html#monitoring-streaming-queries)

## Questions?

If you get stuck:
1. Review the tech-spec.md for detailed streaming patterns
2. Check Spark Structured Streaming documentation
3. Search Spark community forums
4. Review the 100 Days bootcamp materials on streaming
5. Compare with Databricks Structured Streaming examples
