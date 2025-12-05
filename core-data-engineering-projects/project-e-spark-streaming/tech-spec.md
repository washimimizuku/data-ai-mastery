# Technical Specification: Spark Streaming Demo

## Architecture
```
Data Generator → File Stream → Spark Streaming → Delta Lake
                                      ↓
                              Windowed Aggregations
                                      ↓
                              Delta Lake (aggregates)
```

## Technology Stack
- **Python**: 3.11+
- **PySpark**: 3.5+
- **Delta Lake**: 2.4+
- **Structured Streaming**: Built into Spark

## Streaming Configuration

```python
from pyspark.sql import SparkSession

spark = SparkSession.builder \
    .appName("StreamingDemo") \
    .config("spark.jars.packages", "io.delta:delta-core_2.12:2.4.0") \
    .config("spark.sql.extensions", 
            "io.delta.sql.DeltaSparkSessionExtensions") \
    .config("spark.sql.catalog.spark_catalog", 
            "org.apache.spark.sql.delta.catalog.DeltaCatalog") \
    .config("spark.sql.streaming.checkpointLocation", "./checkpoints") \
    .getOrCreate()
```

## Streaming Examples

### Example 1: Simple Pass-Through
```python
# Read stream
stream_df = spark.readStream \
    .format("json") \
    .schema(schema) \
    .option("maxFilesPerTrigger", 1) \
    .load("./input/")

# Write to Delta
query = stream_df.writeStream \
    .format("delta") \
    .outputMode("append") \
    .option("checkpointLocation", "./checkpoints/raw") \
    .start("./data/events")

query.awaitTermination()
```

### Example 2: Windowed Aggregations
```python
from pyspark.sql.functions import window, col, count, avg

# Tumbling window (5 minutes)
windowed = stream_df \
    .withWatermark("timestamp", "10 minutes") \
    .groupBy(
        window(col("timestamp"), "5 minutes"),
        col("event_type")
    ) \
    .agg(
        count("*").alias("event_count"),
        avg("value").alias("avg_value")
    )

# Write aggregates
query = windowed.writeStream \
    .format("delta") \
    .outputMode("update") \
    .option("checkpointLocation", "./checkpoints/agg") \
    .start("./data/event_aggregates")
```

### Example 3: Late Data Handling
```python
# Watermark allows late data up to 10 minutes
stream_with_watermark = stream_df \
    .withWatermark("timestamp", "10 minutes") \
    .groupBy(
        window(col("timestamp"), "5 minutes", "1 minute"),  # Sliding window
        col("user_id")
    ) \
    .count()

query = stream_with_watermark.writeStream \
    .format("delta") \
    .outputMode("append") \
    .option("checkpointLocation", "./checkpoints/late") \
    .start("./data/user_activity")
```

### Example 4: Stream-Static Join
```python
# Static dimension table
users_df = spark.read.format("delta").load("./data/users")

# Join stream with static data
enriched = stream_df.join(users_df, "user_id")

query = enriched.writeStream \
    .format("delta") \
    .outputMode("append") \
    .option("checkpointLocation", "./checkpoints/enriched") \
    .start("./data/enriched_events")
```

## Data Generator

```python
import time
import json
from datetime import datetime, timedelta
import random

def generate_events(output_dir, events_per_batch=100, delay_seconds=5):
    """Generate streaming events"""
    batch = 0
    while True:
        events = []
        for i in range(events_per_batch):
            event = {
                "event_id": f"{batch}_{i}",
                "timestamp": (datetime.now() - timedelta(seconds=random.randint(0, 60))).isoformat(),
                "user_id": random.randint(1, 1000),
                "event_type": random.choice(["click", "view", "purchase"]),
                "value": random.uniform(10, 1000)
            }
            events.append(event)
        
        # Write batch
        with open(f"{output_dir}/batch_{batch}.json", "w") as f:
            for event in events:
                f.write(json.dumps(event) + "\n")
        
        print(f"Generated batch {batch} with {events_per_batch} events")
        batch += 1
        time.sleep(delay_seconds)
```

## Monitoring & Querying

```python
# Query streaming table
current_data = spark.read.format("delta").load("./data/events")
current_data.show()

# View streaming query status
query.status

# View recent progress
query.recentProgress

# Stop query
query.stop()
```

## Checkpointing

Checkpoints enable fault tolerance:
- Store offset information
- Track processed data
- Enable exactly-once semantics
- Allow restart from failure

```python
# Checkpoint location per query
.option("checkpointLocation", "./checkpoints/query_name")
```

## Output Modes

1. **Append**: Only new rows (for non-aggregations)
2. **Update**: Updated rows (for aggregations with watermark)
3. **Complete**: Entire result table (for aggregations without watermark)

## Project Structure
```
project-e-spark-streaming/
├── src/
│   ├── data_generator.py
│   ├── stream_passthrough.py
│   ├── stream_aggregations.py
│   ├── stream_late_data.py
│   └── stream_join.py
├── notebooks/
│   └── streaming_demo.ipynb
├── input/              # Streaming source
├── data/               # Delta tables
│   ├── events/
│   ├── event_aggregates/
│   └── user_activity/
├── checkpoints/        # Streaming checkpoints
└── README.md
```

## Testing Strategy
- Test with small batches
- Verify exactly-once semantics
- Test late data handling
- Validate aggregations
- Test failure recovery

## Performance Considerations
- Tune maxFilesPerTrigger
- Optimize trigger intervals
- Manage state size
- Monitor checkpoint size
- Use appropriate watermarks
