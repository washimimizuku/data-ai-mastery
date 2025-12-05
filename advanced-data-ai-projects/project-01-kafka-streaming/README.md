# Project 1: Real-Time Event Streaming Platform with Kafka

## Objective

Build a production-grade real-time event streaming platform demonstrating enterprise-level streaming architecture, event-driven design patterns, and modern data engineering practices.

**What You'll Build**: A complete streaming platform with Kafka, FastAPI microservices, Spark Structured Streaming, Snowflake integration, and comprehensive monitoring.

**What You'll Learn**: Kafka architecture, stream processing, FastAPI, Schema Registry, exactly-once semantics, and production streaming patterns.

## Time Estimate

**2-3 months (160-240 hours)**

- Weeks 1-2: Kafka setup, producers, consumers (40-60h)
- Weeks 3-4: Stream processing with Spark/Flink (40-60h)
- Weeks 5-6: FastAPI microservices, Schema Registry (40-60h)
- Weeks 7-8: Snowflake integration, monitoring (40-60h)

## Prerequisites

### Required Knowledge
- [100 Days Data & AI](https://github.com/washimimizuku/100-days-data-ai) - Days 1-50
  - Days 21-30: Kafka fundamentals
  - Days 31-40: Spark and streaming
  - Days 41-50: Airflow and orchestration
- [60 Days Advanced](https://github.com/washimimizuku/60-days-advanced-data-ai) - Days 1-24
  - Days 1-10: Advanced Kafka patterns
  - Days 15-21: Production orchestration

### Technical Requirements
- Docker and Docker Compose
- Python 3.9+
- Understanding of distributed systems
- SQL knowledge
- AWS account (optional, for cloud deployment)

## Architecture Overview

### System Components

```
Event Sources → FastAPI → Kafka Cluster → Stream Processing → Snowflake
                            ↓                    ↓
                      Schema Registry      Monitoring Stack
```

**Core Components:**
- **Kafka Cluster**: 3-broker setup with Zookeeper
- **Schema Registry**: Centralized schema management
- **FastAPI Services**: Event ingestion and consumption APIs
- **Stream Processor**: Spark Structured Streaming or Flink
- **Data Warehouse**: Snowflake via Kafka Connect
- **Monitoring**: Prometheus + Grafana

### Technology Stack

**Streaming Layer:**
- Apache Kafka 3.6+ (event streaming)
- Confluent Schema Registry (schema management)
- Kafka Connect (data integration)

**Processing Layer:**
- Apache Spark 3.5+ (batch + streaming)
- Apache Flink 1.18+ (alternative stream processor)

**Application Layer:**
- FastAPI (Python microservices)
- Pydantic (data validation)
- kafka-python or confluent-kafka-python

**Storage Layer:**
- Snowflake (data warehouse)
- PostgreSQL (metadata store)

**Infrastructure:**
- Docker + Docker Compose (local development)
- Kubernetes (production deployment)
- Terraform (IaC)

## Getting Started

### Step 1: Review Documentation
1. `prd.md` - Product requirements and goals
2. `tech-spec.md` - Technical architecture details
3. `implementation-plan.md` - Week-by-week implementation guide

### Step 2: Set Up Local Environment
```bash
# Clone and set up project
git clone <your-repo>
cd project-01-kafka-streaming

# Start Kafka cluster with Docker Compose
docker-compose up -d

# Install Python dependencies
pip install -r requirements.txt
```

### Step 3: Verify Kafka Setup
```bash
# Check Kafka is running
docker ps

# Create test topic
kafka-topics --create --topic test --bootstrap-server localhost:9092

# Test producer/consumer
kafka-console-producer --topic test --bootstrap-server localhost:9092
kafka-console-consumer --topic test --from-beginning --bootstrap-server localhost:9092
```

## Core Implementation

### 1. Kafka Infrastructure

**Multi-Broker Cluster:**
- 3 Kafka brokers for high availability
- Zookeeper ensemble (3 nodes)
- Topic partitioning strategy (by key hash)
- Replication factor: 3 (production standard)
- Min in-sync replicas: 2

**Topic Design:**
- Naming convention: `<domain>.<entity>.<event-type>`
- Partition count based on throughput requirements
- Retention policy: 7 days (configurable)
- Cleanup policy: delete or compact

### 2. Schema Registry

**Schema Management:**
- Avro schemas for all events
- Schema versioning and evolution
- Compatibility modes: BACKWARD, FORWARD, FULL
- Schema validation on produce/consume

**Schema Evolution Patterns:**
- Add optional fields (backward compatible)
- Remove optional fields (forward compatible)
- Rename fields with aliases (full compatible)

### 3. Stream Processing

**Spark Structured Streaming:**
- Micro-batch processing (1-5 second intervals)
- Windowing: tumbling (fixed), sliding (overlapping), session (gap-based)
- Stateful operations: aggregations, joins, deduplication
- Watermarking for late data handling
- Exactly-once semantics with checkpointing

**Processing Patterns:**
- Event-time processing (not processing-time)
- Stateful aggregations with state store
- Stream-stream joins with watermarks
- Stream-static joins for enrichment

### 4. FastAPI Microservices

**Event Publishing Service:**
- POST `/events/{topic}` - Publish events
- Async Kafka producer with batching
- Schema validation before publishing
- Error handling with retries

**Event Consumption Service:**
- GET `/events/{topic}` - Consume events (SSE)
- WebSocket support for real-time streaming
- Consumer group management
- Offset management (auto-commit vs manual)

**Health & Metrics:**
- `/health` - Service health check
- `/metrics` - Prometheus metrics
- `/docs` - OpenAPI documentation

### 5. Data Sink Integration

**Kafka Connect:**
- Snowflake Sink Connector
- Batch size: 10,000 records
- Flush interval: 60 seconds
- Error handling: DLQ (dead letter queue)
- Transformations: SMT (Single Message Transforms)

**Data Flow:**
- Kafka → Kafka Connect → Snowflake staging → Snowpipe → Tables
- Schema mapping: Avro → Snowflake types
- Partitioning strategy in Snowflake

### 6. Monitoring & Observability

**Metrics Collection:**
- Kafka broker metrics (JMX)
- Producer/consumer metrics
- Stream processing metrics
- Application metrics (FastAPI)

**Dashboards:**
- Kafka cluster health
- Topic throughput and lag
- Consumer group lag
- Stream processing latency
- Error rates and DLQ monitoring

## Integration Points

### Kafka → Spark Streaming
- Read from Kafka topics with `readStream`
- Process with DataFrame API
- Write to Kafka or external sinks
- Checkpoint to HDFS or S3

### Kafka → Snowflake
- Kafka Connect Snowflake Sink
- Automatic table creation
- Schema evolution handling
- Snowpipe for continuous loading

### FastAPI → Kafka
- Async producer for non-blocking
- Batching for throughput
- Compression (snappy or lz4)
- Idempotent producer for exactly-once

## Performance Targets

**Throughput:**
- Producer: 100K+ messages/second
- Consumer: 100K+ messages/second
- Stream processing: 50K+ events/second

**Latency:**
- End-to-end: <5 seconds (p95)
- Producer ack: <100ms (p95)
- Consumer fetch: <50ms (p95)

**Availability:**
- Kafka cluster: 99.9% uptime
- Zero data loss with replication
- Automatic failover <30 seconds

## Success Criteria

- [ ] Multi-broker Kafka cluster running
- [ ] Schema Registry with Avro schemas
- [ ] Producers with error handling and retries
- [ ] Consumer groups with parallel processing
- [ ] Stream processing with windowing
- [ ] FastAPI microservices deployed
- [ ] Snowflake integration with Kafka Connect
- [ ] Monitoring with Prometheus and Grafana
- [ ] Comprehensive documentation and diagrams
- [ ] Performance benchmarks (throughput, latency)

## Learning Outcomes

- Design and implement Kafka architectures
- Build producers and consumers with best practices
- Implement stream processing with Spark/Flink
- Create FastAPI microservices for event-driven systems
- Integrate Kafka with data warehouses
- Monitor and troubleshoot streaming pipelines
- Explain exactly-once semantics
- Handle schema evolution

## Next Steps

1. Add to portfolio with architecture diagrams
2. Write blog post: "Building Production Kafka Pipelines"
3. Continue to Project 2: Rust + FastAPI pipeline
4. Extend with Kafka Streams API or ksqlDB

## Resources

- [Apache Kafka Docs](https://kafka.apache.org/documentation/)
- [Confluent Schema Registry](https://docs.confluent.io/platform/current/schema-registry/index.html)
- [Spark Structured Streaming](https://spark.apache.org/docs/latest/structured-streaming-programming-guide.html)
- [FastAPI Docs](https://fastapi.tiangolo.com/)
- [Kafka Connect](https://docs.confluent.io/platform/current/connect/index.html)
