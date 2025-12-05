# Design Document: Real-Time Event Streaming Platform with Kafka

## Overview

The Real-Time Event Streaming Platform is a production-grade system that ingests events from multiple sources, processes them in real-time, and delivers processed data to Snowflake for analytics. The platform leverages Apache Kafka as the central event streaming backbone, FastAPI for REST API services, Apache Flink for stream processing, and integrates with AWS cloud services for production deployment.

The system is designed to handle high-throughput event streams (10,000+ events/second) with low latency (p95 < 1 second) while maintaining reliability through exactly-once processing semantics, comprehensive error handling, and robust monitoring.

## Architecture

### High-Level Architecture

```
┌─────────────────┐
│  Data Sources   │
│ (IoT, Click,    │
│  Transactions)  │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  FastAPI        │
│  Service        │◄──── REST API Clients
│  (Event API)    │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Kafka Cluster  │
│  - Topics       │
│  - Partitions   │
│  - Replication  │
└────┬────────────┘
     │
     ├──────────────────┐
     ▼                  ▼
┌─────────────┐   ┌──────────────┐
│   Stream    │   │    Kafka     │
│  Processor  │   │   Connect    │
│  (Flink)    │   │  (Snowflake  │
│             │   │    Sink)     │
└─────┬───────┘   └──────┬───────┘
      │                  │
      │                  ▼
      │           ┌──────────────┐
      │           │  Snowflake   │
      │           │  Data        │
      │           │  Warehouse   │
      │           └──────────────┘
      │
      └──────────────────┐
                         ▼
                  ┌──────────────┐
                  │  Monitoring  │
                  │  (Prometheus │
                  │  + Grafana)  │
                  └──────────────┘
```

### Component Interaction Flow

1. **Event Ingestion**: External systems send events to FastAPI Service via REST API
2. **Validation & Publishing**: FastAPI validates events against Schema Registry and publishes to Kafka
3. **Stream Processing**: Flink consumes from Kafka, applies transformations, and writes results back to Kafka
4. **Data Warehousing**: Kafka Connect consumes processed events and loads them into Snowflake
5. **Monitoring**: Prometheus scrapes metrics from all components, Grafana visualizes them

### AWS Deployment Architecture

```
┌─────────────────────────────────────────────────────────┐
│                         VPC                              │
│                                                          │
│  ┌────────────────┐         ┌──────────────────────┐   │
│  │  Public Subnet │         │   Private Subnet     │   │
│  │                │         │                      │   │
│  │  ┌──────────┐  │         │  ┌────────────────┐ │   │
│  │  │   ALB    │  │         │  │  Amazon MSK    │ │   │
│  │  └────┬─────┘  │         │  │  (Kafka)       │ │   │
│  │       │        │         │  └────────────────┘ │   │
│  └───────┼────────┘         │                      │   │
│          │                  │  ┌────────────────┐ │   │
│          │                  │  │  ECS Fargate   │ │   │
│          └──────────────────┼─►│  (FastAPI)     │ │   │
│                             │  └────────────────┘ │   │
│                             │                      │   │
│                             │  ┌────────────────┐ │   │
│                             │  │  ECS Fargate   │ │   │
│                             │  │  (Kafka        │ │   │
│                             │  │   Connect)     │ │   │
│                             │  └────────────────┘ │   │
│                             └──────────────────────┘   │
└─────────────────────────────────────────────────────────┘
                                      │
                                      ▼
                              ┌──────────────┐
                              │  Snowflake   │
                              │  (External)  │
                              └──────────────┘
```

## Components and Interfaces

### 1. FastAPI Service

**Responsibilities:**
- Accept event submissions via REST API
- Validate events against schemas
- Publish events to Kafka topics
- Provide event query capabilities
- Expose health and metrics endpoints
- Stream events via WebSocket

**Interfaces:**

```python
# REST API Endpoints
POST   /api/v1/events/{topic}          # Publish event
GET    /api/v1/events/{topic}/latest   # Get recent events
GET    /api/v1/metrics                 # Service metrics
GET    /api/v1/health                  # Health check
WS     /api/v1/stream/{topic}          # WebSocket stream

# Request/Response Models
class Event(BaseModel):
    event_id: str
    timestamp: datetime
    event_type: str
    user_id: Optional[str]
    payload: dict
    metadata: Optional[dict]

class EventResponse(BaseModel):
    status: str
    event_id: str
    topic: str
    partition: int
    offset: int
    timestamp: datetime

class HealthResponse(BaseModel):
    status: str
    kafka_connected: bool
    schema_registry_connected: bool
    uptime_seconds: float
```

**Configuration:**
- Async request handling with asyncio
- Connection pooling for Kafka producers
- JWT authentication middleware
- Rate limiting (100 requests/minute per client)
- CORS configuration for web clients

### 2. Kafka Cluster

**Topic Design:**

```
events.iot.sensors      (12 partitions, replication factor 3)
events.clickstream      (24 partitions, replication factor 3)
events.transactions     (12 partitions, replication factor 3)
events.aggregated       (12 partitions, replication factor 3)
events.dlq              (6 partitions, replication factor 3)
```

**Partitioning Strategy:**
- Key-based partitioning using `user_id` or `device_id`
- Ensures message ordering within partition
- Load balancing across partitions

**Producer Configuration:**
```properties
acks=all                                    # Wait for all replicas
compression.type=snappy                     # Balance speed/size
max.in.flight.requests.per.connection=5     # Pipeline requests
enable.idempotence=true                     # Prevent duplicates
linger.ms=10                                # Batch for throughput
batch.size=16384                            # Batch size in bytes
```

**Consumer Configuration:**
```properties
enable.auto.commit=false                    # Manual offset management
isolation.level=read_committed              # Read only committed messages
max.poll.records=500                        # Records per poll
fetch.min.bytes=1024                        # Minimum fetch size
```

### 3. Schema Registry

**Responsibilities:**
- Store and version Avro/Protobuf schemas
- Validate event schemas
- Support schema evolution
- Provide schema compatibility checking

**Schema Example (Avro):**
```json
{
  "type": "record",
  "name": "Event",
  "namespace": "com.platform.events",
  "fields": [
    {"name": "event_id", "type": "string"},
    {"name": "timestamp", "type": "long"},
    {"name": "event_type", "type": "string"},
    {"name": "user_id", "type": ["null", "string"], "default": null},
    {"name": "device_id", "type": ["null", "string"], "default": null},
    {"name": "payload", "type": "string"}
  ]
}
```

### 4. Stream Processor (Apache Flink)

**Responsibilities:**
- Consume events from Kafka topics
- Filter invalid/malformed events
- Enrich events with additional metadata
- Apply windowing operations
- Compute aggregations
- Handle late-arriving data
- Write results to output topics

**Processing Pipeline:**
```
DataStream<Event>
  → filter(isValid)
  → map(enrichEvent)
  → keyBy(event -> event.userId)
  → window(TumblingEventTimeWindows.of(Time.minutes(1)))
  → aggregate(new EventAggregator())
  → addSink(kafkaSink)
```

**Windowing Strategies:**
- Tumbling windows: 1-minute, 5-minute, 15-minute
- Sliding windows: 5-minute window, 1-minute slide
- Session windows: 30-minute gap
- Watermark strategy: 10-second max out-of-orderness

**Aggregation Operations:**
- Count events by type
- Sum transaction amounts
- Average sensor readings
- Min/max values per window
- Distinct user counts

### 5. Kafka Connect (Snowflake Sink)

**Responsibilities:**
- Consume processed events from Kafka
- Transform events for Snowflake schema
- Batch events for efficient loading
- Load data into Snowflake tables
- Handle errors and retries

**Connector Configuration:**
```json
{
  "name": "snowflake-sink-connector",
  "config": {
    "connector.class": "com.snowflake.kafka.connector.SnowflakeSinkConnector",
    "topics": "events.aggregated",
    "snowflake.url.name": "${SNOWFLAKE_URL}",
    "snowflake.user.name": "${SNOWFLAKE_USER}",
    "snowflake.private.key": "${SNOWFLAKE_PRIVATE_KEY}",
    "snowflake.database.name": "EVENTS_DB",
    "snowflake.schema.name": "RAW",
    "buffer.count.records": "10000",
    "buffer.flush.time": "60",
    "buffer.size.bytes": "5000000",
    "tasks.max": "4"
  }
}
```

### 6. Monitoring Stack

**Prometheus Metrics:**
```
# Kafka metrics
kafka_server_broker_topic_metrics_messages_in_per_sec
kafka_consumer_lag_seconds
kafka_producer_record_send_rate

# FastAPI metrics
http_request_duration_seconds
http_requests_total
http_requests_in_progress

# Flink metrics
flink_taskmanager_job_task_operator_records_consumed_rate
flink_taskmanager_job_task_operator_records_emitted_rate
flink_taskmanager_job_task_operator_latency
```

**Grafana Dashboards:**
1. Kafka Cluster Overview (broker health, topic metrics)
2. Consumer Lag Dashboard (lag by topic/partition)
3. API Performance (request rate, latency, errors)
4. Stream Processing (throughput, latency, backpressure)

**Alert Rules:**
- Consumer lag > 100,000 messages
- API error rate > 1%
- Kafka broker down
- DLQ message count increasing
- End-to-end latency > 2 seconds (p95)

## Data Models

### Event Model (Core)

```python
from datetime import datetime
from typing import Optional, Dict, Any
from pydantic import BaseModel, Field
from enum import Enum

class EventType(str, Enum):
    IOT_SENSOR = "iot.sensor"
    CLICKSTREAM = "clickstream"
    TRANSACTION = "transaction"

class Event(BaseModel):
    event_id: str = Field(..., description="Unique event identifier")
    timestamp: datetime = Field(..., description="Event occurrence time")
    event_type: EventType = Field(..., description="Type of event")
    user_id: Optional[str] = Field(None, description="User identifier")
    device_id: Optional[str] = Field(None, description="Device identifier")
    payload: Dict[str, Any] = Field(..., description="Event-specific data")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")
    
    class Config:
        json_schema_extra = {
            "example": {
                "event_id": "evt_123456",
                "timestamp": "2024-01-15T10:30:00Z",
                "event_type": "iot.sensor",
                "user_id": "user_789",
                "device_id": "device_456",
                "payload": {
                    "temperature": 22.5,
                    "humidity": 65.0
                }
            }
        }
```

### Aggregated Event Model

```python
class AggregatedEvent(BaseModel):
    window_start: datetime
    window_end: datetime
    event_type: EventType
    count: int
    sum_value: Optional[float]
    avg_value: Optional[float]
    min_value: Optional[float]
    max_value: Optional[float]
    distinct_users: int
```

### Snowflake Table Schema

```sql
CREATE TABLE events_db.raw.events (
    event_id VARCHAR(255) PRIMARY KEY,
    timestamp TIMESTAMP_NTZ,
    event_type VARCHAR(100),
    user_id VARCHAR(255),
    device_id VARCHAR(255),
    payload VARIANT,
    metadata VARIANT,
    ingestion_time TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP()
);

CREATE TABLE events_db.analytics.aggregated_events (
    window_start TIMESTAMP_NTZ,
    window_end TIMESTAMP_NTZ,
    event_type VARCHAR(100),
    count INTEGER,
    sum_value FLOAT,
    avg_value FLOAT,
    min_value FLOAT,
    max_value FLOAT,
    distinct_users INTEGER,
    processed_time TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP(),
    PRIMARY KEY (window_start, event_type)
);
```


## Correctness Properties

*A property is a characteristic or behavior that should hold true across all valid executions of a system-essentially, a formal statement about what the system should do. Properties serve as the bridge between human-readable specifications and machine-verifiable correctness guarantees.*

### Property 1: Schema validation consistency
*For any* event submitted to the FastAPI Service, the event should be validated against the registered schema in the Schema Registry, and only events that match the schema should be accepted.
**Validates: Requirements 1.1**

### Property 2: Event routing correctness
*For any* valid event received by the FastAPI Service, the event should be published to the Kafka topic that corresponds to its event_type field.
**Validates: Requirements 1.2**

### Property 3: Multi-format support
*For any* event in JSON, Avro, or Protobuf format, the Event Streaming Platform should successfully parse and process the event.
**Validates: Requirements 1.3, 1.4, 1.5**

### Property 4: Partition key consistency
*For any* two events with the same partition key (user_id or device_id), both events should be routed to the same Kafka partition.
**Validates: Requirements 2.1**

### Property 5: Dead letter queue routing
*For any* message that fails processing after maximum retry attempts, the message should appear in the Dead Letter Queue topic.
**Validates: Requirements 2.4, 7.2**

### Property 6: Exactly-once offset semantics
*For any* message successfully processed by a consumer, if the consumer fails before committing the offset, the message should be reprocessed exactly once after recovery.
**Validates: Requirements 2.5, 7.3**

### Property 7: Invalid event filtering
*For any* stream of events containing both valid and invalid events, the Stream Processor should emit only the valid events to downstream topics.
**Validates: Requirements 3.2**

### Property 8: Window assignment correctness
*For any* event with a timestamp, the Stream Processor should assign the event to all windows that contain that timestamp according to the window definition (tumbling or sliding).
**Validates: Requirements 3.3**

### Property 9: Aggregation computation correctness
*For any* window of events, the computed aggregations (count, sum, average) should match the mathematical result of applying those operations to all events in the window.
**Validates: Requirements 3.4**

### Property 10: Late event handling
*For any* event arriving after the watermark has passed its timestamp, the Stream Processor should handle the event according to the configured policy (drop, include in late results, or update existing window).
**Validates: Requirements 3.5**

### Property 11: API response completeness
*For any* successful POST request to the events endpoint, the response should contain the event_id, topic name, partition number, and offset.
**Validates: Requirements 4.1**

### Property 12: Recent events query correctness
*For any* GET request to the latest events endpoint, the returned events should be from the specified topic and ordered by timestamp in descending order.
**Validates: Requirements 4.2**

### Property 13: WebSocket streaming consistency
*For any* WebSocket connection to a topic, all events published to that topic after connection establishment should be streamed to the client in order.
**Validates: Requirements 4.5**

### Property 14: Snowflake transformation correctness
*For any* message received by the Snowflake Sink, the transformed message should contain all required fields for the target Snowflake table schema.
**Validates: Requirements 5.2**

### Property 15: Snowflake data persistence
*For any* message successfully processed by the Snowflake Sink, querying the Snowflake table should return a row with the event data.
**Validates: Requirements 5.3**

### Property 16: Retry exponential backoff
*For any* failed operation (producer publish or sink write), retry attempts should occur with exponentially increasing delays between attempts.
**Validates: Requirements 5.4, 7.1**

### Property 17: Buffer flush threshold
*For any* sequence of messages sent to the Snowflake Sink, when the buffer count reaches the configured threshold, all buffered messages should be flushed to Snowflake.
**Validates: Requirements 5.5**

### Property 18: Rebalancing data integrity
*For any* consumer group rebalancing event, all messages should be processed exactly once with no data loss.
**Validates: Requirements 8.3**

### Property 19: Dynamic topic creation
*For any* event with a new event_type not previously seen, the Event Streaming Platform should create a corresponding Kafka topic if configured for dynamic topic creation.
**Validates: Requirements 8.4**

### Property 20: JWT authentication enforcement
*For any* request to the FastAPI Service without a valid JWT token, the request should be rejected with a 401 Unauthorized status.
**Validates: Requirements 9.3**

### Property 21: Message delivery guarantee
*For any* large batch of events published to the platform, at least 99.9% of the events should be successfully delivered to Snowflake.
**Validates: Requirements 11.4**

## Error Handling

### Error Categories

**1. Validation Errors**
- Invalid schema
- Missing required fields
- Type mismatches
- **Handling**: Return 400 Bad Request with detailed error message

**2. Authentication/Authorization Errors**
- Invalid JWT token
- Expired token
- Missing credentials
- **Handling**: Return 401 Unauthorized or 403 Forbidden

**3. Kafka Producer Errors**
- Broker unavailable
- Topic doesn't exist
- Serialization failure
- **Handling**: Retry with exponential backoff (max 3 attempts), then return 503 Service Unavailable

**4. Consumer Processing Errors**
- Deserialization failure
- Business logic exception
- Downstream service unavailable
- **Handling**: Route to DLQ after 3 retry attempts, log error with context

**5. Stream Processing Errors**
- Late data beyond allowed lateness
- State corruption
- Checkpoint failure
- **Handling**: Log error, emit to side output for late data, alert on checkpoint failures

**6. Sink Errors**
- Snowflake connection failure
- Schema mismatch
- Constraint violation
- **Handling**: Retry with exponential backoff (max 5 attempts), route to DLQ if persistent

### Error Response Format

```python
class ErrorResponse(BaseModel):
    error_code: str
    message: str
    details: Optional[Dict[str, Any]]
    timestamp: datetime
    request_id: str

# Example
{
    "error_code": "SCHEMA_VALIDATION_FAILED",
    "message": "Event does not match registered schema",
    "details": {
        "missing_fields": ["user_id"],
        "schema_version": "1.2.0"
    },
    "timestamp": "2024-01-15T10:30:00Z",
    "request_id": "req_abc123"
}
```

### Retry Strategy

```python
def exponential_backoff(attempt: int, base_delay: float = 1.0, max_delay: float = 60.0) -> float:
    """Calculate delay with exponential backoff and jitter"""
    delay = min(base_delay * (2 ** attempt), max_delay)
    jitter = random.uniform(0, delay * 0.1)
    return delay + jitter

# Retry configuration
MAX_RETRIES = 3
BASE_DELAY = 1.0  # seconds
MAX_DELAY = 60.0  # seconds
```

### Dead Letter Queue Processing

**DLQ Message Format:**
```python
class DLQMessage(BaseModel):
    original_message: bytes
    error_type: str
    error_message: str
    retry_count: int
    first_attempt_time: datetime
    last_attempt_time: datetime
    source_topic: str
    source_partition: int
    source_offset: int
```

**DLQ Monitoring:**
- Alert when DLQ message count > 100
- Daily report of DLQ messages by error type
- Manual review process for DLQ messages
- Reprocessing capability for fixed issues

## Testing Strategy

### Unit Testing

**Framework**: pytest with pytest-asyncio for async tests

**Coverage Areas:**
- Event validation logic
- Schema serialization/deserialization
- Partition key calculation
- Aggregation functions
- Error handling paths
- API request/response models

**Example Unit Tests:**
```python
def test_event_validation_valid_event():
    """Test that valid events pass validation"""
    event = Event(
        event_id="evt_123",
        timestamp=datetime.now(),
        event_type=EventType.IOT_SENSOR,
        payload={"temperature": 22.5}
    )
    assert validate_event(event) == True

def test_partition_key_calculation():
    """Test partition key is derived from user_id"""
    event = Event(user_id="user_123", ...)
    assert get_partition_key(event) == "user_123"
```

### Property-Based Testing

**Framework**: Hypothesis (Python)

**Configuration**: Each property test should run a minimum of 100 iterations

**Test Tagging**: Each property-based test must include a comment with the format:
`# Feature: kafka-event-streaming-platform, Property {number}: {property_text}`

**Coverage Areas:**
- Schema validation across random event structures
- Event routing with random event types
- Partition assignment with random keys
- Aggregation correctness with random event streams
- Window assignment with random timestamps
- Error handling with random failure scenarios

**Example Property Tests:**
```python
from hypothesis import given, strategies as st

@given(st.builds(Event))
def test_property_1_schema_validation_consistency(event):
    """
    Feature: kafka-event-streaming-platform, Property 1: Schema validation consistency
    For any event, validation should be consistent with schema
    """
    is_valid = validate_against_schema(event)
    if is_valid:
        assert can_serialize(event)
    else:
        with pytest.raises(ValidationError):
            serialize_event(event)

@given(st.lists(st.builds(Event), min_size=10, max_size=100))
def test_property_9_aggregation_correctness(events):
    """
    Feature: kafka-event-streaming-platform, Property 9: Aggregation computation correctness
    For any window of events, aggregations should match mathematical result
    """
    window = create_window(events)
    result = compute_aggregations(window)
    
    assert result.count == len(events)
    assert result.sum_value == sum(e.payload.get('value', 0) for e in events)
    assert result.avg_value == result.sum_value / result.count
```

### Integration Testing

**Framework**: pytest with testcontainers for Kafka, Schema Registry

**Coverage Areas:**
- End-to-end event flow (API → Kafka → Flink → Snowflake)
- Kafka Connect integration
- Schema Registry integration
- Authentication flow
- Error scenarios (broker down, network issues)

**Example Integration Test:**
```python
@pytest.mark.integration
async def test_end_to_end_event_flow(kafka_container, schema_registry_container):
    """Test complete event flow from API to Kafka"""
    # Publish event via API
    response = await client.post("/api/v1/events/iot.sensors", json=event_data)
    assert response.status_code == 200
    
    # Verify event in Kafka
    consumer = create_consumer()
    messages = consumer.poll(timeout=5.0)
    assert len(messages) == 1
    assert messages[0].value() == event_data
```

### Load Testing

**Framework**: Locust

**Scenarios:**
- Sustained load: 10,000 events/second for 10 minutes
- Spike load: Ramp from 1,000 to 50,000 events/second
- Endurance: 5,000 events/second for 1 hour

**Metrics to Measure:**
- Request latency (p50, p95, p99)
- Throughput (events/second)
- Error rate
- Consumer lag
- End-to-end latency

**Example Load Test:**
```python
from locust import HttpUser, task, between

class EventPublisher(HttpUser):
    wait_time = between(0.01, 0.1)
    
    @task
    def publish_event(self):
        event = generate_random_event()
        self.client.post(
            "/api/v1/events/iot.sensors",
            json=event,
            headers={"Authorization": f"Bearer {self.token}"}
        )
```

### Performance Testing

**Tools:**
- kafka-producer-perf-test for Kafka throughput
- kafka-consumer-perf-test for consumer throughput
- Custom scripts for end-to-end latency

**Benchmarks:**
- Producer throughput: Target 50,000 msgs/sec
- Consumer throughput: Target 40,000 msgs/sec
- API latency: p95 < 100ms
- End-to-end latency: p95 < 1 second

## Deployment

### Local Development

**Prerequisites:**
- Docker and Docker Compose
- Python 3.11+
- Poetry or pip

**Setup:**
```bash
# Start infrastructure
docker-compose up -d

# Install dependencies
poetry install

# Run FastAPI service
poetry run uvicorn app.main:app --reload --port 8000

# Run Flink job
./bin/flink run -c com.platform.StreamProcessor target/stream-processor.jar
```

**Docker Compose Services:**
- Kafka (3 brokers)
- Zookeeper
- Schema Registry
- Kafka Connect
- Prometheus
- Grafana
- Flink JobManager
- Flink TaskManager

### AWS Production Deployment

**Infrastructure as Code**: Terraform

**Components:**
1. **VPC**: 3 availability zones, public and private subnets
2. **Amazon MSK**: 3 brokers (kafka.m5.large), encryption enabled
3. **ECS Cluster**: Fargate launch type
4. **Application Load Balancer**: HTTPS with ACM certificate
5. **CloudWatch**: Log groups for all services
6. **Secrets Manager**: Store credentials and keys

**Deployment Process:**
```bash
# 1. Provision infrastructure
cd terraform
terraform init
terraform plan
terraform apply

# 2. Build and push Docker images
docker build -t fastapi-service:latest .
docker tag fastapi-service:latest ${ECR_REPO}/fastapi-service:latest
docker push ${ECR_REPO}/fastapi-service:latest

# 3. Deploy ECS services
aws ecs update-service --cluster event-platform --service fastapi-service --force-new-deployment

# 4. Deploy Kafka Connect
aws ecs run-task --cluster event-platform --task-definition kafka-connect

# 5. Submit Flink job
./bin/flink run -m yarn-cluster -c com.platform.StreamProcessor target/stream-processor.jar
```

### CI/CD Pipeline

**GitHub Actions Workflow:**
```yaml
name: Deploy Event Platform

on:
  push:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Run unit tests
        run: poetry run pytest tests/unit
      - name: Run integration tests
        run: poetry run pytest tests/integration

  build:
    needs: test
    runs-on: ubuntu-latest
    steps:
      - name: Build Docker image
        run: docker build -t fastapi-service:${{ github.sha }} .
      - name: Push to ECR
        run: |
          aws ecr get-login-password | docker login --username AWS --password-stdin ${ECR_REPO}
          docker push ${ECR_REPO}/fastapi-service:${{ github.sha }}

  deploy:
    needs: build
    runs-on: ubuntu-latest
    steps:
      - name: Deploy to ECS
        run: |
          aws ecs update-service \
            --cluster event-platform \
            --service fastapi-service \
            --force-new-deployment
```

### Configuration Management

**Environment Variables:**
```bash
# Kafka
KAFKA_BOOTSTRAP_SERVERS=localhost:9092
KAFKA_SECURITY_PROTOCOL=SASL_SSL
KAFKA_SASL_MECHANISM=SCRAM-SHA-512

# Schema Registry
SCHEMA_REGISTRY_URL=http://localhost:8081

# Snowflake
SNOWFLAKE_ACCOUNT=xy12345.us-east-1
SNOWFLAKE_USER=kafka_connect
SNOWFLAKE_DATABASE=EVENTS_DB
SNOWFLAKE_SCHEMA=RAW

# FastAPI
JWT_SECRET_KEY=<secret>
JWT_ALGORITHM=HS256
CORS_ORIGINS=["https://app.example.com"]

# Monitoring
PROMETHEUS_PORT=9090
GRAFANA_PORT=3000
```

### Monitoring and Alerting

**Prometheus Alert Rules:**
```yaml
groups:
  - name: event_platform
    rules:
      - alert: HighConsumerLag
        expr: kafka_consumer_lag_seconds > 60
        for: 5m
        annotations:
          summary: "Consumer lag is high"
          
      - alert: HighAPIErrorRate
        expr: rate(http_requests_total{status=~"5.."}[5m]) > 0.01
        for: 2m
        annotations:
          summary: "API error rate exceeds 1%"
          
      - alert: DLQMessagesIncreasing
        expr: increase(kafka_topic_partition_current_offset{topic="events.dlq"}[10m]) > 10
        for: 5m
        annotations:
          summary: "DLQ is accumulating messages"
```

**Grafana Dashboards:**
1. **Kafka Overview**: Broker metrics, topic throughput, partition distribution
2. **Consumer Lag**: Lag by topic and consumer group, lag trends
3. **API Performance**: Request rate, latency percentiles, error rate
4. **Stream Processing**: Processing rate, backpressure, checkpoint duration
5. **System Health**: CPU, memory, network I/O

### Security Considerations

**Network Security:**
- VPC with private subnets for Kafka and internal services
- Security groups restricting access to necessary ports
- NAT Gateway for outbound internet access
- VPC endpoints for AWS services

**Data Security:**
- TLS 1.2+ for all connections
- Encryption at rest for Kafka (AWS KMS)
- Encryption in transit for all data flows
- Secrets stored in AWS Secrets Manager

**Access Control:**
- IAM roles for ECS tasks with least privilege
- Kafka ACLs for topic-level access control
- API authentication with JWT tokens
- Schema Registry authentication

**Compliance:**
- Audit logging for all API requests
- Data retention policies (30 days in Kafka, indefinite in Snowflake)
- PII handling guidelines
- Regular security scanning of Docker images

### Cost Optimization

**Estimated Monthly Costs (AWS):**
- Amazon MSK (3 x kafka.m5.large): $450
- ECS Fargate (2 tasks, 1 vCPU, 2GB): $60
- Application Load Balancer: $25
- Data transfer: $50
- CloudWatch Logs: $20
- **Total**: ~$605/month

**Optimization Strategies:**
- Use Spot instances for non-critical workloads
- Implement data lifecycle policies (archive old data to S3)
- Right-size Kafka broker instances based on actual load
- Use compression to reduce data transfer costs
- Schedule non-production environments to run only during business hours

### Disaster Recovery

**Backup Strategy:**
- Kafka topic data replicated across 3 brokers
- Snowflake automatic backups (7-day retention)
- Configuration stored in Git
- Infrastructure as Code in version control

**Recovery Procedures:**
1. **Kafka Broker Failure**: Automatic failover to replica brokers
2. **Complete Kafka Cluster Failure**: Restore from MSK backup, replay from Snowflake if needed
3. **FastAPI Service Failure**: ECS automatically restarts failed tasks
4. **Snowflake Unavailability**: Messages buffer in Kafka, automatic retry when restored

**RTO/RPO Targets:**
- Recovery Time Objective (RTO): 15 minutes
- Recovery Point Objective (RPO): 5 minutes (data in Kafka)

