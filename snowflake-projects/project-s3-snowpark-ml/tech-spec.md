# Technical Specification: Snowpark ML & Feature Engineering Platform

## Architecture Overview

```
Snowflake Data → Snowpark Feature Engineering → Snowpark ML Training → Model Registry
                                                         ↓
                                              External ML (SageMaker/Databricks)
                                                         ↓
                                                  Model Deployment
```

## Technology Stack

- **Platform**: Snowflake Enterprise with Snowpark
- **Language**: Python 3.11+, Scala 2.12+ (optional)
- **ML**: Snowpark ML, scikit-learn, XGBoost
- **External**: AWS SageMaker (optional integration)
- **Notebooks**: Snowflake Notebooks or Jupyter

## Detailed Design

### 1. Feature Engineering

```python
from snowflake.snowpark import Session
from snowflake.snowpark.functions import col, sum, avg, count, lag
from snowflake.snowpark.window import Window

# Create session
session = Session.builder.configs(connection_params).create()

# Feature engineering
customers_df = session.table("customers")
orders_df = session.table("orders")

# Aggregation features
customer_features = orders_df.group_by("customer_id").agg([
    sum("amount").alias("total_spent"),
    count("order_id").alias("order_count"),
    avg("amount").alias("avg_order_value")
])

# Window functions
window_spec = Window.partition_by("customer_id").order_by("order_date")
recency_features = orders_df.with_column(
    "days_since_last_order",
    datediff("day", lag("order_date").over(window_spec), col("order_date"))
)

# Join features
final_features = customers_df.join(
    customer_features,
    on="customer_id"
).join(
    recency_features,
    on="customer_id"
)

# Save to feature store
final_features.write.mode("overwrite").save_as_table("feature_store.customer_features")
```

### 2. Snowpark ML Training

```python
from snowflake.ml.modeling.xgboost import XGBClassifier
from snowflake.ml.modeling.preprocessing import StandardScaler
from snowflake.ml.modeling.pipeline import Pipeline

# Load features
features_df = session.table("feature_store.customer_features")

# Create pipeline
pipeline = Pipeline(steps=[
    ("scaler", StandardScaler()),
    ("classifier", XGBClassifier(n_estimators=100, max_depth=5))
])

# Train model
pipeline.fit(features_df, target_col="churn")

# Save model
pipeline.save("models/churn_model_v1")
```

### 3. UDFs and Stored Procedures

```python
# Vectorized UDF for feature transformation
@udf(name="calculate_rfm_score", is_permanent=True, stage_location="@udf_stage")
def calculate_rfm_score(recency: int, frequency: int, monetary: float) -> int:
    r_score = 5 if recency < 30 else (4 if recency < 60 else 3)
    f_score = 5 if frequency > 10 else (4 if frequency > 5 else 3)
    m_score = 5 if monetary > 1000 else (4 if monetary > 500 else 3)
    return r_score + f_score + m_score

# Stored procedure for training workflow
@sproc(name="train_churn_model", is_permanent=True, stage_location="@sproc_stage")
def train_churn_model(session: Session, table_name: str) -> str:
    # Feature engineering
    features = session.table(table_name)
    
    # Train model
    model = XGBClassifier()
    model.fit(features, target_col="churn")
    
    # Save and return metrics
    metrics = model.evaluate(features)
    return f"Model trained. Accuracy: {metrics['accuracy']}"
```

### 4. Feature Store Implementation

```sql
-- Feature store schema
CREATE SCHEMA feature_store;

-- Feature table with versioning
CREATE TABLE feature_store.customer_features (
    customer_id INTEGER,
    feature_version INTEGER,
    total_spent DECIMAL(10,2),
    order_count INTEGER,
    avg_order_value DECIMAL(10,2),
    days_since_last_order INTEGER,
    rfm_score INTEGER,
    created_at TIMESTAMP_LTZ DEFAULT CURRENT_TIMESTAMP(),
    PRIMARY KEY (customer_id, feature_version)
);

-- Feature lineage tracking
CREATE TABLE feature_store.feature_lineage (
    feature_name VARCHAR(255),
    source_tables ARRAY,
    transformation_logic VARCHAR,
    created_by VARCHAR(255),
    created_at TIMESTAMP_LTZ DEFAULT CURRENT_TIMESTAMP()
);
```

### 5. Hybrid Architecture - SageMaker Integration

```python
import boto3
import sagemaker
from sagemaker.xgboost import XGBoost

# Export data from Snowflake
features_df.write.mode("overwrite").save_as_table("temp.training_data")

# Export to S3
session.sql("""
    COPY INTO @s3_stage/training_data/
    FROM temp.training_data
    FILE_FORMAT = (TYPE = PARQUET)
""").collect()

# Train in SageMaker
xgb = XGBoost(
    entry_point="train.py",
    role=sagemaker_role,
    instance_count=1,
    instance_type="ml.m5.xlarge"
)
xgb.fit({"train": "s3://bucket/training_data/"})

# Deploy model
predictor = xgb.deploy(instance_type="ml.m5.large", initial_instance_count=1)

# Score in Snowflake using external function
CREATE EXTERNAL FUNCTION predict_churn(features VARIANT)
RETURNS VARIANT
API_INTEGRATION = aws_api_integration
AS 'https://sagemaker-endpoint.amazonaws.com/predict';
```

### 6. Performance Optimization

```python
# Use Snowpark optimized warehouses
session.sql("USE WAREHOUSE snowpark_optimized_wh").collect()

# Caching for iterative development
features_df.cache_result()

# Efficient sampling
sample_df = features_df.sample(n=100000)

# Parallel processing
from snowflake.snowpark.functions import pandas_udf
from snowflake.snowpark.types import PandasSeriesType

@pandas_udf(name="batch_predict", is_permanent=True)
def batch_predict(features: PandasSeriesType[dict]) -> PandasSeriesType[float]:
    # Vectorized prediction
    import pandas as pd
    return pd.Series([model.predict(f) for f in features])
```

## Implementation Phases

### Phase 1: Feature Engineering (Day 1)
- Set up Snowpark environment
- Implement feature transformations
- Create feature store
- Test at scale (100M+ rows)

### Phase 2: ML Training (Day 2)
- Implement Snowpark ML training
- Create UDFs and stored procedures
- Model versioning
- Hyperparameter tuning

### Phase 3: Hybrid Architecture (Day 3)
- SageMaker integration
- External functions
- Performance comparison
- Cost analysis

### Phase 4: Documentation (Day 4)
- Architecture diagrams
- Performance benchmarks
- "ML on Snowflake" guide
- Cost optimization recommendations

## Deliverables

### Code
- Snowpark feature engineering scripts
- ML training pipelines
- UDFs and stored procedures
- Hybrid architecture examples

### Documentation
- Architecture diagram
- "ML on Snowflake: Architecture Patterns"
- Performance comparison (Snowpark vs. Spark)
- When to use Snowpark vs. external compute
- Feature store implementation guide

### Metrics
- Feature engineering performance (rows/second)
- Training time comparison
- Inference latency
- Cost per model training run
