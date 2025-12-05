# Project S3: Snowpark ML & Feature Engineering Platform

## Overview

Build a complete ML platform using Snowpark for Python, demonstrating large-scale feature engineering, model training with Snowpark ML, UDF deployment for inference, and hybrid architecture patterns.

**What You'll Build**: An end-to-end ML platform that performs distributed feature engineering in Snowflake, trains models using Snowpark ML, deploys UDFs for batch and real-time inference, and integrates with external ML platforms.

**What You'll Learn**: Snowpark Python API, distributed feature engineering, Snowpark ML library, Python UDFs, vectorized UDFs, and hybrid cloud ML architectures.

## Time Estimate

**3-4 days (24-32 hours)**

### Day 1: Feature Engineering (8 hours)
- Hours 1-2: Snowpark session setup
- Hours 3-5: Feature transformations
- Hours 6-7: Feature Store implementation
- Hour 8: Testing at scale

### Day 2: ML Training (8 hours)
- Hours 1-3: Snowpark ML model training
- Hours 4-5: Hyperparameter tuning
- Hours 6-7: Model evaluation
- Hour 8: Model versioning

### Day 3: Model Deployment (8 hours)
- Hours 1-3: Python UDFs for inference
- Hours 4-5: Vectorized UDFs
- Hours 6-7: Batch scoring
- Hour 8: Testing

### Day 4: Optimization (6-8 hours)
- Hours 1-3: Performance tuning
- Hours 4-5: Monitoring setup
- Hours 6-8: Documentation

## Prerequisites

### Required Knowledge
- [30 Days of Snowflake](https://github.com/washimimizuku/30-days-snowflake-data-ai) - Days 21-30
  - Snowpark basics and ML features
- [100 Days Data & AI](https://github.com/washimimizuku/100-days-data-ai) - Days 51-70
  - ML fundamentals and MLOps
- [60 Days Advanced](https://github.com/washimimizuku/60-days-advanced-data-ai) - Days 25-39
  - Feature stores and advanced ML

### Technical Requirements
- Snowflake account with Snowpark enabled
- Python 3.9+ installed
- Understanding of ML concepts (classification, regression)
- SQL and Python knowledge

### Snowflake Setup
- Enterprise Edition or higher
- Snowpark-optimized warehouse (recommended)
- ACCOUNTADMIN or sufficient privileges
- Stage for UDF deployment

## Getting Started

### Step 1: Review Documentation
1. `prd.md` - Product requirements and goals
2. `tech-spec.md` - Technical architecture and code examples
3. `implementation-plan.md` - Day-by-day implementation guide

### Step 2: Install Snowpark

```bash
# Create virtual environment
python -m venv snowpark_env
source snowpark_env/bin/activate  # Windows: snowpark_env\Scripts\activate

# Install Snowpark and ML libraries
pip install snowflake-snowpark-python[pandas]
pip install snowflake-ml-python
pip install scikit-learn xgboost

# Verify installation
python -c "from snowflake.snowpark import Session; print('✓ Snowpark installed')"
```

### Step 3: Create Snowpark Session

**Create Connection File (`connection.json`):**
```json
{
  "account": "<your_account>",
  "user": "<your_user>",
  "password": "<your_password>",
  "role": "ACCOUNTADMIN",
  "warehouse": "COMPUTE_WH",
  "database": "ML_DATABASE",
  "schema": "PUBLIC"
}
```

**Initialize Session:**
```python
# snowpark_session.py
from snowflake.snowpark import Session
import json

# Load connection parameters
with open('connection.json') as f:
    connection_params = json.load(f)

# Create session
session = Session.builder.configs(connection_params).create()

print(f"✓ Connected to Snowflake")
print(f"  Current database: {session.get_current_database()}")
print(f"  Current schema: {session.get_current_schema()}")
print(f"  Current warehouse: {session.get_current_warehouse()}")

# Test query
result = session.sql("SELECT CURRENT_VERSION()").collect()
print(f"  Snowflake version: {result[0][0]}")
```

### Step 4: Create Snowflake Objects

```sql
-- Create database and schemas
CREATE DATABASE ML_DATABASE;
CREATE SCHEMA ML_DATABASE.FEATURE_STORE;
CREATE SCHEMA ML_DATABASE.MODELS;

-- Create Snowpark-optimized warehouse
CREATE WAREHOUSE SNOWPARK_WH
  WAREHOUSE_SIZE = 'MEDIUM'
  WAREHOUSE_TYPE = 'SNOWPARK-OPTIMIZED'
  AUTO_SUSPEND = 60
  AUTO_RESUME = TRUE;

USE WAREHOUSE SNOWPARK_WH;

-- Create stage for UDFs
CREATE STAGE ML_DATABASE.PUBLIC.UDF_STAGE;

-- Load sample data (using TPC-H)
CREATE TABLE ML_DATABASE.PUBLIC.CUSTOMERS AS
SELECT * FROM SNOWFLAKE_SAMPLE_DATA.TPCH_SF1.CUSTOMER LIMIT 10000;

CREATE TABLE ML_DATABASE.PUBLIC.ORDERS AS
SELECT * FROM SNOWFLAKE_SAMPLE_DATA.TPCH_SF1.ORDERS LIMIT 100000;

SELECT '✓ Snowflake objects created' AS status;
```

## Core Implementation

### 1. Feature Engineering with Snowpark

**Load and Transform Data:**
```python
from snowflake.snowpark import Session
from snowflake.snowpark.functions import col, sum, avg, count, max, min, datediff, current_date
from snowflake.snowpark.types import IntegerType, FloatType

# Create session (using connection from above)
session = Session.builder.configs(connection_params).create()

# Load tables as DataFrames
customers_df = session.table("CUSTOMERS")
orders_df = session.table("ORDERS")

# Feature engineering - customer aggregations
customer_features = (
    orders_df
    .group_by("O_CUSTKEY")
    .agg([
        count("O_ORDERKEY").alias("total_orders"),
        sum("O_TOTALPRICE").alias("total_spent"),
        avg("O_TOTALPRICE").alias("avg_order_value"),
        max("O_ORDERDATE").alias("last_order_date"),
        min("O_ORDERDATE").alias("first_order_date")
    ])
)

# Add derived features
customer_features = customer_features.with_column(
    "days_since_last_order",
    datediff("day", col("last_order_date"), current_date())
)

customer_features = customer_features.with_column(
    "customer_lifetime_days",
    datediff("day", col("first_order_date"), col("last_order_date"))
)

# Join with customer data
final_features = (
    customers_df
    .join(customer_features, customers_df["C_CUSTKEY"] == customer_features["O_CUSTKEY"])
    .select(
        col("C_CUSTKEY").alias("customer_id"),
        col("C_MKTSEGMENT").alias("market_segment"),
        col("C_ACCTBAL").alias("account_balance"),
        col("total_orders"),
        col("total_spent"),
        col("avg_order_value"),
        col("days_since_last_order"),
        col("customer_lifetime_days")
    )
)

# Show sample
final_features.show(5)

# Save to Feature Store
final_features.write.mode("overwrite").save_as_table("FEATURE_STORE.CUSTOMER_FEATURES")

print("✓ Features saved to FEATURE_STORE.CUSTOMER_FEATURES")
```

**Window Functions for Advanced Features:**
```python
from snowflake.snowpark.window import Window

# Create window specification
window_spec = Window.partition_by("O_CUSTKEY").order_by("O_ORDERDATE")

# Calculate order sequence and gaps
orders_with_sequence = (
    orders_df
    .with_column("order_number", row_number().over(window_spec))
    .with_column("prev_order_date", lag("O_ORDERDATE", 1).over(window_spec))
    .with_column(
        "days_between_orders",
        datediff("day", col("prev_order_date"), col("O_ORDERDATE"))
    )
)

# Aggregate window features
order_patterns = (
    orders_with_sequence
    .group_by("O_CUSTKEY")
    .agg([
        avg("days_between_orders").alias("avg_days_between_orders"),
        min("days_between_orders").alias("min_days_between_orders"),
        max("days_between_orders").alias("max_days_between_orders")
    ])
)

order_patterns.show(5)
```

### 2. Snowpark ML Model Training

**Train Classification Model:**
```python
from snowflake.ml.modeling.xgboost import XGBClassifier
from snowflake.ml.modeling.preprocessing import StandardScaler
from snowflake.ml.modeling.model_selection import train_test_split

# Load features
features_df = session.table("FEATURE_STORE.CUSTOMER_FEATURES")

# Create target variable (example: high-value customer)
features_df = features_df.with_column(
    "is_high_value",
    (col("total_spent") > 100000).cast(IntegerType())
)

# Select feature columns
feature_cols = [
    "account_balance",
    "total_orders",
    "total_spent",
    "avg_order_value",
    "days_since_last_order",
    "customer_lifetime_days"
]

# Split data
train_df, test_df = train_test_split(
    features_df,
    test_size=0.2,
    random_state=42
)

# Train model
model = XGBClassifier(
    n_estimators=100,
    max_depth=5,
    learning_rate=0.1,
    input_cols=feature_cols,
    label_cols=["is_high_value"],
    output_cols=["prediction"]
)

model.fit(train_df)

print("✓ Model trained successfully")

# Evaluate model
predictions = model.predict(test_df)
predictions.select("customer_id", "is_high_value", "prediction").show(10)

# Get feature importance
importance = model.feature_importances_
for feature, score in zip(feature_cols, importance):
    print(f"  {feature}: {score:.4f}")
```

**Model Evaluation:**
```python
from snowflake.ml.modeling.metrics import accuracy_score, precision_score, recall_score, f1_score

# Calculate metrics
accuracy = accuracy_score(
    df=predictions,
    y_true_col_names=["is_high_value"],
    y_pred_col_names=["prediction"]
)

precision = precision_score(
    df=predictions,
    y_true_col_names=["is_high_value"],
    y_pred_col_names=["prediction"]
)

recall = recall_score(
    df=predictions,
    y_true_col_names=["is_high_value"],
    y_pred_col_names=["prediction"]
)

f1 = f1_score(
    df=predictions,
    y_true_col_names=["is_high_value"],
    y_pred_col_names=["prediction"]
)

print(f"\nModel Metrics:")
print(f"  Accuracy:  {accuracy:.4f}")
print(f"  Precision: {precision:.4f}")
print(f"  Recall:    {recall:.4f}")
print(f"  F1 Score:  {f1:.4f}")
```

### 3. Deploy Model as UDF

**Create Python UDF:**
```python
from snowflake.snowpark.functions import udf
from snowflake.snowpark.types import FloatType, StructType, StructField

# Define UDF for prediction
@udf(
    name="predict_high_value_customer",
    is_permanent=True,
    stage_location="@UDF_STAGE",
    packages=["xgboost", "scikit-learn"],
    replace=True
)
def predict_high_value(
    account_balance: float,
    total_orders: int,
    total_spent: float,
    avg_order_value: float,
    days_since_last_order: int,
    customer_lifetime_days: int
) -> float:
    # Load model (in production, load from stage)
    import xgboost as xgb
    import numpy as np
    
    # Simple scoring logic (replace with actual model)
    features = np.array([[
        account_balance,
        total_orders,
        total_spent,
        avg_order_value,
        days_since_last_order,
        customer_lifetime_days
    ]])
    
    # Score based on rules (replace with model.predict)
    score = (total_spent / 100000) * 0.5 + (total_orders / 100) * 0.3 + (account_balance / 10000) * 0.2
    return min(score, 1.0)

print("✓ UDF created: predict_high_value_customer")
```

**Use UDF for Scoring:**
```sql
-- Score all customers using UDF
SELECT
    customer_id,
    market_segment,
    predict_high_value_customer(
        account_balance,
        total_orders,
        total_spent,
        avg_order_value,
        days_since_last_order,
        customer_lifetime_days
    ) AS high_value_score
FROM FEATURE_STORE.CUSTOMER_FEATURES
ORDER BY high_value_score DESC
LIMIT 10;
```

### 4. Vectorized UDF for Performance

**Create Vectorized UDF:**
```python
from snowflake.snowpark.functions import pandas_udf
from snowflake.snowpark.types import PandasSeriesType, PandasDataFrameType

# Vectorized UDF processes batches
@pandas_udf(
    name="batch_predict_high_value",
    is_permanent=True,
    stage_location="@UDF_STAGE",
    packages=["pandas", "numpy"],
    replace=True
)
def batch_predict(
    df: PandasDataFrameType[
        float, int, float, float, int, int
    ]
) -> PandasSeriesType[float]:
    import pandas as pd
    import numpy as np
    
    # Vectorized scoring
    scores = (
        (df.iloc[:, 2] / 100000) * 0.5 +  # total_spent
        (df.iloc[:, 1] / 100) * 0.3 +      # total_orders
        (df.iloc[:, 0] / 10000) * 0.2      # account_balance
    )
    
    return scores.clip(0, 1)

print("✓ Vectorized UDF created: batch_predict_high_value")
```

### 5. Stored Procedure for Training Workflow

**Create Training Stored Procedure:**
```python
from snowflake.snowpark.functions import sproc

@sproc(
    name="train_customer_model",
    is_permanent=True,
    stage_location="@UDF_STAGE",
    packages=["snowflake-ml-python", "xgboost"],
    replace=True
)
def train_customer_model(session: Session, feature_table: str) -> str:
    from snowflake.ml.modeling.xgboost import XGBClassifier
    
    # Load features
    features_df = session.table(feature_table)
    
    # Create target
    features_df = features_df.with_column(
        "is_high_value",
        (col("total_spent") > 100000).cast(IntegerType())
    )
    
    # Train model
    model = XGBClassifier(
        n_estimators=100,
        max_depth=5,
        input_cols=["account_balance", "total_orders", "total_spent"],
        label_cols=["is_high_value"]
    )
    
    model.fit(features_df)
    
    # Save model
    model_name = f"customer_model_{session.sql('SELECT CURRENT_TIMESTAMP()').collect()[0][0]}"
    
    return f"Model trained and saved: {model_name}"

print("✓ Stored procedure created: train_customer_model")
```

**Call Stored Procedure:**
```sql
-- Execute training workflow
CALL train_customer_model('FEATURE_STORE.CUSTOMER_FEATURES');
```

### 6. Feature Store with Versioning

**Create Versioned Feature Table:**
```sql
-- Feature table with versioning
CREATE TABLE FEATURE_STORE.CUSTOMER_FEATURES_VERSIONED (
    customer_id INTEGER,
    feature_version INTEGER,
    market_segment VARCHAR(50),
    account_balance FLOAT,
    total_orders INTEGER,
    total_spent FLOAT,
    avg_order_value FLOAT,
    days_since_last_order INTEGER,
    customer_lifetime_days INTEGER,
    created_at TIMESTAMP_LTZ DEFAULT CURRENT_TIMESTAMP(),
    PRIMARY KEY (customer_id, feature_version)
);

-- Insert features with version
INSERT INTO FEATURE_STORE.CUSTOMER_FEATURES_VERSIONED
SELECT
    customer_id,
    1 AS feature_version,
    market_segment,
    account_balance,
    total_orders,
    total_spent,
    avg_order_value,
    days_since_last_order,
    customer_lifetime_days,
    CURRENT_TIMESTAMP()
FROM FEATURE_STORE.CUSTOMER_FEATURES;
```

### 7. Monitoring and Performance

**Query Performance Metrics:**
```sql
-- Monitor Snowpark query performance
SELECT
    query_id,
    query_text,
    warehouse_name,
    execution_time / 1000 AS execution_seconds,
    bytes_scanned / 1024 / 1024 / 1024 AS gb_scanned,
    rows_produced
FROM SNOWFLAKE.ACCOUNT_USAGE.QUERY_HISTORY
WHERE query_text ILIKE '%FEATURE_STORE%'
  AND start_time >= DATEADD('hour', -1, CURRENT_TIMESTAMP())
ORDER BY start_time DESC
LIMIT 10;
```

**UDF Performance:**
```sql
-- Monitor UDF execution
SELECT
    function_name,
    COUNT(*) AS execution_count,
    AVG(execution_time_ms) AS avg_execution_ms,
    MAX(execution_time_ms) AS max_execution_ms
FROM SNOWFLAKE.ACCOUNT_USAGE.FUNCTIONS
WHERE function_name LIKE 'PREDICT%'
  AND execution_time >= DATEADD('day', -1, CURRENT_TIMESTAMP())
GROUP BY function_name;
```

## Success Criteria

- [ ] Snowpark session connected and working
- [ ] Feature engineering pipeline processing 10K+ rows
- [ ] Feature Store with versioned features
- [ ] Snowpark ML model trained successfully
- [ ] Model evaluation metrics calculated
- [ ] Python UDF deployed for inference
- [ ] Vectorized UDF for batch scoring
- [ ] Stored procedure for training workflow
- [ ] Performance monitoring queries working
- [ ] Documentation complete with code examples

## Cost Optimization

### Warehouse Sizing
- Use **Snowpark-optimized warehouses** for ML workloads
- Start with **MEDIUM** size, scale as needed
- Enable **AUTO_SUSPEND** (60 seconds)
- Use **AUTO_RESUME** for on-demand access

### Query Optimization
- Use **caching** for iterative development:
  ```python
  features_df.cache_result()
  ```
- **Sample data** for experimentation:
  ```python
  sample_df = features_df.sample(n=10000)
  ```
- **Filter early** to reduce data scanned

### UDF Costs
- Vectorized UDFs are more cost-effective than row-by-row
- Batch predictions instead of real-time when possible
- Monitor UDF execution time and optimize

**Estimated Monthly Cost** (for this project):
- Snowpark Warehouse (Medium, 8 hours/day): ~$200-300
- Storage (100 GB): ~$2-5
- UDF execution: ~$20-50
- **Total**: ~$220-355/month

## Common Challenges

### Session Connection Fails
Verify account identifier format, check credentials, ensure warehouse exists

### UDF Deployment Errors
Check stage permissions, verify package names, ensure Python version compatibility

### Slow Feature Engineering
Use Snowpark-optimized warehouse, add filters early, consider sampling

### Model Training Fails
Check data types, ensure no nulls in features, verify sufficient warehouse size

## Learning Outcomes

- Build distributed feature engineering with Snowpark
- Train ML models using Snowpark ML library
- Deploy models as Python UDFs
- Create vectorized UDFs for performance
- Implement Feature Store patterns
- Monitor Snowpark query performance
- Optimize warehouse usage and costs
- Navigate Snowpark API and documentation

## Next Steps

1. Add to portfolio with Snowpark architecture diagram
2. Write blog post: "ML in Snowflake: Snowpark vs External Compute"
3. Prepare for SnowPro Advanced Data Engineer certification
4. Extend with Snowpark Container Services for deep learning

## Resources

- [Snowpark Python Docs](https://docs.snowflake.com/en/developer-guide/snowpark/python/index)
- [Snowpark ML Docs](https://docs.snowflake.com/en/developer-guide/snowpark-ml/index)
- [Python UDFs](https://docs.snowflake.com/en/developer-guide/udf/python/udf-python)
- [Snowpark Optimized Warehouses](https://docs.snowflake.com/en/user-guide/warehouses-snowpark-optimized)
- [Feature Engineering Guide](https://docs.snowflake.com/en/user-guide/ml-powered-features)
