# Project D3: MLOps with MLflow & Feature Store

## Overview

Build a complete MLOps platform on Databricks demonstrating Feature Store, MLflow experiment tracking, Model Registry, model serving, A/B testing, and drift detection.

**What You'll Build**: An end-to-end ML platform with centralized feature engineering, automated experiment tracking, model deployment with serving endpoints, and production monitoring.

**What You'll Learn**: Databricks Feature Store, MLflow native integration, Model Registry workflows, real-time model serving, A/B testing, and drift detection.

## Time Estimate

**3-4 days (24-32 hours)**

### Day 1: Feature Store (8 hours)
- Hours 1-2: Feature Store setup
- Hours 3-5: Feature engineering pipelines
- Hours 6-7: Feature table creation
- Hour 8: Feature serving testing

### Day 2: ML Training (8 hours)
- Hours 1-3: MLflow experiment tracking
- Hours 4-5: Model training with Feature Store
- Hours 6-7: Hyperparameter tuning
- Hour 8: Model Registry

### Day 3: Model Serving (8 hours)
- Hours 1-3: Serving endpoint deployment
- Hours 4-5: Batch inference
- Hours 6-7: A/B testing setup
- Hour 8: Testing and validation

### Day 4: Monitoring (6-8 hours)
- Hours 1-3: Drift detection setup
- Hours 4-5: Automated retraining
- Hours 6-8: Documentation

## Prerequisites

### Required Knowledge
- [30 Days of Databricks](https://github.com/washimimizuku/30-days-databricks-data-ai) - Days 21-30
  - ML on Databricks and MLOps patterns
- [100 Days Data & AI](https://github.com/washimimizuku/100-days-data-ai) - Days 51-70
  - ML fundamentals and MLOps basics
- [60 Days Advanced](https://github.com/washimimizuku/60-days-advanced-data-ai) - Days 25-39
  - Feature stores and advanced MLOps

### Technical Requirements
- Databricks workspace with ML runtime
- Understanding of ML concepts (classification, regression)
- Python and SQL knowledge
- scikit-learn or XGBoost familiarity

### Databricks Setup
- ML runtime cluster (DBR 13.3 ML or higher)
- Feature Store enabled
- Model Serving enabled
- Unity Catalog (optional, for governance)

## Getting Started

### Step 1: Review Documentation
1. `prd.md` - Product requirements and goals
2. `tech-spec.md` - Technical architecture and code examples
3. `implementation-plan.md` - Day-by-day implementation guide

### Step 2: Create ML Cluster

**Navigate to Compute:**
1. Click **Compute** in left sidebar
2. Click **Create Compute**
3. Configure cluster:
   - **Cluster Name**: `ml-cluster`
   - **Cluster Mode**: Single Node (for development)
   - **Databricks Runtime**: 13.3 LTS ML
   - **Node Type**: i3.xlarge or similar
   - **Terminate after**: 30 minutes of inactivity
4. Click **Create Compute**

### Step 3: Prepare Sample Data

```python
# Generate sample customer churn data
from pyspark.sql import SparkSession
from pyspark.sql.functions import *
import random

spark = SparkSession.builder.getOrCreate()

# Generate orders data
orders_data = []
for i in range(10000):
    orders_data.append({
        "order_id": i,
        "customer_id": random.randint(1, 1000),
        "amount": round(random.uniform(10, 500), 2),
        "order_date": (datetime.now() - timedelta(days=random.randint(0, 365))).isoformat()
    })

orders_df = spark.createDataFrame(orders_data)
orders_df.write.mode("overwrite").saveAsTable("ml.orders")

# Generate churn labels
labels_data = []
for cid in range(1, 1001):
    labels_data.append({
        "customer_id": cid,
        "churn": random.choice([0, 1])
    })

labels_df = spark.createDataFrame(labels_data)
labels_df.write.mode("overwrite").saveAsTable("ml.customer_labels")

print("✓ Sample data created: ml.orders and ml.customer_labels")
```

## Core Implementation

### 1. Feature Store Setup

**Create Feature Table (UI):**
1. Go to **Machine Learning** → **Feature Store**
2. Click **Create Feature Table**
3. Enter details:
   - **Name**: `ml.customer_features`
   - **Primary Keys**: `customer_id`
   - **Description**: Customer behavioral features
4. Click **Create**

**Create Feature Table (Code):**
```python
from databricks.feature_store import FeatureStoreClient
from pyspark.sql.functions import *

fs = FeatureStoreClient()

# Compute customer features
def compute_customer_features(orders_df):
    features_df = (
        orders_df
        .groupBy("customer_id")
        .agg(
            count("order_id").alias("total_orders"),
            sum("amount").alias("total_spent"),
            avg("amount").alias("avg_order_value"),
            datediff(current_date(), max("order_date")).alias("days_since_last_order"),
            countDistinct(to_date("order_date")).alias("active_days"),
            stddev("amount").alias("order_amount_std")
        )
        .withColumn("avg_days_between_orders", 
            col("days_since_last_order") / col("total_orders"))
    )
    return features_df

# Load orders
orders_df = spark.table("ml.orders")
features_df = compute_customer_features(orders_df)

# Create feature table
fs.create_table(
    name="ml.customer_features",
    primary_keys=["customer_id"],
    df=features_df,
    description="Customer behavioral features for churn prediction",
    tags={"team": "ml", "project": "churn"}
)

print("✓ Feature table created: ml.customer_features")
```

**Update Features (Scheduled):**
```python
# Update feature table with new data
orders_df = spark.table("ml.orders")
features_df = compute_customer_features(orders_df)

fs.write_table(
    name="ml.customer_features",
    df=features_df,
    mode="merge"  # Merge updates existing records
)

print("✓ Features updated")
```

### 2. MLflow Experiment Tracking

**Create Experiment (UI):**
1. Go to **Machine Learning** → **Experiments**
2. Click **Create Experiment**
3. Enter name: `/ml/churn_prediction`
4. Click **Create**

**Train Model with MLflow:**
```python
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from databricks.feature_store import FeatureLookup

# Set experiment
mlflow.set_experiment("/ml/churn_prediction")

# Define feature lookups
feature_lookups = [
    FeatureLookup(
        table_name="ml.customer_features",
        lookup_key="customer_id"
    )
]

# Create training set from Feature Store
labels_df = spark.table("ml.customer_labels")

training_set = fs.create_training_set(
    df=labels_df,
    feature_lookups=feature_lookups,
    label="churn",
    exclude_columns=["customer_id"]
)

# Load as pandas DataFrame
training_df = training_set.load_df().toPandas()

# Split data
X = training_df.drop("churn", axis=1)
y = training_df["churn"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train with MLflow tracking
with mlflow.start_run(run_name="rf_baseline") as run:
    # Log parameters
    n_estimators = 100
    max_depth = 10
    mlflow.log_param("n_estimators", n_estimators)
    mlflow.log_param("max_depth", max_depth)
    mlflow.log_param("model_type", "RandomForest")
    
    # Train model
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=42
    )
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Log metrics
    mlflow.log_metric("accuracy", accuracy_score(y_test, y_pred))
    mlflow.log_metric("precision", precision_score(y_test, y_pred))
    mlflow.log_metric("recall", recall_score(y_test, y_pred))
    mlflow.log_metric("f1_score", f1_score(y_test, y_pred))
    
    # Log model with Feature Store
    fs.log_model(
        model=model,
        artifact_path="model",
        flavor=mlflow.sklearn,
        training_set=training_set,
        registered_model_name="churn_prediction_model"
    )
    
    print(f"✓ Model logged to MLflow. Run ID: {run.info.run_id}")
```

**View Experiment (UI):**
1. Go to **Machine Learning** → **Experiments**
2. Click on `/ml/churn_prediction`
3. See all runs with metrics and parameters
4. Compare runs side-by-side
5. View artifacts and models

### 3. Hyperparameter Tuning

```python
from hyperopt import fmin, tpe, hp, SparkTrials, STATUS_OK
import mlflow

# Define search space
search_space = {
    'n_estimators': hp.choice('n_estimators', [50, 100, 200]),
    'max_depth': hp.choice('max_depth', [5, 10, 15, 20]),
    'min_samples_split': hp.choice('min_samples_split', [2, 5, 10])
}

# Objective function
def train_model(params):
    with mlflow.start_run(nested=True):
        mlflow.log_params(params)
        
        model = RandomForestClassifier(**params, random_state=42)
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        mlflow.log_metric("accuracy", accuracy)
        
        return {'loss': -accuracy, 'status': STATUS_OK}

# Run hyperparameter tuning
with mlflow.start_run(run_name="hyperopt_tuning"):
    spark_trials = SparkTrials(parallelism=4)
    
    best_params = fmin(
        fn=train_model,
        space=search_space,
        algo=tpe.suggest,
        max_evals=20,
        trials=spark_trials
    )
    
    mlflow.log_params(best_params)
    print(f"✓ Best parameters: {best_params}")
```

### 4. Model Registry

**Register Model (UI):**
1. Go to **Machine Learning** → **Experiments**
2. Select your experiment and run
3. Click **Register Model**
4. Enter model name: `churn_prediction_model`
5. Click **Register**

**Manage Model Versions (Code):**
```python
from mlflow.tracking import MlflowClient

client = MlflowClient()

# Get latest model version
model_name = "churn_prediction_model"
latest_version = client.get_latest_versions(model_name, stages=["None"])[0]

# Add description
client.update_model_version(
    name=model_name,
    version=latest_version.version,
    description="Random Forest model trained on customer behavioral features"
)

# Add tags
client.set_model_version_tag(
    name=model_name,
    version=latest_version.version,
    key="validation_status",
    value="approved"
)

# Transition to Staging
client.transition_model_version_stage(
    name=model_name,
    version=latest_version.version,
    stage="Staging",
    archive_existing_versions=False
)

print(f"✓ Model version {latest_version.version} moved to Staging")

# After validation, move to Production
client.transition_model_version_stage(
    name=model_name,
    version=latest_version.version,
    stage="Production",
    archive_existing_versions=True  # Archive old production versions
)

print(f"✓ Model version {latest_version.version} moved to Production")
```

**View Model Registry (UI):**
1. Go to **Machine Learning** → **Models**
2. Click on `churn_prediction_model`
3. See all versions and stages
4. View lineage and serving endpoints

### 5. Model Serving

**Create Serving Endpoint (UI):**
1. Go to **Machine Learning** → **Serving**
2. Click **Create Serving Endpoint**
3. Configure:
   - **Name**: `churn-prediction-endpoint`
   - **Model**: `churn_prediction_model`
   - **Version**: Select version or use "Production" stage
   - **Compute Size**: Small
   - **Scale to Zero**: Enabled
4. Click **Create**

**Create Serving Endpoint (Code):**
```python
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import EndpointCoreConfigInput, ServedModelInput

w = WorkspaceClient()

# Create endpoint
w.serving_endpoints.create(
    name="churn-prediction-endpoint",
    config=EndpointCoreConfigInput(
        served_models=[
            ServedModelInput(
                model_name="churn_prediction_model",
                model_version="1",
                workload_size="Small",
                scale_to_zero_enabled=True
            )
        ]
    )
)

print("✓ Serving endpoint created")
```

**Invoke Endpoint:**
```python
import requests
import json
import os

# Get workspace URL and token
workspace_url = spark.conf.get("spark.databricks.workspaceUrl")
token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()

# Invoke endpoint
url = f"https://{workspace_url}/serving-endpoints/churn-prediction-endpoint/invocations"
headers = {
    "Authorization": f"Bearer {token}",
    "Content-Type": "application/json"
}

data = {
    "dataframe_records": [
        {"customer_id": 1},
        {"customer_id": 2},
        {"customer_id": 3}
    ]
}

response = requests.post(url, headers=headers, json=data)
predictions = response.json()

print(f"✓ Predictions: {predictions}")
```

**Batch Inference:**
```python
# Score batch of customers
customers_df = spark.table("ml.customer_labels").select("customer_id")

predictions_df = fs.score_batch(
    model_uri=f"models:/churn_prediction_model/Production",
    df=customers_df
)

# Save predictions
predictions_df.write.mode("overwrite").saveAsTable("ml.churn_predictions")

print("✓ Batch predictions saved to ml.churn_predictions")
```

### 6. A/B Testing

**Deploy Two Versions:**
```python
# Update endpoint with traffic splitting
w.serving_endpoints.update_config(
    name="churn-prediction-endpoint",
    served_models=[
        ServedModelInput(
            model_name="churn_prediction_model",
            model_version="1",
            workload_size="Small",
            scale_to_zero_enabled=True,
            traffic_percentage=80  # 80% to version 1
        ),
        ServedModelInput(
            model_name="churn_prediction_model",
            model_version="2",
            workload_size="Small",
            scale_to_zero_enabled=True,
            traffic_percentage=20  # 20% to version 2
        )
    ]
)

print("✓ A/B test configured: 80% v1, 20% v2")
```

**Monitor A/B Test:**
```sql
-- Query serving logs
SELECT
    model_version,
    COUNT(*) as request_count,
    AVG(latency_ms) as avg_latency,
    AVG(prediction_score) as avg_prediction_score
FROM system.serving.inference_log
WHERE endpoint_name = 'churn-prediction-endpoint'
  AND timestamp >= current_timestamp() - INTERVAL 7 DAYS
GROUP BY model_version;
```

### 7. Drift Detection

**Enable Lakehouse Monitoring:**
```python
from databricks.lakehouse_monitoring import create_monitor

# Create monitor for predictions table
create_monitor(
    table_name="ml.churn_predictions",
    profile_type="InferenceLog",
    model_id_col="model_version",
    prediction_col="prediction",
    timestamp_col="prediction_timestamp",
    granularities=["1 day"],
    slicing_exprs=["customer_segment"]
)

print("✓ Drift monitor created")
```

**Query Drift Metrics:**
```sql
-- Check feature drift
SELECT
    window.start as date,
    feature_name,
    drift_score,
    CASE 
        WHEN drift_score > 0.1 THEN 'HIGH'
        WHEN drift_score > 0.05 THEN 'MEDIUM'
        ELSE 'LOW'
    END as drift_level
FROM ml.drift_metrics
WHERE date >= current_date() - 30
ORDER BY drift_score DESC;
```

**Automated Retraining Trigger:**
```python
# Check for high drift
drift_df = spark.sql("""
    SELECT COUNT(*) as high_drift_count
    FROM ml.drift_metrics
    WHERE drift_score > 0.1
      AND date >= current_date() - 7
""")

high_drift_count = drift_df.collect()[0]["high_drift_count"]

if high_drift_count > 0:
    print(f"⚠️ High drift detected ({high_drift_count} features)")
    
    # Trigger retraining job
    from databricks.sdk import WorkspaceClient
    w = WorkspaceClient()
    
    run = w.jobs.run_now(job_id=retraining_job_id)
    print(f"✓ Retraining job triggered: {run.run_id}")
else:
    print("✓ No significant drift detected")
```

### 8. Automated ML Pipeline

**Create Job (UI):**
1. Go to **Workflows** → **Jobs**
2. Click **Create Job**
3. Add tasks:
   - **Task 1**: Feature Engineering
   - **Task 2**: Model Training (depends on Task 1)
   - **Task 3**: Model Validation (depends on Task 2)
   - **Task 4**: Model Deployment (depends on Task 3)
4. Set schedule: Daily at 2 AM
5. Click **Create**

## Success Criteria

- [ ] Feature Store with customer behavioral features
- [ ] MLflow experiments with tracked runs
- [ ] Hyperparameter tuning completed
- [ ] Model registered in Model Registry
- [ ] Model transitioned through Staging → Production
- [ ] Serving endpoint deployed and tested
- [ ] A/B testing configured with traffic splitting
- [ ] Drift detection monitoring enabled
- [ ] Automated retraining pipeline created
- [ ] Batch inference working
- [ ] Documentation complete

## Cost Optimization

### Compute Costs
- Use **Single Node** clusters for development
- Enable **Autoscaling** for production (1-5 workers)
- Set **Auto-termination** to 30 minutes
- Use **Spot instances** for training (50-70% savings)

### Serving Costs
- Enable **Scale to Zero** for low-traffic endpoints
- Use **Small** workload size initially
- Monitor **Request Rate** and adjust size

### Storage Costs
- Feature Store uses Delta tables (standard storage costs)
- Enable **Auto Optimize** on feature tables
- Set **VACUUM** retention to 7 days

**Estimated Monthly Cost** (for this project):
- ML Cluster (Single Node, 8 hours/day): ~$150-200
- Model Serving (Small, scale-to-zero): ~$50-100
- Storage (100 GB): ~$2-5
- **Total**: ~$200-305/month

## Common Challenges

### Feature Store Not Found
Ensure Feature Store is enabled in workspace settings

### Model Registration Fails
Check that model was logged with `fs.log_model()` not `mlflow.log_model()`

### Serving Endpoint Errors
Verify model version exists and is in correct stage

### Drift Monitoring Not Working
Ensure predictions table has required columns (timestamp, prediction, model_version)

## Learning Outcomes

- Build centralized Feature Store on Databricks
- Track ML experiments with MLflow
- Manage model lifecycle with Model Registry
- Deploy models with real-time serving endpoints
- Implement A/B testing for safe deployments
- Monitor model performance and detect drift
- Automate ML pipelines with Databricks Jobs
- Navigate Databricks ML UI for MLOps tasks

## Next Steps

1. Add to portfolio with MLOps architecture diagram
2. Write blog post: "Databricks MLOps vs SageMaker"
3. Prepare for Databricks ML Associate certification
4. Extend with deep learning models (PyTorch/TensorFlow)

## Resources

- [Feature Store Docs](https://docs.databricks.com/machine-learning/feature-store/index.html)
- [MLflow on Databricks](https://docs.databricks.com/mlflow/index.html)
- [Model Serving](https://docs.databricks.com/machine-learning/model-serving/index.html)
- [Lakehouse Monitoring](https://docs.databricks.com/lakehouse-monitoring/index.html)
- [MLOps Guide](https://www.databricks.com/product/machine-learning/mlops)
