# Technical Specification: MLOps with MLflow & Feature Store

## Architecture Overview

```
Data → Feature Store → ML Training → MLflow Registry → Model Serving
                           ↓                              ↓
                    Hyperparameter Tuning          Monitoring & Drift
```

## Technology Stack

- **Platform**: Databricks with MLflow
- **ML**: Spark ML, scikit-learn, XGBoost, PyTorch
- **Feature Store**: Databricks Feature Store
- **Serving**: Databricks Model Serving
- **Monitoring**: Databricks Lakehouse Monitoring
- **Orchestration**: Databricks Jobs

## Detailed Design

### 1. Feature Store Implementation

```python
from databricks.feature_store import FeatureStoreClient
from pyspark.sql.functions import *

fs = FeatureStoreClient()

# Create feature table
def compute_customer_features(data_df):
    features_df = (
        data_df
        .groupBy("customer_id")
        .agg(
            count("order_id").alias("total_orders"),
            sum("amount").alias("total_spent"),
            avg("amount").alias("avg_order_value"),
            datediff(current_date(), max("order_date")).alias("days_since_last_order"),
            countDistinct(to_date("order_date")).alias("active_days")
        )
    )
    return features_df

# Create feature table
fs.create_table(
    name="ml.customer_features",
    primary_keys=["customer_id"],
    df=compute_customer_features(orders_df),
    description="Customer behavioral features for churn prediction"
)

# Update features (scheduled job)
fs.write_table(
    name="ml.customer_features",
    df=compute_customer_features(orders_df),
    mode="merge"
)
```

### 2. ML Training with MLflow

```python
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from databricks.feature_store import FeatureLookup

# Set experiment
mlflow.set_experiment("/ml/churn_prediction")

# Feature lookups
feature_lookups = [
    FeatureLookup(
        table_name="ml.customer_features",
        lookup_key="customer_id"
    )
]

# Create training set
training_set = fs.create_training_set(
    df=labels_df,
    feature_lookups=feature_lookups,
    label="churn",
    exclude_columns=["customer_id"]
)

training_df = training_set.load_df()

# Train with MLflow tracking
with mlflow.start_run(run_name="rf_baseline") as run:
    # Log parameters
    mlflow.log_param("n_estimators", 100)
    mlflow.log_param("max_depth", 10)
    
    # Train model
    X_train, X_test, y_train, y_test = train_test_split(
        training_df.drop("churn"),
        training_df["churn"],
        test_size=0.2
    )
    
    model = RandomForestClassifier(n_estimators=100, max_depth=10)
    model.fit(X_train, y_train)
    
    # Log metrics
    accuracy = model.score(X_test, y_test)
    mlflow.log_metric("accuracy", accuracy)
    
    # Log model with feature store
    fs.log_model(
        model=model,
        artifact_path="model",
        flavor=mlflow.sklearn,
        training_set=training_set,
        registered_model_name="churn_prediction_model"
    )
```

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
        
        model = RandomForestClassifier(**params)
        model.fit(X_train, y_train)
        
        accuracy = model.score(X_test, y_test)
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
```

### 4. Model Registry & Governance

```python
from mlflow.tracking import MlflowClient

client = MlflowClient()

# Register model
model_uri = f"runs:/{run.info.run_id}/model"
model_details = mlflow.register_model(
    model_uri=model_uri,
    name="churn_prediction_model"
)

# Transition to staging
client.transition_model_version_stage(
    name="churn_prediction_model",
    version=model_details.version,
    stage="Staging"
)

# Add model description and tags
client.update_model_version(
    name="churn_prediction_model",
    version=model_details.version,
    description="Random Forest model for customer churn prediction"
)

client.set_model_version_tag(
    name="churn_prediction_model",
    version=model_details.version,
    key="validation_status",
    value="approved"
)

# Transition to production
client.transition_model_version_stage(
    name="churn_prediction_model",
    version=model_details.version,
    stage="Production"
)
```

### 5. Model Serving

```python
# Deploy model to serving endpoint
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import EndpointCoreConfigInput, ServedModelInput

w = WorkspaceClient()

# Create serving endpoint
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

# Invoke endpoint
import requests
import json

def score_model(customer_ids):
    url = f"https://{workspace_url}/serving-endpoints/churn-prediction-endpoint/invocations"
    headers = {"Authorization": f"Bearer {token}"}
    
    data = {"dataframe_records": [{"customer_id": cid} for cid in customer_ids]}
    
    response = requests.post(url, headers=headers, json=data)
    return response.json()

# Batch inference
predictions_df = fs.score_batch(
    model_uri=f"models:/churn_prediction_model/Production",
    df=customers_df
)
```

### 6. A/B Testing

```python
# Deploy two model versions
w.serving_endpoints.update_config(
    name="churn-prediction-endpoint",
    served_models=[
        ServedModelInput(
            model_name="churn_prediction_model",
            model_version="1",
            workload_size="Small",
            scale_to_zero_enabled=True,
            traffic_percentage=80  # 80% traffic to v1
        ),
        ServedModelInput(
            model_name="churn_prediction_model",
            model_version="2",
            workload_size="Small",
            scale_to_zero_enabled=True,
            traffic_percentage=20  # 20% traffic to v2
        )
    ]
)

# Monitor A/B test results
ab_test_metrics = spark.sql("""
    SELECT
        model_version,
        COUNT(*) as prediction_count,
        AVG(prediction_score) as avg_score,
        AVG(actual_churn) as actual_churn_rate
    FROM ml.predictions_log
    WHERE prediction_date >= current_date() - 7
    GROUP BY model_version
""")
```

### 7. Model Monitoring

```python
from databricks.lakehouse_monitoring import create_monitor

# Create monitor for model predictions
create_monitor(
    table_name="ml.predictions",
    profile_type="InferenceLog",
    model_id_col="model_version",
    prediction_col="prediction",
    timestamp_col="prediction_timestamp",
    granularities=["1 day"],
    slicing_exprs=["customer_segment"]
)

# Data drift detection
drift_metrics = spark.sql("""
    SELECT
        window.start as date,
        feature_name,
        drift_score,
        CASE WHEN drift_score > 0.1 THEN 'HIGH' ELSE 'LOW' END as drift_level
    FROM ml.drift_metrics
    WHERE date >= current_date() - 30
    ORDER BY drift_score DESC
""")

# Automated retraining trigger
if drift_metrics.filter("drift_level = 'HIGH'").count() > 0:
    # Trigger retraining job
    w.jobs.run_now(job_id=retraining_job_id)
```

### 8. Automated ML Pipeline

```python
# Databricks Job configuration
{
  "name": "ML Pipeline - Churn Prediction",
  "tasks": [
    {
      "task_key": "feature_engineering",
      "notebook_task": {
        "notebook_path": "/ml/feature_engineering"
      },
      "new_cluster": {
        "spark_version": "13.3.x-scala2.12",
        "node_type_id": "i3.xlarge",
        "num_workers": 2
      }
    },
    {
      "task_key": "model_training",
      "depends_on": [{"task_key": "feature_engineering"}],
      "notebook_task": {
        "notebook_path": "/ml/model_training"
      },
      "new_cluster": {
        "spark_version": "13.3.x-cpu-ml-scala2.12",
        "node_type_id": "i3.xlarge",
        "num_workers": 4
      }
    },
    {
      "task_key": "model_validation",
      "depends_on": [{"task_key": "model_training"}],
      "notebook_task": {
        "notebook_path": "/ml/model_validation"
      }
    },
    {
      "task_key": "model_deployment",
      "depends_on": [{"task_key": "model_validation"}],
      "notebook_task": {
        "notebook_path": "/ml/model_deployment"
      }
    }
  ],
  "schedule": {
    "quartz_cron_expression": "0 0 2 * * ?",
    "timezone_id": "UTC"
  }
}
```

## Implementation Phases

### Phase 1: Feature Store (Day 1)
- Set up Feature Store
- Implement feature engineering
- Create feature tables
- Test feature serving

### Phase 2: ML Training (Day 2)
- Implement training pipeline
- MLflow experiment tracking
- Hyperparameter tuning
- Model registry

### Phase 3: Deployment (Day 3)
- Model serving endpoints
- Batch inference
- A/B testing setup
- Monitoring

### Phase 4: Automation & Documentation (Day 4)
- Automated ML pipeline
- CI/CD integration
- Architecture diagrams
- MLOps guide

## Deliverables

### Code
- Feature engineering notebooks
- ML training pipelines
- Model serving code
- Monitoring queries
- Databricks Jobs configuration

### Documentation
- Architecture diagram
- "ML Platform Comparison: Databricks vs. SageMaker"
- MLOps reference architecture
- Feature Store implementation guide
- Model deployment patterns

### Metrics
- Training time and cost
- Inference latency (p50, p95, p99)
- Model performance metrics
- Feature computation time
