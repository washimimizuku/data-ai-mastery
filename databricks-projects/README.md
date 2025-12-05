# Databricks Platform Projects

Platform-specific projects demonstrating Databricks capabilities including Unity Catalog, Delta Live Tables, and MLflow. These projects require a Databricks workspace (Community Edition or trial available).

## Projects Overview

### Project D1: Unity Catalog Governance
**Time**: 3-4 days (24-32h) | **Focus**: Data governance and security

Implement comprehensive data governance with Unity Catalog including catalogs, schemas, access control, data lineage, and audit logging.

**Stack**: Databricks, Unity Catalog, Delta Lake, SQL, Python

**Key Features**:
- Multi-level namespace (catalog → schema → table)
- Fine-grained access control (GRANT/REVOKE)
- Row-level security and column masking
- Data lineage tracking
- Audit logging and compliance

---

### Project D2: Delta Live Tables Pipeline
**Time**: 3-4 days (24-32h) | **Focus**: Declarative ETL pipelines

Build production ETL pipelines using Delta Live Tables with data quality expectations, monitoring, and automatic error handling.

**Stack**: Databricks, Delta Live Tables, Delta Lake, Python, SQL

**Key Features**:
- Declarative pipeline definitions
- Streaming and batch tables
- Data quality expectations
- Automatic dependency management
- Built-in monitoring and observability

---

### Project D3: MLOps with MLflow
**Time**: 3-4 days (24-32h) | **Focus**: End-to-end ML lifecycle

Implement complete MLOps workflow with MLflow including experiment tracking, model registry, deployment, and monitoring.

**Stack**: Databricks, MLflow, Feature Store, Model Serving, Python

**Key Features**:
- Experiment tracking and comparison
- Feature Store for feature engineering
- Model registry with versioning
- Model serving with REST APIs
- A/B testing and monitoring

---

## Recommended Learning Path

### Sequential Path (9-12 days)
1. **Project D1** - Unity Catalog (3-4 days)
   - Foundation for data governance
   - Required for D2 and D3

2. **Project D2** - Delta Live Tables (3-4 days)
   - Build on Unity Catalog
   - Create production data pipelines

3. **Project D3** - MLOps (3-4 days)
   - Use data from D2
   - Complete ML lifecycle

**Total**: 9-12 days (72-96 hours)

---

## Technology Coverage

| Project | Unity Catalog | Delta Live Tables | MLflow | Feature Store | Model Serving |
|---------|--------------|-------------------|--------|---------------|---------------|
| D1. Unity Catalog | ✅✅ | - | - | - | - |
| D2. Delta Live Tables | ✅ | ✅✅ | - | - | - |
| D3. MLOps | ✅ | - | ✅✅ | ✅✅ | ✅✅ |

---

## Prerequisites

### Required Knowledge
- SQL fundamentals
- Python programming
- Data engineering basics
- ML fundamentals (for D3)

### Recommended Bootcamps
- [100 Days Data & AI](https://github.com/washimimizuku/100-days-data-ai) - Days 1-50
- [30 Days of Databricks](https://github.com/washimimizuku/30-days-databricks-data-ai) - Complete

### Databricks Account
- **Community Edition** (free, limited features)
- **Trial** (14 days, full features) - Recommended
- **Paid** (production use)

Sign up: https://www.databricks.com/try-databricks

---

## Key Features

**All projects include**:
- Step-by-step Databricks UI navigation
- SQL and Python code examples
- Notebook implementations
- Architecture diagrams
- Best practices and patterns
- Cost optimization tips

**All projects demonstrate**:
- Platform-specific features
- Production-ready patterns
- Security and governance
- Performance optimization
- Monitoring and observability

---

## Data Sources

**See [DATA_SOURCES.md](./DATA_SOURCES.md) for comprehensive data source guide.**

### Quick Reference

| Project | Data Sources | Size | Notes |
|---------|-------------|------|-------|
| D1. Unity Catalog | Sample datasets, CSV uploads | 1-10GB | Multi-domain data |
| D2. Delta Live Tables | Streaming + batch sources | 10-100GB | CDC, logs, events |
| D3. MLOps | ML datasets (tabular, images) | 10-50GB | Training data |

### Databricks Sample Datasets
```python
# Available in all Databricks workspaces
display(spark.read.format("delta").load("/databricks-datasets/samples/"))

# Common samples
# - NYC Taxi: /databricks-datasets/nyctaxi/
# - IoT: /databricks-datasets/iot-stream/
# - Retail: /databricks-datasets/retail-org/
```

---

## Cost Considerations

### Databricks Pricing
- **Community Edition**: Free (limited to 1 cluster, 15GB RAM)
- **Trial**: Free for 14 days (full features)
- **Standard**: ~$0.40/DBU + compute costs
- **Premium**: ~$0.55/DBU + compute costs
- **Enterprise**: ~$0.65/DBU + compute costs

### Estimated Costs (Trial/Paid)
- **D1 Unity Catalog**: $20-40 (mostly compute)
- **D2 Delta Live Tables**: $30-50 (streaming costs)
- **D3 MLOps**: $40-60 (training + serving)
- **Total**: $90-150 for all 3 projects

### Cost Optimization Tips
1. Use **Community Edition** for learning (free)
2. Use **trial** for full features (14 days free)
3. **Terminate clusters** when not in use
4. Use **spot instances** for batch workloads
5. **Right-size clusters** (don't over-provision)
6. Use **serverless** where available
7. Set **auto-termination** (15-30 minutes)

---

## Databricks Workspace Setup

### 1. Create Account
```bash
# Sign up at https://www.databricks.com/try-databricks
# Choose cloud provider (AWS, Azure, GCP)
# Select region closest to you
```

### 2. Create Cluster
```python
# Cluster configuration (UI or API)
{
    "cluster_name": "learning-cluster",
    "spark_version": "14.3.x-scala2.12",
    "node_type_id": "i3.xlarge",  # AWS example
    "num_workers": 2,
    "autotermination_minutes": 30,
    "spark_conf": {
        "spark.databricks.delta.preview.enabled": "true"
    }
}
```

### 3. Install Libraries
```python
# Install via UI: Compute → Libraries → Install New
# Or via notebook
%pip install mlflow scikit-learn pandas numpy
```

---

## Project Structure

Each project folder contains:
- `prd.md` - Product requirements and goals
- `tech-spec.md` - Technical implementation details
- `implementation-plan.md` - Day-by-day guide
- `README.md` - Overview and getting started

---

## Learning Outcomes

### After D1 (Unity Catalog)
- Implement data governance at scale
- Configure fine-grained access control
- Set up row-level security and masking
- Track data lineage
- Manage audit logs

### After D2 (Delta Live Tables)
- Build declarative ETL pipelines
- Implement data quality checks
- Handle streaming and batch data
- Monitor pipeline health
- Optimize pipeline performance

### After D3 (MLOps)
- Track ML experiments
- Manage features with Feature Store
- Deploy models to production
- Implement A/B testing
- Monitor model performance

---

## Integration with Other Projects

### With Core Projects
- Use **core-data-engineering** projects as foundation
- Apply **core-ai-development** ML skills

### With Advanced Projects
- **Project 3 (Lakehouse)**: Unity Catalog governance
- **Project 4 (MLOps)**: MLflow patterns
- **Project 9 (Full Platform)**: Databricks as ML platform

### With Snowflake Projects
- Compare governance approaches (Unity Catalog vs Snowflake)
- Evaluate platform trade-offs
- Understand multi-cloud strategies

---

## Why Databricks?

### Strengths
1. **Unified platform** - Data + AI in one place
2. **Delta Lake** - ACID transactions, time travel
3. **Collaborative** - Notebooks, shared workspaces
4. **Scalable** - Auto-scaling clusters
5. **MLOps** - Built-in MLflow integration
6. **Performance** - Photon engine optimization

### Use Cases
- Large-scale data processing
- ML/AI workloads
- Real-time streaming
- Data science collaboration
- Lakehouse architecture

---

## Comparison: Databricks vs Snowflake

| Feature | Databricks | Snowflake |
|---------|-----------|-----------|
| **Focus** | Data + AI | Data warehousing |
| **Processing** | Spark-based | SQL-based |
| **ML Support** | Native (MLflow) | Limited (Snowpark ML) |
| **Streaming** | Strong (Delta Live Tables) | Good (Snowpipe Streaming) |
| **Notebooks** | Built-in | External (Hex, Jupyter) |
| **Governance** | Unity Catalog | Native governance |
| **Cost Model** | DBU + compute | Compute + storage |
| **Best For** | ML/AI, complex ETL | SQL analytics, BI |

**Recommendation**: Learn both platforms to maximize career opportunities.

---

## Success Criteria

### Technical Milestones
- [ ] Unity Catalog with multi-level governance
- [ ] Delta Live Tables pipeline with quality checks
- [ ] MLflow experiment tracking and model registry
- [ ] Model deployment with REST API
- [ ] Monitoring dashboards for all projects

### Portfolio Deliverables
- [ ] Databricks notebooks with documentation
- [ ] Architecture diagrams
- [ ] SQL and Python code examples
- [ ] Performance benchmarks
- [ ] Cost analysis report
- [ ] Screenshots/videos of Databricks UI

---

## Getting Started

### Quick Start (Community Edition)
1. Sign up for Databricks Community Edition
2. Create a cluster (free tier)
3. Import sample notebooks
4. Start with Project D1

### Full Experience (Trial)
1. Sign up for 14-day trial
2. Enable Unity Catalog
3. Create production-grade cluster
4. Complete all 3 projects
5. Export notebooks before trial ends

### Step-by-Step
1. **Week 1**: Set up Databricks, complete D1
2. **Week 2**: Build Delta Live Tables pipeline (D2)
3. **Week 3**: Implement MLOps workflow (D3)
4. **Week 4**: Documentation and portfolio

---

## Additional Resources

### Official Documentation
- [Databricks Documentation](https://docs.databricks.com/)
- [Unity Catalog Guide](https://docs.databricks.com/data-governance/unity-catalog/)
- [Delta Live Tables](https://docs.databricks.com/delta-live-tables/)
- [MLflow on Databricks](https://docs.databricks.com/mlflow/)

### Learning Resources
- [Databricks Academy](https://www.databricks.com/learn/training) - Free courses
- [Databricks Community](https://community.databricks.com/) - Forums
- [Databricks Blog](https://www.databricks.com/blog) - Best practices

### Certifications
- **Databricks Certified Data Engineer Associate**
- **Databricks Certified Machine Learning Associate**
- **Databricks Certified Data Analyst Associate**

---

## Next Steps

1. **Set up Databricks account** (Community or Trial)
2. **Complete 30 Days of Databricks** bootcamp (if not done)
3. **Start with Project D1** (Unity Catalog)
4. **Progress to D2 and D3** sequentially
5. **Document everything** for your portfolio
6. **Consider certification** to validate skills

These projects complement the Snowflake projects and demonstrate platform versatility for your portfolio.
