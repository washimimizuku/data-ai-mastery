# Advanced Data & AI Projects

Production-grade, comprehensive projects integrating multiple technologies to demonstrate end-to-end data engineering, ML, and GenAI capabilities. These are capstone-level projects requiring 2-4 months each.

## Projects Overview

### Project 1: Kafka Streaming Platform
**Time**: 2-3 months (160-240h) | **Focus**: Real-time event streaming

Build a complete Kafka-based streaming platform with producers, consumers, stream processing, and monitoring.

**Stack**: Kafka, Flink, Spark Streaming, Schema Registry, Kafka Connect, Prometheus, Grafana

---

### Project 2: Data Pipeline with Rust + FastAPI
**Time**: 2-3 months (160-240h) | **Focus**: High-performance data processing

Build a production data pipeline with Rust for processing, FastAPI for APIs, and Iceberg for storage.

**Stack**: Rust, FastAPI, PyO3, Iceberg, Airflow, PostgreSQL, S3

---

### Project 3: Lakehouse Architecture
**Time**: 2-3 months (160-240h) | **Focus**: Modern data lakehouse

Implement a complete lakehouse with Delta Lake/Iceberg, medallion architecture, and Unity Catalog.

**Stack**: Delta Lake, Iceberg, Spark, dbt, Unity Catalog, Great Expectations

---

### Project 4: MLOps Platform
**Time**: 2-3 months (160-240h) | **Focus**: End-to-end ML lifecycle

Build a complete MLOps platform with experiment tracking, model registry, deployment, and monitoring.

**Stack**: MLflow, Kubernetes, KServe, Airflow, Prometheus, Grafana

---

### Project 5: Deep Learning Pipeline
**Time**: 2-3 months (160-240h) | **Focus**: Distributed training and serving

Build a deep learning pipeline with distributed training, model optimization, and production serving.

**Stack**: PyTorch, Ray, Horovod, TensorRT, Triton, MLflow

---

### Project 6: Production RAG System
**Time**: 2-3 months (160-240h) | **Focus**: Enterprise RAG architecture

Build a production-grade RAG system with document ingestion, vector search, reranking, and evaluation.

**Stack**: LangChain, Pinecone/Qdrant, OpenAI/Anthropic, FastAPI, Redis

---

### Project 7: Multi-Agent AI System
**Time**: 2-3 months (160-240h) | **Focus**: Agentic AI orchestration

Build a multi-agent system with LangGraph, tool integration, and complex workflows.

**Stack**: LangGraph, LangChain, OpenAI/Anthropic, FastAPI, PostgreSQL

---

### Project 8: LLM Fine-Tuning Platform
**Time**: 2-3 months (160-240h) | **Focus**: Efficient fine-tuning and serving

Build an LLM fine-tuning platform with LoRA/QLoRA, evaluation, and optimized inference.

**Stack**: Hugging Face, PEFT, vLLM, TGI, MLflow, Modal/RunPod

---

### Project 9: AI-Powered Analytics Platform (Capstone)
**Time**: 3-4 months (240-320h) | **Focus**: Full-stack integration

Build a complete AI-powered analytics platform integrating all technologies from Projects 1-8.

**Stack**: React, FastAPI, Kafka, Snowflake, Databricks, LangChain, Airflow, Kubernetes

---

## Recommended Learning Paths

### Path 1: Data Engineering Focus (6-9 months)
1. Project 1 - Kafka Streaming (2-3 months)
2. Project 2 - Rust Pipeline (2-3 months)
3. Project 3 - Lakehouse (2-3 months)

**Covers**: Streaming, batch processing, lakehouse architecture

---

### Path 2: ML/MLOps Focus (6-9 months)
1. Project 4 - MLOps Platform (2-3 months)
2. Project 5 - Deep Learning (2-3 months)
3. Project 8 - LLM Fine-Tuning (2-3 months)

**Covers**: ML lifecycle, distributed training, LLM fine-tuning

---

### Path 3: GenAI Focus (6-9 months)
1. Project 6 - RAG System (2-3 months)
2. Project 7 - Multi-Agent (2-3 months)
3. Project 8 - LLM Fine-Tuning (2-3 months)

**Covers**: RAG, agents, fine-tuning

---

### Path 4: Full-Stack (12 months)
1. Choose 3 projects from different categories (6-9 months)
2. Project 9 - Full Platform (3-4 months)

**Covers**: Everything, culminating in capstone

---

## Technology Coverage

| Project | Kafka | Rust | Lakehouse | MLOps | DL | RAG | Agents | LLM FT | Full-Stack |
|---------|-------|------|-----------|-------|----|----|--------|--------|------------|
| 1. Kafka | ✅✅ | - | - | - | - | - | - | - | - |
| 2. Rust Pipeline | ✅ | ✅✅ | ✅ | - | - | - | - | - | - |
| 3. Lakehouse | - | - | ✅✅ | - | - | - | - | - | - |
| 4. MLOps | - | - | - | ✅✅ | - | - | - | - | - |
| 5. Deep Learning | - | - | - | ✅ | ✅✅ | - | - | - | - |
| 6. RAG | - | - | - | - | - | ✅✅ | - | - | - |
| 7. Agents | - | - | - | - | - | - | ✅✅ | - | - |
| 8. LLM Fine-Tune | - | - | - | ✅ | ✅ | - | - | ✅✅ | - |
| 9. Full Platform | ✅ | ✅ | ✅ | ✅ | - | ✅ | ✅ | - | ✅✅ |

---

## Prerequisites

### Required
- Complete [100 Days Data & AI](https://github.com/washimimizuku/100-days-data-ai) bootcamp
- Complete [60 Days Advanced](https://github.com/washimimizuku/60-days-advanced-data-ai) bootcamp
- 4-6 core projects completed (from core-data-engineering or core-ai-development)
- Strong programming skills (Python, SQL)
- Understanding of distributed systems

### Recommended
- Cloud platform experience (AWS/Azure/GCP)
- Docker and Kubernetes basics
- CI/CD experience
- Production system experience

---

## Key Features

**All projects include**:
- Complete architecture documentation
- Production-ready code
- Infrastructure as code (Terraform/Helm)
- Monitoring and observability
- CI/CD pipelines
- Comprehensive testing
- Performance benchmarks
- Cost analysis

**All projects demonstrate**:
- System design skills
- Production best practices
- Scalability patterns
- Security considerations
- Performance optimization
- Operational excellence

---

## Data Sources

**See [DATA_SOURCES.md](./DATA_SOURCES.md) for comprehensive data source guide.**

### By Project

| Project | Data Sources | Size | Notes |
|---------|-------------|------|-------|
| 1. Kafka | IoT sensors, clickstream, logs | Streaming | Real-time generation |
| 2. Rust Pipeline | E-commerce, transactions | 100M+ rows | Batch + streaming |
| 3. Lakehouse | Multi-source (APIs, DBs, files) | 1TB+ | Bronze/Silver/Gold |
| 4. MLOps | ML datasets (tabular, images) | 10GB-100GB | Training data |
| 5. Deep Learning | ImageNet, COCO, custom | 100GB+ | Large datasets |
| 6. RAG | Documents, PDFs, wikis | 10K-1M docs | Text corpus |
| 7. Agents | Task descriptions, APIs | N/A | Synthetic + real |
| 8. LLM Fine-Tune | Instruction datasets | 10K-100K examples | Alpaca, Dolly |
| 9. Full Platform | All of the above | Multi-TB | Comprehensive |

---

## Time Investment

### Minimum (6 months)
- 2 comprehensive projects
- 320-480 hours total
- 3-4 hours/day

### Standard (9 months)
- 3 comprehensive projects
- 480-720 hours total
- 3-4 hours/day

### Comprehensive (12 months)
- 5 projects including capstone
- 1000-1200 hours total
- 3-4 hours/day

---

## Cost Considerations

### Cloud Infrastructure
- Development: $100-300/month per project
- Production deployment: $500-2000/month
- Use free tiers and spot instances where possible

### Services
- LLM APIs (Projects 6, 7, 9): $50-200/month
- Managed services: $100-500/month
- Monitoring tools: $0-100/month (free tiers available)

### Total Estimated Cost
- Per project: $200-500 for development
- Full portfolio (5 projects): $1000-2500

**Cost optimization tips**:
- Use local development where possible
- Leverage free tiers (AWS, GCP, Azure)
- Tear down resources when not in use
- Use spot instances for batch workloads
- Consider open-source alternatives

---

## Why These Projects?

1. **Production-grade** - Real-world architecture and best practices
2. **Comprehensive** - End-to-end system design
3. **Portfolio-ready** - Impressive for job applications
4. **Technology integration** - Multiple tools working together
5. **Scalable** - Designed for production scale
6. **Well-documented** - Architecture diagrams and documentation
7. **Measurable** - Performance benchmarks and metrics
8. **Modern stack** - 2025-relevant technologies

---

## Getting Started

### Before You Start
1. Complete prerequisite bootcamps
2. Build 4-6 core projects first
3. Set up cloud accounts (AWS/Azure/GCP)
4. Budget time and money ($200-500 per project)

### For Each Project
1. Review project folder:
   - `prd.md` - Product requirements
   - `tech-spec.md` - Technical architecture
   - `implementation-plan.md` - Week-by-week guide
   - `README.md` - Overview and getting started

2. Set up infrastructure:
   - Local development environment
   - Cloud accounts and credentials
   - CI/CD pipelines

3. Follow implementation plan:
   - Week-by-week milestones
   - Incremental development
   - Regular testing and validation

4. Document everything:
   - Architecture diagrams
   - API documentation
   - Deployment guides
   - Performance benchmarks

---

## Success Criteria

### Technical Excellence
- [ ] Production-ready code quality
- [ ] Comprehensive testing (unit, integration, e2e)
- [ ] Monitoring and observability
- [ ] Security best practices
- [ ] Performance benchmarks met

### Documentation
- [ ] Architecture diagrams
- [ ] API documentation
- [ ] Deployment guides
- [ ] Runbooks and troubleshooting
- [ ] Blog post or technical write-up

### Portfolio Impact
- [ ] GitHub repository with clear README
- [ ] Demo video (5-10 minutes)
- [ ] Live deployment (if applicable)
- [ ] Performance metrics and benchmarks
- [ ] Cost analysis

---

## Next Steps

1. **Choose your path** based on career goals
2. **Start with Project 1-3** for data engineering focus
3. **Start with Project 4-5** for ML/MLOps focus
4. **Start with Project 6-8** for GenAI focus
5. **Complete with Project 9** as your capstone

Each project builds on previous knowledge and can be completed independently or as part of a learning path.
