# Technical Specification: Enterprise Data Sharing & Marketplace Platform

## Architecture Overview

### System Components
```
Provider Account → Secure Shares → Consumer Accounts (Internal/External)
       ↓                                    ↓
  Private Exchange                   Reader Accounts
       ↓
Snowflake Marketplace
```

## Technology Stack

### Core Technologies
- **Platform**: Snowflake Enterprise Edition
- **Accounts**: Multi-account setup (1 provider, 3+ consumers)
- **Security**: Row-level security, dynamic masking, RBAC
- **Governance**: Object tagging, access policies
- **Monitoring**: Account usage views, query history

### Supporting Technologies
- **IaC**: Terraform with Snowflake provider
- **CI/CD**: GitHub Actions
- **Documentation**: Markdown, architecture diagrams
- **Sample Data**: TPC-H or synthetic e-commerce data

## Detailed Design

### 1. Account Architecture

#### Provider Account
- Database: `SHARED_DATA_PROD`
- Schemas: `PUBLIC_DATA`, `RESTRICTED_DATA`, `INTERNAL_DATA`
- Secure shares for each consumer
- Central governance and tagging

#### Consumer Accounts
- **Internal Consumer**: Full access to internal data
- **Partner Consumer**: Restricted access with masking
- **External Consumer**: Public data only via reader account

### 2. Data Sharing Implementation

#### Secure Shares
```sql
-- Create share for internal consumer
CREATE SHARE INTERNAL_SHARE;
GRANT USAGE ON DATABASE SHARED_DATA_PROD TO SHARE INTERNAL_SHARE;
GRANT USAGE ON SCHEMA PUBLIC_DATA TO SHARE INTERNAL_SHARE;
GRANT SELECT ON ALL TABLES IN SCHEMA PUBLIC_DATA TO SHARE INTERNAL_SHARE;

-- Add consumer account
ALTER SHARE INTERNAL_SHARE ADD ACCOUNTS = internal_consumer_account;
```

#### Cross-Region Sharing
- Replication for cross-region access
- Failover configuration
- Cost optimization strategies

### 3. Security Implementation

#### Row-Level Security
```sql
-- Create row access policy
CREATE ROW ACCESS POLICY customer_region_policy
AS (region STRING) RETURNS BOOLEAN ->
  CASE
    WHEN CURRENT_ROLE() = 'INTERNAL_ROLE' THEN TRUE
    WHEN CURRENT_ROLE() = 'PARTNER_ROLE' THEN region = CURRENT_REGION()
    ELSE FALSE
  END;

-- Apply policy to table
ALTER TABLE customers ADD ROW ACCESS POLICY customer_region_policy ON (region);
```

#### Dynamic Data Masking
```sql
-- Create masking policy for PII
CREATE MASKING POLICY email_mask AS (val STRING) RETURNS STRING ->
  CASE
    WHEN CURRENT_ROLE() IN ('ADMIN', 'INTERNAL_ROLE') THEN val
    ELSE REGEXP_REPLACE(val, '.+@', '****@')
  END;

-- Apply to column
ALTER TABLE customers MODIFY COLUMN email SET MASKING POLICY email_mask;
```

### 4. Governance Framework

#### Tag-Based Classification
```sql
-- Create tags
CREATE TAG pii_tag ALLOWED_VALUES 'high', 'medium', 'low';
CREATE TAG data_classification ALLOWED_VALUES 'public', 'internal', 'confidential';

-- Apply tags
ALTER TABLE customers SET TAG pii_tag = 'high', data_classification = 'confidential';
ALTER TABLE customers MODIFY COLUMN email SET TAG pii_tag = 'high';
```

#### Access Policies
```sql
-- Create access policy based on tags
CREATE TAG-BASED ACCESS POLICY confidential_policy
  ON TAG data_classification = 'confidential'
  FOR SELECT
  TO ROLE partner_role
  USING (FALSE);
```

### 5. Private Data Exchange

#### Setup
```sql
-- Create private exchange
CREATE DATA EXCHANGE my_private_exchange;

-- Create listing
CREATE LISTING customer_analytics_listing
  FOR DATA EXCHANGE my_private_exchange
  AS SELECT * FROM shared_data_prod.public_data.customer_analytics;

-- Add consumers
ALTER LISTING customer_analytics_listing ADD ACCOUNTS = (partner_account_1, partner_account_2);
```

### 6. Reader Accounts

#### Configuration
```sql
-- Create reader account
CREATE MANAGED ACCOUNT reader_external_partner
  ADMIN_NAME = 'partner_admin'
  ADMIN_PASSWORD = '<secure_password>'
  TYPE = READER;

-- Share data with reader account
ALTER SHARE EXTERNAL_SHARE ADD ACCOUNTS = reader_external_partner;
```

#### Cost Attribution
```sql
-- Monitor reader account usage
SELECT * FROM SNOWFLAKE.ACCOUNT_USAGE.METERING_HISTORY
WHERE SERVICE_TYPE = 'READER_ACCOUNT'
  AND ACCOUNT_NAME = 'reader_external_partner';
```

### 7. Monitoring & Cost Management

#### Usage Tracking
```sql
-- Track share usage by consumer
SELECT
  share_name,
  consumer_account,
  SUM(credits_used) as total_credits,
  COUNT(DISTINCT query_id) as query_count
FROM SNOWFLAKE.ACCOUNT_USAGE.QUERY_HISTORY
WHERE database_name IN (SELECT database_name FROM shares)
GROUP BY 1, 2;
```

#### Resource Monitors
```sql
-- Create resource monitor per consumer
CREATE RESOURCE MONITOR partner_monitor
  WITH CREDIT_QUOTA = 1000
  TRIGGERS
    ON 75 PERCENT DO NOTIFY
    ON 100 PERCENT DO SUSPEND;
```

## Data Model

### Sample Datasets
- **Customers**: 1M rows with PII (email, phone, address)
- **Orders**: 10M rows with transaction data
- **Products**: 100K rows with catalog data
- **Analytics**: Pre-aggregated metrics for sharing

### Schema Design
```sql
-- Provider database structure
SHARED_DATA_PROD
├── PUBLIC_DATA (shareable to all)
│   ├── products
│   ├── product_categories
│   └── public_metrics
├── RESTRICTED_DATA (shareable with masking)
│   ├── customers (with RLS and masking)
│   ├── orders (with RLS)
│   └── customer_analytics
└── INTERNAL_DATA (internal only)
    ├── sensitive_customer_data
    └── financial_data
```

## Implementation Phases

### Phase 1: Foundation (Day 1)
- Set up provider and consumer accounts
- Create sample datasets
- Implement basic secure shares

### Phase 2: Security (Day 2)
- Implement row-level security
- Configure dynamic data masking
- Set up tag-based governance

### Phase 3: Advanced Sharing (Day 3)
- Configure Private Data Exchange
- Set up reader accounts
- Implement cross-region sharing

### Phase 4: Monitoring & Documentation (Day 4)
- Build usage dashboards
- Create cost attribution reports
- Write architecture documentation
- Create migration guide

## Deliverables

### Code & Configuration
- Terraform scripts for account setup
- SQL scripts for all objects
- Security policies and governance framework
- Monitoring queries and dashboards

### Documentation
- Architecture diagram (multi-account setup)
- Security implementation guide
- Cost attribution model
- "Enterprise Data Sharing Guide" (customer-facing)
- Migration playbook

### Metrics & Analysis
- Cost comparison: sharing vs. data replication
- Performance benchmarks
- Security compliance checklist
- ROI calculator for data sharing

## Testing Strategy

### Functional Testing
- Verify data access per consumer role
- Test row-level security policies
- Validate data masking rules
- Confirm cross-region access

### Security Testing
- Attempt unauthorized access
- Verify audit logging
- Test policy enforcement
- Validate PII protection

### Performance Testing
- Query performance on shared data
- Concurrent access from multiple consumers
- Large dataset sharing (1TB+)
