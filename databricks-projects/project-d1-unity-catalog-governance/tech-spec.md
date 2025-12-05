# Technical Specification: Unity Catalog Governance & Data Mesh

## Architecture Overview

```
Unity Catalog Metastore
    ↓
Multiple Workspaces (Sales, Marketing, Finance)
    ↓
Domain-Oriented Catalogs
    ↓
Fine-Grained Access Control + Lineage
```

## Technology Stack

- **Platform**: Databricks with Unity Catalog
- **Storage**: AWS S3 / Azure ADLS / GCS
- **Governance**: Unity Catalog, Delta Sharing
- **IaC**: Terraform with Databricks provider
- **Language**: Python, SQL

## Detailed Design

### 1. Unity Catalog Setup

```sql
-- Create metastore (one per region)
CREATE METASTORE my_metastore
  STORAGE 's3://my-metastore-bucket/';

-- Assign to workspaces
ALTER METASTORE my_metastore
  ASSIGN WORKSPACE workspace_id_1, workspace_id_2;

-- Create catalogs (domain-oriented)
CREATE CATALOG sales_domain;
CREATE CATALOG marketing_domain;
CREATE CATALOG finance_domain;

-- Create schemas
CREATE SCHEMA sales_domain.customer_data;
CREATE SCHEMA sales_domain.transaction_data;
```

### 2. Data Mesh Implementation

```sql
-- Sales domain catalog
CREATE CATALOG sales_domain COMMENT 'Sales team data products';

-- Marketing domain catalog
CREATE CATALOG marketing_domain COMMENT 'Marketing team data products';

-- Shared/common catalog
CREATE CATALOG common_domain COMMENT 'Shared data products';

-- Domain ownership
GRANT ALL PRIVILEGES ON CATALOG sales_domain TO `sales_team`;
GRANT ALL PRIVILEGES ON CATALOG marketing_domain TO `marketing_team`;

-- Cross-domain access
GRANT SELECT ON CATALOG common_domain TO `sales_team`;
GRANT SELECT ON CATALOG common_domain TO `marketing_team`;
```

### 3. Fine-Grained Access Control

#### Table-Level Security
```sql
-- Grant table access
GRANT SELECT ON TABLE sales_domain.customer_data.customers TO `analyst_role`;
GRANT MODIFY ON TABLE sales_domain.customer_data.customers TO `data_engineer_role`;
```

#### Column-Level Security
```sql
-- Create view with column masking
CREATE VIEW sales_domain.customer_data.customers_masked AS
SELECT
    customer_id,
    name,
    CASE
        WHEN is_member('pii_access_group') THEN email
        ELSE 'REDACTED'
    END AS email,
    CASE
        WHEN is_member('pii_access_group') THEN phone
        ELSE 'REDACTED'
    END AS phone,
    city,
    state
FROM sales_domain.customer_data.customers;

GRANT SELECT ON VIEW sales_domain.customer_data.customers_masked TO `analyst_role`;
```

#### Row-Level Security
```sql
-- Create row filter function
CREATE FUNCTION sales_domain.customer_data.region_filter(region STRING)
RETURN IF(
    is_member('admin_group'),
    TRUE,
    region = current_user_region()
);

-- Apply row filter to table
ALTER TABLE sales_domain.customer_data.customers
SET ROW FILTER sales_domain.customer_data.region_filter ON (region);
```

#### Attribute-Based Access Control (ABAC)
```sql
-- Create dynamic view based on user attributes
CREATE VIEW sales_domain.customer_data.customers_abac AS
SELECT
    customer_id,
    name,
    email,
    region,
    CASE
        WHEN is_member('finance_team') THEN total_spent
        ELSE NULL
    END AS total_spent
FROM sales_domain.customer_data.customers
WHERE
    CASE
        WHEN is_member('global_access') THEN TRUE
        WHEN is_member('us_access') THEN region = 'US'
        WHEN is_member('eu_access') THEN region = 'EU'
        ELSE FALSE
    END;
```

### 4. Data Classification & Tagging

```sql
-- Create tags
CREATE TAG pii_level;
CREATE TAG data_classification;
CREATE TAG compliance_requirement;

-- Apply tags to tables
ALTER TABLE sales_domain.customer_data.customers
SET TAGS (
    pii_level = 'high',
    data_classification = 'confidential',
    compliance_requirement = 'GDPR'
);

-- Apply tags to columns
ALTER TABLE sales_domain.customer_data.customers
ALTER COLUMN email SET TAGS (pii_level = 'high');
ALTER TABLE sales_domain.customer_data.customers
ALTER COLUMN phone SET TAGS (pii_level = 'high');

-- Query by tags
SELECT * FROM system.information_schema.table_tags
WHERE tag_name = 'pii_level' AND tag_value = 'high';
```

### 5. Data Lineage

```python
from databricks.sdk import WorkspaceClient

w = WorkspaceClient()

# Get lineage for a table
lineage = w.lineage.get_lineage_by_table(
    table_name="sales_domain.customer_data.customers"
)

# Get column-level lineage
column_lineage = w.lineage.get_lineage_by_column(
    table_name="sales_domain.customer_data.customers",
    column_name="email"
)
```

### 6. External Locations & Delta Sharing

```sql
-- Create external location
CREATE EXTERNAL LOCATION s3_external_data
URL 's3://external-data-bucket/'
WITH (STORAGE CREDENTIAL aws_credential);

-- Create external table
CREATE TABLE sales_domain.external_data.partner_data
USING DELTA
LOCATION 's3://external-data-bucket/partner_data/';

-- Delta Sharing - create share
CREATE SHARE customer_analytics_share;

-- Add tables to share
ALTER SHARE customer_analytics_share
ADD TABLE sales_domain.customer_data.customer_metrics;

-- Grant access to recipient
GRANT SELECT ON SHARE customer_analytics_share TO RECIPIENT external_partner;
```

### 7. Audit & Compliance

```sql
-- Query audit logs
SELECT
    event_time,
    user_identity.email,
    request_params.full_name_arg as table_name,
    action_name,
    response.status_code
FROM system.access.audit
WHERE action_name IN ('getTable', 'readTable')
    AND event_date >= current_date() - 7
ORDER BY event_time DESC;

-- Access history for specific table
SELECT
    user_identity.email,
    COUNT(*) as access_count,
    MAX(event_time) as last_access
FROM system.access.audit
WHERE request_params.full_name_arg = 'sales_domain.customer_data.customers'
    AND action_name = 'readTable'
GROUP BY user_identity.email;

-- Compliance report - PII access
SELECT
    t.table_catalog,
    t.table_schema,
    t.table_name,
    tt.tag_value as pii_level,
    COUNT(DISTINCT a.user_identity.email) as unique_users,
    COUNT(*) as total_accesses
FROM system.information_schema.tables t
JOIN system.information_schema.table_tags tt
    ON t.table_catalog = tt.catalog_name
    AND t.table_schema = tt.schema_name
    AND t.table_name = tt.table_name
LEFT JOIN system.access.audit a
    ON t.table_catalog || '.' || t.table_schema || '.' || t.table_name = a.request_params.full_name_arg
WHERE tt.tag_name = 'pii_level'
    AND a.event_date >= current_date() - 30
GROUP BY 1, 2, 3, 4;
```

### 8. Lakehouse Federation

```sql
-- Create connection to external database
CREATE CONNECTION postgres_connection
TYPE postgresql
OPTIONS (
    host 'postgres.example.com',
    port '5432',
    user 'readonly_user',
    password secret('postgres_password')
);

-- Query external data
SELECT * FROM postgres_connection.public.customers
WHERE region = 'US';

-- Join Unity Catalog tables with external data
SELECT
    uc.customer_id,
    uc.name,
    pg.order_count
FROM sales_domain.customer_data.customers uc
JOIN postgres_connection.public.customer_orders pg
    ON uc.customer_id = pg.customer_id;
```

## Implementation Phases

### Phase 1: Foundation (Day 1)
- Set up Unity Catalog metastore
- Create multi-workspace setup
- Implement domain catalogs (data mesh)
- Load sample data

### Phase 2: Security (Day 2)
- Implement table/column/row-level security
- Create data masking views
- Set up ABAC policies
- Test access controls

### Phase 3: Governance (Day 3)
- Implement data classification tags
- Set up Delta Sharing
- Configure external locations
- Test lineage tracking

### Phase 4: Compliance & Documentation (Day 4)
- Build audit queries
- Create compliance reports
- Architecture diagrams
- Implementation guide

## Deliverables

### Code
- Terraform scripts for Unity Catalog setup
- SQL scripts for all objects
- Python scripts for lineage queries
- Audit and compliance queries

### Documentation
- Architecture diagram (multi-workspace)
- "Unity Catalog Implementation Guide"
- Data mesh reference architecture
- Security best practices
- Compliance checklist (GDPR, HIPAA)

### Metrics
- Number of tables governed
- Access control policies implemented
- Audit coverage
- Lineage completeness
