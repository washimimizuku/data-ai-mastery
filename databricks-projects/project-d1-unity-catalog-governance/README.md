# Project D1: Unity Catalog Governance & Data Mesh

## Overview

Build a production-grade data governance platform demonstrating Unity Catalog, data mesh patterns, and enterprise security on Databricks.

**What You'll Build**: A complete governance platform with centralized metastore, multi-workspace architecture, fine-grained access control, data lineage, and data mesh patterns.

**What You'll Learn**: Unity Catalog setup, data mesh architecture, ABAC, row-level security, data lineage, Delta Sharing, and Lakehouse Federation.

## Time Estimate

**3-4 days (24-32 hours)**

### Day 1: Foundation (8 hours)
- Hours 1-2: Unity Catalog metastore setup
- Hours 3-4: Multi-workspace configuration
- Hours 5-6: Domain catalogs (data mesh)
- Hours 7-8: Sample data loading

### Day 2: Security (8 hours)
- Hours 1-2: Table-level permissions
- Hours 3-4: Column masking views
- Hours 5-6: Row-level security
- Hours 7-8: ABAC policies

### Day 3: Governance (8 hours)
- Hours 1-3: Data classification tags
- Hours 4-5: Delta Sharing setup
- Hours 6-7: External locations
- Hour 8: Lineage verification

### Day 4: Compliance (6-8 hours)
- Hours 1-3: Audit queries and reports
- Hours 4-5: Lakehouse Federation
- Hours 6-8: Documentation

## Prerequisites

### Required Knowledge
- [30 Days of Databricks](https://github.com/washimimizuku/30-days-databricks-data-ai) - Days 1-30
  - Days 1-5: Databricks workspace basics
  - Days 11-15: Unity Catalog fundamentals
  - Days 21-25: Governance and security
- [60 Days Advanced](https://github.com/washimimizuku/60-days-advanced-data-ai) - Days 6-10
  - Data governance patterns

### Technical Requirements
- Databricks workspace with Unity Catalog enabled
- Account admin access (for metastore creation)
- Cloud storage (AWS S3, Azure ADLS, or GCS)
- Understanding of SQL and data governance concepts

### Databricks Account Setup
- Premium or Enterprise tier workspace
- Account Console access
- Cloud provider credentials configured
- Multiple workspaces (optional, for multi-workspace demo)

## Getting Started

### Step 1: Review Documentation
1. `prd.md` - Product requirements and goals
2. `tech-spec.md` - Technical architecture and SQL examples
3. `implementation-plan.md` - Day-by-day implementation guide

### Step 2: Unity Catalog Metastore Setup

**Navigate to Account Console:**
1. Go to `accounts.cloud.databricks.com` (or your cloud-specific URL)
2. Click **Data** → **Create Metastore**
3. Enter metastore name: `my_metastore`
4. Select region (must match workspace region)
5. Configure storage location:
   - **AWS**: `s3://my-metastore-bucket/`
   - **Azure**: `abfss://container@storage.dfs.core.windows.net/`
   - **GCP**: `gs://my-metastore-bucket/`

**Create Storage Credential (UI):**
1. In Account Console, go to **Data** → **Storage Credentials**
2. Click **Create Credential**
3. Enter credential name: `metastore_credential`
4. Provide IAM role ARN (AWS) or service principal (Azure)
5. Click **Create**

**Assign Metastore to Workspace:**
1. In Account Console, go to **Workspaces**
2. Select your workspace
3. Click **Unity Catalog** tab
4. Click **Assign Metastore**
5. Select `my_metastore`
6. Click **Assign**

### Step 3: Create Domain Catalogs (Data Mesh)

Open a Databricks SQL Warehouse or Notebook:

```sql
-- Create catalogs for each domain
CREATE CATALOG sales_domain
  COMMENT 'Sales team data products';

CREATE CATALOG marketing_domain
  COMMENT 'Marketing team data products';

CREATE CATALOG finance_domain
  COMMENT 'Finance team data products';

CREATE CATALOG common_domain
  COMMENT 'Shared data products across domains';

-- Create schemas within sales domain
CREATE SCHEMA sales_domain.customer_data
  COMMENT 'Customer information and profiles';

CREATE SCHEMA sales_domain.transaction_data
  COMMENT 'Sales transactions and orders';

-- Verify catalogs
SHOW CATALOGS;
```

### Step 4: Load Sample Data

```python
# Create sample customer data with PII
from pyspark.sql import SparkSession
from pyspark.sql.functions import *

spark = SparkSession.builder.getOrCreate()

# Generate sample data
customers_df = spark.createDataFrame([
    (1, "John Doe", "john@example.com", "555-0101", "US", "New York", 15000),
    (2, "Jane Smith", "jane@example.com", "555-0102", "US", "California", 22000),
    (3, "Bob Johnson", "bob@example.com", "555-0103", "EU", "London", 18000),
    (4, "Alice Brown", "alice@example.com", "555-0104", "EU", "Paris", 25000),
], ["customer_id", "name", "email", "phone", "region", "city", "total_spent"])

# Write to Unity Catalog
customers_df.write.mode("overwrite").saveAsTable(
    "sales_domain.customer_data.customers"
)

print("✓ Sample data loaded to sales_domain.customer_data.customers")
```

## Core Implementation

### 1. Table-Level Permissions

**Grant Access via UI:**
1. Navigate to **Data Explorer** (left sidebar)
2. Select catalog → schema → table
3. Click **Permissions** tab
4. Click **Grant**
5. Select principal (user/group)
6. Choose privileges: SELECT, MODIFY, etc.
7. Click **Grant**

**Grant Access via SQL:**
```sql
-- Create groups
CREATE GROUP analyst_team;
CREATE GROUP data_engineer_team;

-- Add users to groups (via Account Console → User Management)

-- Grant table-level access
GRANT SELECT ON TABLE sales_domain.customer_data.customers 
  TO `analyst_team`;

GRANT ALL PRIVILEGES ON TABLE sales_domain.customer_data.customers 
  TO `data_engineer_team`;

-- Verify permissions
SHOW GRANTS ON TABLE sales_domain.customer_data.customers;
```

### 2. Column-Level Security (Data Masking)

```sql
-- Create masked view for analysts
CREATE VIEW sales_domain.customer_data.customers_masked AS
SELECT
    customer_id,
    name,
    -- Mask email for non-privileged users
    CASE
        WHEN is_member('pii_access_group') THEN email
        ELSE 'REDACTED'
    END AS email,
    -- Mask phone
    CASE
        WHEN is_member('pii_access_group') THEN phone
        ELSE 'XXX-XXXX'
    END AS phone,
    city,
    region,
    -- Hide financial data from analysts
    CASE
        WHEN is_member('finance_team') THEN total_spent
        ELSE NULL
    END AS total_spent
FROM sales_domain.customer_data.customers;

-- Grant access to masked view
GRANT SELECT ON VIEW sales_domain.customer_data.customers_masked 
  TO `analyst_team`;

-- Revoke access to raw table
REVOKE SELECT ON TABLE sales_domain.customer_data.customers 
  FROM `analyst_team`;
```

### 3. Row-Level Security

**Create Row Filter Function:**
```sql
-- Function to filter rows by region
CREATE FUNCTION sales_domain.customer_data.region_filter(region STRING)
RETURN IF(
    is_member('admin_group'),
    TRUE,  -- Admins see all rows
    region = current_user_region()  -- Others see only their region
);

-- Apply row filter to table
ALTER TABLE sales_domain.customer_data.customers
SET ROW FILTER sales_domain.customer_data.region_filter ON (region);

-- Test: Users will only see rows matching their region
SELECT * FROM sales_domain.customer_data.customers;
```

**View Row Filters in UI:**
1. Go to **Data Explorer**
2. Select table with row filter
3. Click **Details** tab
4. See **Row Filter** section

### 4. Data Classification Tags

**Create and Apply Tags via UI:**
1. Navigate to **Data Explorer**
2. Select table or column
3. Click **Tags** section
4. Click **Add Tag**
5. Enter tag name and value
6. Click **Save**

**Create and Apply Tags via SQL:**
```sql
-- Create tags
CREATE TAG pii_level;
CREATE TAG data_classification;
CREATE TAG compliance_requirement;

-- Apply tags to table
ALTER TABLE sales_domain.customer_data.customers
SET TAGS (
    pii_level = 'high',
    data_classification = 'confidential',
    compliance_requirement = 'GDPR'
);

-- Apply tags to specific columns
ALTER TABLE sales_domain.customer_data.customers
ALTER COLUMN email SET TAGS (pii_level = 'high');

ALTER TABLE sales_domain.customer_data.customers
ALTER COLUMN phone SET TAGS (pii_level = 'high');

-- Query tables by tags
SELECT 
    catalog_name,
    schema_name,
    table_name,
    tag_name,
    tag_value
FROM system.information_schema.table_tags
WHERE tag_name = 'pii_level' 
  AND tag_value = 'high';
```

### 5. Data Lineage

**View Lineage in UI:**
1. Go to **Data Explorer**
2. Select any table
3. Click **Lineage** tab
4. See upstream and downstream dependencies
5. Click nodes to explore relationships

**Query Lineage via API:**
```python
from databricks.sdk import WorkspaceClient

w = WorkspaceClient()

# Get table lineage
lineage = w.lineage.get_lineage_by_table(
    table_name="sales_domain.customer_data.customers"
)

print(f"Upstream tables: {lineage.upstreams}")
print(f"Downstream tables: {lineage.downstreams}")

# Get column-level lineage
column_lineage = w.lineage.get_lineage_by_column(
    table_name="sales_domain.customer_data.customers",
    column_name="email"
)
```

### 6. Delta Sharing

**Create Share via UI:**
1. Go to **Data Explorer**
2. Click **Delta Sharing** (left sidebar)
3. Click **Create Share**
4. Enter share name: `customer_analytics_share`
5. Click **Create**
6. Click **Add Table**
7. Select tables to share
8. Click **Add**

**Create Share via SQL:**
```sql
-- Create share
CREATE SHARE customer_analytics_share
  COMMENT 'Customer analytics data for external partners';

-- Add tables to share
ALTER SHARE customer_analytics_share
ADD TABLE sales_domain.customer_data.customer_metrics;

-- Create recipient
CREATE RECIPIENT external_partner
  COMMENT 'External analytics partner';

-- Grant access
GRANT SELECT ON SHARE customer_analytics_share 
  TO RECIPIENT external_partner;

-- Get activation link
DESCRIBE RECIPIENT external_partner;
```

### 7. External Locations

**Create External Location via UI:**
1. Go to **Data Explorer**
2. Click **External Data** → **External Locations**
3. Click **Create Location**
4. Enter location name: `s3_external_data`
5. Enter URL: `s3://external-data-bucket/`
6. Select storage credential
7. Click **Create**

**Create External Location via SQL:**
```sql
-- Create storage credential (if not exists)
CREATE STORAGE CREDENTIAL aws_credential
  WITH (AWS_IAM_ROLE = 'arn:aws:iam::123456789:role/databricks-external');

-- Create external location
CREATE EXTERNAL LOCATION s3_external_data
  URL 's3://external-data-bucket/'
  WITH (STORAGE CREDENTIAL aws_credential);

-- Create external table
CREATE TABLE sales_domain.external_data.partner_data
  USING DELTA
  LOCATION 's3://external-data-bucket/partner_data/';
```

### 8. Audit Logging

**View Audit Logs in UI:**
1. Go to **Admin Console**
2. Click **Audit Logs**
3. Filter by user, action, or date
4. Export logs if needed

**Query Audit Logs via SQL:**
```sql
-- Recent table access
SELECT
    event_time,
    user_identity.email,
    request_params.full_name_arg as table_name,
    action_name,
    response.status_code
FROM system.access.audit
WHERE action_name IN ('getTable', 'readTable')
  AND event_date >= current_date() - 7
ORDER BY event_time DESC
LIMIT 100;

-- Access summary by user
SELECT
    user_identity.email,
    COUNT(*) as access_count,
    COUNT(DISTINCT request_params.full_name_arg) as unique_tables,
    MAX(event_time) as last_access
FROM system.access.audit
WHERE action_name = 'readTable'
  AND event_date >= current_date() - 30
GROUP BY user_identity.email
ORDER BY access_count DESC;

-- PII access compliance report
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
    ON CONCAT(t.table_catalog, '.', t.table_schema, '.', t.table_name) 
       = a.request_params.full_name_arg
WHERE tt.tag_name = 'pii_level'
  AND a.event_date >= current_date() - 30
GROUP BY 1, 2, 3, 4
ORDER BY total_accesses DESC;
```

### 9. Lakehouse Federation

**Create Connection via UI:**
1. Go to **Data Explorer**
2. Click **External Data** → **Connections**
3. Click **Create Connection**
4. Select connection type (PostgreSQL, MySQL, etc.)
5. Enter connection details
6. Test connection
7. Click **Create**

**Create Connection via SQL:**
```sql
-- Create connection to PostgreSQL
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
WHERE region = 'US'
LIMIT 10;

-- Join Unity Catalog with external data
SELECT
    uc.customer_id,
    uc.name,
    uc.email,
    pg.order_count,
    pg.last_order_date
FROM sales_domain.customer_data.customers uc
JOIN postgres_connection.public.customer_orders pg
    ON uc.customer_id = pg.customer_id
WHERE uc.region = 'US';
```

## Success Criteria

- [ ] Unity Catalog metastore created and assigned to workspace(s)
- [ ] Domain catalogs created (sales, marketing, finance)
- [ ] Sample data loaded with PII fields
- [ ] Table-level permissions configured
- [ ] Column masking views created and tested
- [ ] Row-level security implemented with row filters
- [ ] Data classification tags applied
- [ ] Data lineage verified in UI
- [ ] Delta Sharing configured with external recipient
- [ ] External location created and tested
- [ ] Audit queries working
- [ ] Lakehouse Federation connection established
- [ ] Documentation complete with SQL scripts

## Cost Optimization

### Compute Costs
- Use **SQL Warehouses** for governance queries (cheaper than clusters)
- Choose **Serverless** SQL Warehouses for auto-scaling
- Set auto-stop timeout to 10-15 minutes
- Use **Photon** acceleration for better price/performance

### Storage Costs
- Unity Catalog metastore storage: ~$0.023/GB/month (S3 Standard)
- Use **lifecycle policies** to move old data to cheaper tiers
- Enable **Delta table optimization** to reduce storage:
  ```sql
  OPTIMIZE sales_domain.customer_data.customers;
  VACUUM sales_domain.customer_data.customers RETAIN 168 HOURS;
  ```

### Monitoring Costs
- Check **Account Console** → **Usage** for cost breakdown
- Set up **budget alerts** in cloud provider
- Monitor **DBU consumption** by workspace

**Estimated Monthly Cost** (for this project):
- SQL Warehouse (Serverless, 2X-Small): ~$50-100
- Storage (100 GB): ~$2-5
- **Total**: ~$50-105/month

## Common Challenges

### Metastore Assignment Issues
If workspace doesn't show Unity Catalog, verify region match between metastore and workspace

### Permission Errors
Use `SHOW GRANTS` to debug access issues, ensure groups are created in Account Console

### Row Filter Not Working
Verify function returns BOOLEAN, test with `SELECT region_filter('US')` directly

### Lineage Not Showing
Lineage requires queries to run through Unity Catalog, not external tables

## Learning Outcomes

- Set up Unity Catalog metastore and multi-workspace architecture
- Implement fine-grained access control (table, column, row-level)
- Create data mesh patterns with domain catalogs
- Configure Delta Sharing for external collaboration
- Use Lakehouse Federation to query external systems
- Build audit and compliance reports
- Navigate Databricks UI for governance tasks

## Next Steps

1. Add to portfolio with architecture diagrams
2. Write blog post: "Implementing Data Mesh with Unity Catalog"
3. Continue to Project D2: Delta Live Tables
4. Prepare for Databricks Data Engineer Associate certification

## Resources

- [Unity Catalog Docs](https://docs.databricks.com/data-governance/unity-catalog/index.html)
- [Row-Level Security](https://docs.databricks.com/security/access-control/row-and-column-filters.html)
- [Delta Sharing](https://docs.databricks.com/data-sharing/index.html)
- [Lakehouse Federation](https://docs.databricks.com/query-federation/index.html)
- [Data Mesh on Databricks](https://www.databricks.com/blog/2022/10/19/building-data-mesh-based-databricks-lakehouse.html)
