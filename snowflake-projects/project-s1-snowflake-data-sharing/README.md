# Project S1: Enterprise Data Sharing & Marketplace Platform

## Overview

Build a production-grade data sharing platform demonstrating Snowflake's unique data sharing capabilities, secure data exchange without data movement, and multi-tenant architecture patterns.

**What You'll Build**: A complete data sharing ecosystem with provider/consumer accounts, row-level security, dynamic masking, Private Data Exchange, and Snowflake Marketplace integration.

**What You'll Learn**: Multi-account architecture, zero-copy data sharing, enterprise governance, secure views, cost attribution, and data monetization patterns.

## Time Estimate

**3-4 days (24-32 hours)**

### Day 1: Foundation (8 hours)
- Hours 1-2: Multi-account setup
- Hours 3-4: Sample data creation
- Hours 5-6: Basic secure shares
- Hours 7-8: Testing and validation

### Day 2: Security (8 hours)
- Hours 1-3: Row-level security policies
- Hours 4-5: Dynamic data masking
- Hours 6-7: Tag-based governance
- Hour 8: Audit logging

### Day 3: Advanced Sharing (8 hours)
- Hours 1-3: Private Data Exchange
- Hours 4-5: Reader accounts
- Hours 6-7: Cross-region sharing
- Hour 8: Testing

### Day 4: Monitoring (6-8 hours)
- Hours 1-3: Usage tracking and cost attribution
- Hours 4-5: Resource monitors
- Hours 6-8: Documentation

## Prerequisites

### Required Knowledge
- [30 Days of Snowflake](https://github.com/washimimizuku/30-days-snowflake-data-ai) - Days 1-30
  - Days 1-5: Snowflake basics
  - Days 11-15: Security & governance
  - Days 21-25: Data sharing & marketplace

### Technical Requirements
- Snowflake trial account (or paid with ACCOUNTADMIN access)
- SQL knowledge (intermediate level)
- Understanding of data governance concepts

### Snowflake Account Setup
- Enterprise Edition (for data sharing features)
- ACCOUNTADMIN role access
- Multiple accounts for testing (provider + consumers)
- Virtual warehouse configured

## Getting Started

### Step 1: Review Documentation
1. `prd.md` - Product requirements and goals
2. `tech-spec.md` - Technical architecture and SQL examples
3. `implementation-plan.md` - Day-by-day implementation guide

### Step 2: Set Up Snowflake Accounts

**Create Trial Accounts:**
1. Go to https://signup.snowflake.com/
2. Create **Provider Account** (primary account)
3. Create **Consumer Account** (for testing)
4. Note account identifiers (e.g., `ABC12345.us-east-1`)

**Get Account Identifier:**
```sql
-- In Snowflake UI, run:
SELECT CURRENT_ACCOUNT();
SELECT CURRENT_REGION();

-- Full account identifier format:
-- <account_locator>.<region>.<cloud>
-- Example: ABC12345.us-east-1.aws
```

### Step 3: Create Sample Data (Provider Account)

```sql
-- Switch to ACCOUNTADMIN role
USE ROLE ACCOUNTADMIN;

-- Create database and schema
CREATE DATABASE SHARED_DATA_PROD;
CREATE SCHEMA SHARED_DATA_PROD.PUBLIC_DATA;
CREATE SCHEMA SHARED_DATA_PROD.RESTRICTED_DATA;

-- Create warehouse
CREATE WAREHOUSE SHARING_WH
  WAREHOUSE_SIZE = 'XSMALL'
  AUTO_SUSPEND = 60
  AUTO_RESUME = TRUE;

USE WAREHOUSE SHARING_WH;

-- Generate sample customer data
CREATE OR REPLACE TABLE SHARED_DATA_PROD.RESTRICTED_DATA.CUSTOMERS AS
SELECT
    SEQ4() AS customer_id,
    'customer_' || SEQ4() || '@example.com' AS email,
    '+1-555-' || LPAD(UNIFORM(1000, 9999, RANDOM()), 4, '0') AS phone,
    CASE UNIFORM(1, 4, RANDOM())
        WHEN 1 THEN 'US'
        WHEN 2 THEN 'EU'
        WHEN 3 THEN 'ASIA'
        ELSE 'LATAM'
    END AS region,
    UNIFORM(100, 10000, RANDOM()) AS total_spent,
    DATEADD(day, -UNIFORM(1, 365, RANDOM()), CURRENT_DATE()) AS last_order_date
FROM TABLE(GENERATOR(ROWCOUNT => 10000));

-- Create public products table
CREATE OR REPLACE TABLE SHARED_DATA_PROD.PUBLIC_DATA.PRODUCTS AS
SELECT
    SEQ4() AS product_id,
    'Product ' || SEQ4() AS product_name,
    UNIFORM(10, 1000, RANDOM()) AS price,
    CASE UNIFORM(1, 4, RANDOM())
        WHEN 1 THEN 'Electronics'
        WHEN 2 THEN 'Clothing'
        WHEN 3 THEN 'Food'
        ELSE 'Home'
    END AS category
FROM TABLE(GENERATOR(ROWCOUNT => 1000));

SELECT '✓ Sample data created' AS status;
```

## Core Implementation

### 1. Create Secure Share (Provider Account)

**Create Share via UI:**
1. Click **Data** → **Private Sharing** in left sidebar
2. Click **Shared by My Account** tab
3. Click **Share Data** button
4. Select **Create a Direct Share**
5. Enter share name: `CUSTOMER_ANALYTICS_SHARE`
6. Click **Create**

**Create Share via SQL:**
```sql
-- Create secure share
CREATE SHARE CUSTOMER_ANALYTICS_SHARE
  COMMENT = 'Customer analytics data for partners';

-- Grant database access
GRANT USAGE ON DATABASE SHARED_DATA_PROD TO SHARE CUSTOMER_ANALYTICS_SHARE;

-- Grant schema access
GRANT USAGE ON SCHEMA SHARED_DATA_PROD.PUBLIC_DATA TO SHARE CUSTOMER_ANALYTICS_SHARE;

-- Grant table access
GRANT SELECT ON TABLE SHARED_DATA_PROD.PUBLIC_DATA.PRODUCTS 
  TO SHARE CUSTOMER_ANALYTICS_SHARE;

-- View share details
SHOW GRANTS TO SHARE CUSTOMER_ANALYTICS_SHARE;
```

**Add Consumer Account:**
```sql
-- Add consumer account to share
ALTER SHARE CUSTOMER_ANALYTICS_SHARE 
  ADD ACCOUNTS = <consumer_account_identifier>;

-- Example:
-- ALTER SHARE CUSTOMER_ANALYTICS_SHARE 
--   ADD ACCOUNTS = XYZ67890.us-east-1.aws;

-- View consumers
SHOW SHARES LIKE 'CUSTOMER_ANALYTICS_SHARE';
```

### 2. Access Shared Data (Consumer Account)

**View Available Shares (Consumer UI):**
1. Log into consumer account
2. Click **Data** → **Private Sharing**
3. Click **Shared with My Account** tab
4. See available shares from provider

**Create Database from Share:**
```sql
-- In consumer account
USE ROLE ACCOUNTADMIN;

-- Show available shares
SHOW SHARES;

-- Create database from share
CREATE DATABASE CUSTOMER_ANALYTICS_DB
  FROM SHARE <provider_account>.CUSTOMER_ANALYTICS_SHARE;

-- Example:
-- CREATE DATABASE CUSTOMER_ANALYTICS_DB
--   FROM SHARE ABC12345.CUSTOMER_ANALYTICS_SHARE;

-- Query shared data (zero-copy!)
SELECT * FROM CUSTOMER_ANALYTICS_DB.PUBLIC_DATA.PRODUCTS LIMIT 10;

SELECT '✓ Shared data accessible' AS status;
```

### 3. Row-Level Security (Provider Account)

**Create Row Access Policy:**
```sql
-- Create policy to filter by region
CREATE OR REPLACE ROW ACCESS POLICY SHARED_DATA_PROD.RESTRICTED_DATA.customer_region_policy
AS (region STRING) RETURNS BOOLEAN ->
  CASE
    -- Internal users see all data
    WHEN CURRENT_ROLE() IN ('ACCOUNTADMIN', 'INTERNAL_ROLE') THEN TRUE
    -- External consumers see only their region
    WHEN CURRENT_ACCOUNT() = '<consumer_account_1>' THEN region = 'US'
    WHEN CURRENT_ACCOUNT() = '<consumer_account_2>' THEN region = 'EU'
    ELSE FALSE
  END;

-- Apply policy to table
ALTER TABLE SHARED_DATA_PROD.RESTRICTED_DATA.CUSTOMERS
  ADD ROW ACCESS POLICY SHARED_DATA_PROD.RESTRICTED_DATA.customer_region_policy ON (region);

-- Test policy
SELECT region, COUNT(*) 
FROM SHARED_DATA_PROD.RESTRICTED_DATA.CUSTOMERS 
GROUP BY region;
-- Internal users see all regions
-- External consumers see only their assigned region
```

**View Applied Policies:**
```sql
-- Show row access policies
SHOW ROW ACCESS POLICIES;

-- Describe policy
DESC ROW ACCESS POLICY SHARED_DATA_PROD.RESTRICTED_DATA.customer_region_policy;
```

### 4. Dynamic Data Masking (Provider Account)

**Create Masking Policy:**
```sql
-- Create email masking policy
CREATE OR REPLACE MASKING POLICY SHARED_DATA_PROD.RESTRICTED_DATA.email_mask 
AS (val STRING) RETURNS STRING ->
  CASE
    -- Internal roles see full email
    WHEN CURRENT_ROLE() IN ('ACCOUNTADMIN', 'INTERNAL_ROLE') THEN val
    -- External consumers see masked email
    ELSE REGEXP_REPLACE(val, '.+@', '****@')
  END;

-- Create phone masking policy
CREATE OR REPLACE MASKING POLICY SHARED_DATA_PROD.RESTRICTED_DATA.phone_mask 
AS (val STRING) RETURNS STRING ->
  CASE
    WHEN CURRENT_ROLE() IN ('ACCOUNTADMIN', 'INTERNAL_ROLE') THEN val
    ELSE 'XXX-XXX-' || RIGHT(val, 4)
  END;

-- Apply masking policies
ALTER TABLE SHARED_DATA_PROD.RESTRICTED_DATA.CUSTOMERS 
  MODIFY COLUMN email SET MASKING POLICY SHARED_DATA_PROD.RESTRICTED_DATA.email_mask;

ALTER TABLE SHARED_DATA_PROD.RESTRICTED_DATA.CUSTOMERS 
  MODIFY COLUMN phone SET MASKING POLICY SHARED_DATA_PROD.RESTRICTED_DATA.phone_mask;

-- Test masking
SELECT email, phone FROM SHARED_DATA_PROD.RESTRICTED_DATA.CUSTOMERS LIMIT 5;
-- ACCOUNTADMIN sees: customer_1@example.com, +1-555-1234
-- External sees: ****@example.com, XXX-XXX-1234
```

**View Masking Policies:**
```sql
-- Show all masking policies
SHOW MASKING POLICIES;

-- See which columns have masking
SELECT 
    table_catalog,
    table_schema,
    table_name,
    column_name,
    masking_policy_name
FROM SHARED_DATA_PROD.INFORMATION_SCHEMA.COLUMNS
WHERE masking_policy_name IS NOT NULL;
```

### 5. Tag-Based Governance (Provider Account)

**Create and Apply Tags:**
```sql
-- Create tags
CREATE TAG SHARED_DATA_PROD.RESTRICTED_DATA.pii_level
  ALLOWED_VALUES 'high', 'medium', 'low', 'none';

CREATE TAG SHARED_DATA_PROD.RESTRICTED_DATA.data_classification
  ALLOWED_VALUES 'public', 'internal', 'confidential', 'restricted';

-- Apply tags to tables
ALTER TABLE SHARED_DATA_PROD.RESTRICTED_DATA.CUSTOMERS 
  SET TAG SHARED_DATA_PROD.RESTRICTED_DATA.pii_level = 'high',
      SHARED_DATA_PROD.RESTRICTED_DATA.data_classification = 'confidential';

-- Apply tags to specific columns
ALTER TABLE SHARED_DATA_PROD.RESTRICTED_DATA.CUSTOMERS 
  MODIFY COLUMN email SET TAG SHARED_DATA_PROD.RESTRICTED_DATA.pii_level = 'high';

ALTER TABLE SHARED_DATA_PROD.RESTRICTED_DATA.CUSTOMERS 
  MODIFY COLUMN phone SET TAG SHARED_DATA_PROD.RESTRICTED_DATA.pii_level = 'high';

-- Query objects by tag
SELECT 
    tag_database,
    tag_schema,
    tag_name,
    tag_value,
    object_database,
    object_schema,
    object_name,
    column_name
FROM SNOWFLAKE.ACCOUNT_USAGE.TAG_REFERENCES
WHERE tag_name = 'PII_LEVEL'
  AND tag_value = 'high';
```

### 6. Secure Views for Sharing

**Create Secure View:**
```sql
-- Create secure view with business logic
CREATE OR REPLACE SECURE VIEW SHARED_DATA_PROD.PUBLIC_DATA.CUSTOMER_SUMMARY AS
SELECT
    region,
    COUNT(*) AS customer_count,
    AVG(total_spent) AS avg_spent,
    SUM(total_spent) AS total_revenue
FROM SHARED_DATA_PROD.RESTRICTED_DATA.CUSTOMERS
GROUP BY region;

-- Share secure view instead of raw table
GRANT SELECT ON VIEW SHARED_DATA_PROD.PUBLIC_DATA.CUSTOMER_SUMMARY 
  TO SHARE CUSTOMER_ANALYTICS_SHARE;

-- Secure views hide underlying query logic from consumers
```

### 7. Reader Accounts (Provider Account)

**Create Reader Account via UI:**
1. Go to **Admin** → **Accounts**
2. Click **+ Account**
3. Select **Reader Account**
4. Enter account name: `EXTERNAL_PARTNER_READER`
5. Set admin credentials
6. Click **Create**

**Create Reader Account via SQL:**
```sql
-- Create managed reader account
CREATE MANAGED ACCOUNT EXTERNAL_PARTNER_READER
  ADMIN_NAME = 'partner_admin'
  ADMIN_PASSWORD = '<secure_password>'
  TYPE = READER
  COMMENT = 'Reader account for external partner';

-- Show reader accounts
SHOW MANAGED ACCOUNTS;

-- Share data with reader account
ALTER SHARE CUSTOMER_ANALYTICS_SHARE 
  ADD ACCOUNTS = EXTERNAL_PARTNER_READER;
```

**Monitor Reader Account Usage:**
```sql
-- Track reader account consumption
SELECT
    account_name,
    service_type,
    usage_date,
    SUM(credits_used) AS total_credits
FROM SNOWFLAKE.ACCOUNT_USAGE.METERING_HISTORY
WHERE service_type = 'READER_ACCOUNT'
  AND account_name = 'EXTERNAL_PARTNER_READER'
  AND usage_date >= DATEADD(day, -30, CURRENT_DATE())
GROUP BY 1, 2, 3
ORDER BY usage_date DESC;
```

### 8. Private Data Exchange (Provider Account)

**Create Private Exchange via UI:**
1. Go to **Data** → **Provider Studio**
2. Click **+ Exchange**
3. Select **Private Exchange**
4. Enter name: `PARTNER_DATA_EXCHANGE`
5. Add description and logo
6. Click **Create**

**Create Listing:**
```sql
-- Create listing for private exchange
CREATE LISTING CUSTOMER_ANALYTICS_LISTING
  FOR DATA EXCHANGE PARTNER_DATA_EXCHANGE
  AS SELECT * FROM SHARED_DATA_PROD.PUBLIC_DATA.CUSTOMER_SUMMARY;

-- Add description and metadata
ALTER LISTING CUSTOMER_ANALYTICS_LISTING 
  SET COMMENT = 'Customer analytics aggregated by region';

-- Add specific consumer accounts
ALTER LISTING CUSTOMER_ANALYTICS_LISTING 
  ADD ACCOUNTS = (<consumer_account_1>, <consumer_account_2>);

-- View listings
SHOW LISTINGS IN DATA EXCHANGE PARTNER_DATA_EXCHANGE;
```

### 9. Cost Attribution & Monitoring

**Track Share Usage:**
```sql
-- Query history for shared data
SELECT
    query_id,
    query_text,
    user_name,
    role_name,
    database_name,
    execution_time,
    credits_used_cloud_services,
    query_type
FROM SNOWFLAKE.ACCOUNT_USAGE.QUERY_HISTORY
WHERE database_name IN (
    SELECT database_name 
    FROM SNOWFLAKE.ACCOUNT_USAGE.DATABASES 
    WHERE origin = 'SHARE'
)
AND start_time >= DATEADD(day, -7, CURRENT_DATE())
ORDER BY start_time DESC;

-- Credits consumed by share
SELECT
    share_name,
    consumer_account_name,
    SUM(credits_used) AS total_credits,
    COUNT(DISTINCT query_id) AS query_count
FROM SNOWFLAKE.ACCOUNT_USAGE.QUERY_HISTORY qh
JOIN SNOWFLAKE.ACCOUNT_USAGE.SHARES s 
  ON qh.database_name = s.database_name
WHERE start_time >= DATEADD(day, -30, CURRENT_DATE())
GROUP BY 1, 2
ORDER BY total_credits DESC;
```

**Create Resource Monitor:**
```sql
-- Create resource monitor for cost control
CREATE RESOURCE MONITOR PARTNER_SHARE_MONITOR
  WITH CREDIT_QUOTA = 1000
  FREQUENCY = MONTHLY
  START_TIMESTAMP = IMMEDIATELY
  TRIGGERS
    ON 75 PERCENT DO NOTIFY
    ON 90 PERCENT DO SUSPEND_IMMEDIATE
    ON 100 PERCENT DO SUSPEND_IMMEDIATE;

-- Apply to warehouse
ALTER WAREHOUSE SHARING_WH 
  SET RESOURCE_MONITOR = PARTNER_SHARE_MONITOR;

-- View monitor status
SHOW RESOURCE MONITORS;
```

### 10. Audit Logging

**Query Access Logs:**
```sql
-- Track who accessed shared data
SELECT
    query_id,
    query_text,
    user_name,
    role_name,
    execution_status,
    start_time,
    end_time,
    rows_produced
FROM SNOWFLAKE.ACCOUNT_USAGE.QUERY_HISTORY
WHERE database_name = 'SHARED_DATA_PROD'
  AND start_time >= DATEADD(day, -7, CURRENT_DATE())
ORDER BY start_time DESC;

-- Track share access by consumer
SELECT
    consumer_account_name,
    COUNT(*) AS access_count,
    MIN(access_time) AS first_access,
    MAX(access_time) AS last_access
FROM SNOWFLAKE.ACCOUNT_USAGE.DATA_TRANSFER_HISTORY
WHERE source_type = 'SHARE'
  AND transfer_date >= DATEADD(day, -30, CURRENT_DATE())
GROUP BY 1;
```

## Success Criteria

- [ ] Provider account with sample data created
- [ ] Secure share created and granted to consumer
- [ ] Consumer account accessing shared data (zero-copy)
- [ ] Row-level security policy applied and tested
- [ ] Dynamic data masking for PII columns working
- [ ] Tag-based governance implemented
- [ ] Secure views created for business logic
- [ ] Reader account created for external access
- [ ] Private Data Exchange configured
- [ ] Cost attribution queries working
- [ ] Resource monitors set up
- [ ] Audit logging queries documented

## Cost Optimization

### Compute Costs
- Use **XSMALL** warehouses for sharing operations
- Enable **AUTO_SUSPEND** (60 seconds)
- Enable **AUTO_RESUME** for on-demand access
- **Zero-copy sharing** = no data transfer costs

### Storage Costs
- Provider pays for storage
- Consumers pay for compute only
- No data duplication = significant savings
- Shared data counts toward provider's storage

### Reader Account Costs
- Provider pays for reader account compute
- Set **Resource Monitors** to control costs
- Monitor usage with `METERING_HISTORY` views

**Cost Comparison:**
```
Traditional Data Replication:
- Storage: $23/TB/month × 3 consumers = $69/month
- Data transfer: $0.09/GB × 1TB × 3 = $270
- Total: $339/month

Snowflake Data Sharing:
- Storage: $23/TB/month (provider only)
- Data transfer: $0 (zero-copy)
- Compute: Pay per query (consumers)
- Total: $23/month + query costs
```

## Common Challenges

### Share Not Visible to Consumer
Verify account identifier format: `<account_locator>.<region>.<cloud>`

### Row-Level Security Not Working
Check `CURRENT_ACCOUNT()` value matches policy conditions

### Masking Policy Not Applied
Ensure role has privileges to see masked data, test with different roles

### Reader Account Access Issues
Verify reader account credentials and share grants

## Learning Outcomes

- Design multi-account Snowflake architectures
- Implement zero-copy data sharing
- Configure row-level security and data masking
- Set up tag-based governance
- Create and manage reader accounts
- Track costs and attribute to consumers
- Navigate Snowflake UI for data sharing
- Understand Snowflake's unique sharing capabilities

## Next Steps

1. Add to portfolio with architecture diagrams
2. Write blog post: "Snowflake Data Sharing vs Traditional ETL"
3. Continue to Project S2: CDC Streaming with Snowpipe
4. Prepare for SnowPro Advanced Data Engineer certification

## Resources

- [Data Sharing Docs](https://docs.snowflake.com/en/user-guide/data-sharing-intro)
- [Row-Level Security](https://docs.snowflake.com/en/user-guide/security-row-intro)
- [Dynamic Masking](https://docs.snowflake.com/en/user-guide/security-column-ddm-intro)
- [Reader Accounts](https://docs.snowflake.com/en/user-guide/data-sharing-reader-create)
- [Private Data Exchange](https://docs.snowflake.com/en/user-guide/data-exchange-private)
