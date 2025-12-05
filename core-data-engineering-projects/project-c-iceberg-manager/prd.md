# Product Requirements Document: Iceberg Table Manager

## Overview
Build a local Iceberg table manager demonstrating core table format features including time travel, schema evolution, and snapshot management.

## Goals
- Demonstrate Iceberg fundamentals
- Show ACID transaction capabilities
- Implement time travel queries
- Create practical management tool

## Core Features

### 1. Table Operations
- Create Iceberg tables
- Insert data
- Update records
- Delete records
- Query tables

### 2. Time Travel
- Query historical snapshots
- View snapshot metadata
- Rollback to previous versions
- Compare snapshots

### 3. Schema Evolution
- Add columns
- Drop columns
- Rename columns
- Change column types (compatible changes)

### 4. Snapshot Management
- List all snapshots
- View snapshot details
- Expire old snapshots
- Compact data files

### 5. Query Integration
- Query with DuckDB
- SQL interface
- Metadata queries

## Technical Requirements

### Functionality
- Local Iceberg catalog (Hadoop or SQLite)
- ACID transactions
- Snapshot isolation
- Metadata management

### Usability
- Simple CLI interface
- Clear error messages
- Example workflows

### Quality
- Unit tests
- Integration tests
- Data validation

## Success Metrics
- All Iceberg features demonstrated
- Time travel working correctly
- Schema evolution examples
- < 600 lines of code

## Timeline
2 days implementation
