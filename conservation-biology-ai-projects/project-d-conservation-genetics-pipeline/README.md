# Project D: Conservation Genetics Analysis Pipeline

**Time**: 2-3 days | **Difficulty**: Advanced | **Focus**: Genomic data analysis for conservation decisions

Create a pipeline for analyzing genetic diversity and population structure in endangered species. This project demonstrates practical applications of bioinformatics and population genetics for conservation breeding programs and management decisions.

## Learning Objectives

- Apply bioinformatics tools to conservation genetics problems
- Analyze genetic diversity and population structure
- Implement inbreeding detection and kinship analysis
- Create conservation breeding recommendations
- Understand molecular markers and genomic data types

## Tech Stack

- **Python**: Core programming language
- **BioPython**: Bioinformatics toolkit
- **scikit-learn**: Machine learning for genetic clustering
- **pandas/NumPy**: Data manipulation and analysis
- **matplotlib/plotly**: Genetic data visualization
- **scipy**: Statistical analysis and population genetics
- **Streamlit**: Interactive genetics dashboard
- **ADMIXTURE**: Population structure analysis (external tool)

## Project Structure

```
project-d-conservation-genetics-pipeline/
├── README.md
├── requirements.txt
├── src/
│   ├── __init__.py
│   ├── data_loader.py          # Genetic data loading and preprocessing
│   ├── diversity_analysis.py   # Genetic diversity calculations
│   ├── population_structure.py # Population clustering and admixture
│   ├── inbreeding_analysis.py  # Inbreeding and kinship detection
│   ├── breeding_recommendations.py # Conservation breeding strategies
│   └── genetics_dashboard.py   # Interactive analysis interface
├── notebooks/
│   ├── 01_genetic_data_exploration.ipynb
│   ├── 02_diversity_and_structure.ipynb
│   ├── 03_inbreeding_analysis.ipynb
│   └── 04_breeding_program_design.ipynb
├── data/
│   ├── raw/                   # SNP data, microsatellites, sequences
│   ├── processed/             # Filtered and formatted genetic data
│   └── results/               # Analysis outputs and reports
├── config/
│   └── config.yaml            # Analysis parameters and thresholds
└── tests/
    └── test_genetics_analysis.py
```

## Implementation Steps

### Day 1: Data Processing and Quality Control
1. **Setup Environment**
   - Install BioPython and population genetics libraries
   - Download sample genetic datasets (SNPs, microsatellites)
   - Setup external tools (ADMIXTURE, PLINK)

2. **Data Quality Control**
   - Filter SNPs by call rate and minor allele frequency
   - Remove individuals with excessive missing data
   - Check for Hardy-Weinberg equilibrium deviations

3. **Basic Diversity Analysis**
   - Calculate allelic richness and heterozygosity
   - Compute inbreeding coefficients (FIS)
   - Assess genetic diversity within populations

### Day 2: Population Structure and Admixture Analysis
1. **Population Structure**
   - Principal Component Analysis (PCA) of genetic data
   - ADMIXTURE analysis for population clustering
   - Phylogenetic tree construction from genetic distances

2. **Gene Flow and Migration**
   - Calculate FST between populations
   - Estimate migration rates and gene flow
   - Identify population bottlenecks and founder effects

3. **Spatial Genetics**
   - Isolation by distance analysis
   - Landscape genetics and barrier detection
   - Genetic connectivity assessment

### Day 3: Conservation Applications and Breeding Recommendations
1. **Inbreeding and Kinship Analysis**
   - Pedigree reconstruction from genetic data
   - Kinship coefficient calculations
   - Inbreeding depression assessment

2. **Conservation Breeding**
   - Genetic management recommendations
   - Breeding pair optimization
   - Genetic rescue strategies

3. **Interactive Dashboard**
   - Real-time genetic analysis interface
   - Breeding program simulation
   - Report generation for conservation managers

## Key Features

### Genetic Diversity Analysis
- **Allelic Richness**: Number of alleles per locus
- **Heterozygosity**: Expected and observed heterozygosity
- **Inbreeding Coefficients**: FIS, FIT, FST calculations
- **Effective Population Size**: Contemporary and historical Ne

### Population Structure
- **PCA Analysis**: Principal component visualization
- **ADMIXTURE**: Model-based population clustering
- **Phylogenetic Analysis**: Genetic distance trees
- **Spatial Structure**: Geographic patterns of genetic variation

### Inbreeding and Kinship
- **Kinship Coefficients**: Relatedness between individuals
- **Inbreeding Detection**: Identification of inbred individuals
- **Pedigree Reconstruction**: Family relationships from genetics
- **Runs of Homozygosity**: Genomic inbreeding signatures

### Conservation Applications
- **Breeding Recommendations**: Optimal mating strategies
- **Genetic Rescue**: Identifying source populations for translocation
- **Population Monitoring**: Genetic health assessment over time
- **Reintroduction Planning**: Genetic considerations for releases

## Sample Species and Data Types

The project includes genetic data for:
- **Large Mammals**: Pandas, rhinos, big cats (SNP arrays)
- **Marine Species**: Sea turtles, whales (mitochondrial DNA)
- **Birds**: Island endemics, raptors (microsatellites)
- **Plants**: Rare orchids, trees (chloroplast sequences)

## Expected Outcomes

1. **Comprehensive genetic diversity assessment** for target species
2. **Population structure analysis** revealing management units
3. **Inbreeding detection system** for breeding program management
4. **Conservation breeding recommendations** based on genetic data

## Real-World Applications

- **Zoo Breeding Programs**: Genetic management of captive populations
- **Species Recovery**: Genetic considerations for endangered species
- **Translocation Planning**: Source population selection
- **Habitat Connectivity**: Genetic evidence for corridor effectiveness

## Analysis Performance Targets

- **Data Processing**: Handle 10K+ SNPs and 100+ individuals
- **Population Structure**: Resolve 3-5 genetic clusters
- **Kinship Accuracy**: >90% correct parent-offspring identification
- **Breeding Optimization**: Maximize genetic diversity retention

## Technical Challenges

1. **Missing Data**: Handling incomplete genetic datasets
2. **Population History**: Inferring demographic events from genetics
3. **Marker Selection**: Choosing appropriate molecular markers
4. **Statistical Power**: Adequate sample sizes for reliable inference

## Extensions

- **Genomic Selection**: Predicting fitness from genetic markers
- **Ancient DNA**: Historical genetic diversity analysis
- **Environmental DNA**: Species detection from environmental samples
- **Functional Genomics**: Linking genes to adaptive traits

## Getting Started

```bash
# Setup environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt

# Download sample genetic data
python src/data_loader.py --species panda --marker_type snp

# Start with exploration notebook
jupyter notebook notebooks/01_genetic_data_exploration.ipynb

# Run genetics dashboard
streamlit run src/genetics_dashboard.py
```

## Conservation Genetics Concepts

### Population Genetics Theory
- **Hardy-Weinberg Equilibrium**: Baseline expectations for allele frequencies
- **Genetic Drift**: Random changes in allele frequencies
- **Gene Flow**: Migration and genetic exchange between populations
- **Selection**: Natural selection effects on genetic variation

### Molecular Markers
- **SNPs**: Single nucleotide polymorphisms for genome-wide analysis
- **Microsatellites**: Short tandem repeats for kinship analysis
- **Mitochondrial DNA**: Maternal inheritance and phylogeography
- **Chloroplast DNA**: Plant phylogeography and population structure

### Conservation Applications
- **Genetic Management Units**: Defining populations for management
- **Genetic Rescue**: Increasing fitness through gene flow
- **Outbreeding Depression**: Risks of mixing divergent populations
- **Adaptive Potential**: Genetic basis for environmental adaptation

## Statistical Methods

### Diversity Metrics
- **Allelic Richness**: Rarefaction-corrected allele counts
- **Heterozygosity**: He (expected) and Ho (observed)
- **Inbreeding Coefficient**: FIS = (He - Ho) / He
- **Effective Population Size**: Temporal and linkage disequilibrium methods

### Population Structure
- **F-Statistics**: FST, FIS, FIT hierarchical analysis
- **AMOVA**: Analysis of molecular variance
- **Mantel Tests**: Correlation between genetic and geographic distance
- **Bayesian Clustering**: Model-based population assignment

## Quality Control Procedures

### Data Filtering
- **Call Rate**: Minimum 80% successful genotyping
- **Minor Allele Frequency**: MAF >0.05 for common variants
- **Hardy-Weinberg**: p >0.001 for HWE test
- **Linkage Disequilibrium**: r² <0.8 for independent markers

### Sample Quality
- **Missing Data**: <20% missing genotypes per individual
- **Contamination**: Check for sample mix-ups and contamination
- **Sex Verification**: Confirm sex using genetic markers
- **Relatedness**: Identify and handle closely related individuals

## Resources

- [BioPython Documentation](https://biopython.org/)
- [ADMIXTURE Software](https://dalexander.github.io/admixture/)
- [Conservation Genetics Handbook](https://www.cambridge.org/core/books/conservation-genetics-handbook)
- [Population Genetics Textbook](https://www.sinauer.com/hartl-clark-principles-of-population-genetics-fourth-edition.html)
- [R Package 'adegenet'](https://cran.r-project.org/web/packages/adegenet/index.html)
