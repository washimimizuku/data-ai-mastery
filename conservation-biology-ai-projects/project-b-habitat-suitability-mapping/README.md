# Project B: Habitat Suitability Mapping with Remote Sensing

**Time**: 2-3 days | **Difficulty**: Intermediate | **Focus**: Species distribution modeling and habitat analysis

Create habitat suitability models using satellite data and species occurrence records. This project demonstrates practical applications of Species Distribution Modeling (SDM) and remote sensing for conservation planning and climate change impact assessment.

## Learning Objectives

- Build Species Distribution Models (SDM) using machine learning
- Process satellite imagery for habitat characterization
- Assess climate change impacts on species distributions
- Create habitat connectivity and corridor analyses
- Develop conservation prioritization maps

## Tech Stack

- **Python**: Core programming language
- **GDAL/Rasterio**: Geospatial data processing
- **scikit-learn**: Machine learning for SDM
- **MaxEnt**: Maximum entropy species distribution modeling
- **pandas/NumPy**: Data manipulation and analysis
- **matplotlib/plotly**: Spatial visualization
- **folium**: Interactive mapping
- **geopandas**: Spatial data analysis

## Project Structure

```
project-b-habitat-suitability-mapping/
├── README.md
├── requirements.txt
├── src/
│   ├── __init__.py
│   ├── data_loader.py          # Species occurrence and environmental data
│   ├── environmental_layers.py # Climate and habitat variable processing
│   ├── sdm_models.py          # Species distribution modeling algorithms
│   ├── climate_projections.py # Future climate scenario analysis
│   ├── connectivity_analysis.py # Habitat connectivity and corridors
│   └── mapping_dashboard.py   # Interactive habitat suitability maps
├── notebooks/
│   ├── 01_species_occurrence_analysis.ipynb
│   ├── 02_environmental_data_processing.ipynb
│   ├── 03_habitat_suitability_modeling.ipynb
│   └── 04_climate_change_projections.ipynb
├── data/
│   ├── raw/                   # GBIF occurrences, WorldClim data
│   ├── processed/             # Cleaned occurrence and environmental layers
│   └── models/                # Trained SDM models and predictions
├── config/
│   └── config.yaml            # Species and modeling parameters
└── tests/
    └── test_sdm_models.py
```

## Implementation Steps

### Day 1: Data Acquisition and Environmental Layer Processing
1. **Setup Environment**
   - Install geospatial libraries (GDAL, rasterio)
   - Download species occurrence data from GBIF
   - Acquire environmental layers (WorldClim, satellite data)

2. **Species Occurrence Processing**
   - Clean and validate occurrence records
   - Remove spatial bias and duplicates
   - Generate pseudo-absence points

3. **Environmental Data Processing**
   - Process climate variables (temperature, precipitation)
   - Extract habitat variables from satellite imagery
   - Create topographic and soil variables

### Day 2: Species Distribution Modeling
1. **Model Development**
   - Implement MaxEnt species distribution models
   - Build ensemble models with multiple algorithms
   - Cross-validation and model evaluation

2. **Habitat Suitability Mapping**
   - Generate current habitat suitability maps
   - Calculate variable importance and response curves
   - Assess model performance and uncertainty

3. **Model Validation**
   - Independent test data evaluation
   - Spatial cross-validation techniques
   - Expert knowledge validation

### Day 3: Climate Change Analysis and Conservation Planning
1. **Climate Change Projections**
   - Project species distributions under future climate scenarios
   - Calculate range shifts and habitat loss
   - Identify climate refugia and vulnerable populations

2. **Connectivity Analysis**
   - Map habitat connectivity and corridors
   - Identify fragmentation hotspots
   - Design conservation corridor networks

3. **Conservation Prioritization**
   - Integrate multiple species habitat models
   - Identify biodiversity hotspots and gaps
   - Create interactive conservation planning maps

## Key Features

### Species Distribution Models
- **MaxEnt**: Maximum entropy modeling for presence-only data
- **Random Forest**: Ensemble learning for presence-absence data
- **Ensemble Models**: Combining multiple SDM algorithms
- **Uncertainty Mapping**: Model agreement and confidence intervals

### Environmental Variables
- **Climate Data**: Temperature, precipitation, seasonality
- **Topographic Variables**: Elevation, slope, aspect, roughness
- **Habitat Variables**: Land cover, vegetation indices (NDVI, EVI)
- **Human Impact**: Distance to roads, population density, land use

### Climate Change Analysis
- **Future Projections**: RCP scenarios for 2050 and 2070
- **Range Shift Analysis**: Direction and magnitude of distribution changes
- **Climate Refugia**: Areas of stable suitable habitat
- **Vulnerability Assessment**: Species-specific climate sensitivity

### Conservation Applications
- **Protected Area Gap Analysis**: Identifying underrepresented habitats
- **Corridor Design**: Connecting fragmented habitats
- **Restoration Prioritization**: Optimal locations for habitat restoration
- **Monitoring Design**: Strategic placement of survey sites

## Sample Species

The project includes habitat models for:
- **Large Carnivores**: Jaguars, snow leopards, wolves
- **Primates**: Orangutans, lemurs, howler monkeys
- **Endemic Species**: Island birds, mountain plants
- **Migratory Species**: Monarch butterflies, Arctic terns

## Expected Outcomes

1. **High-accuracy habitat models** with AUC >0.8
2. **Climate change impact assessments** for target species
3. **Conservation corridor maps** for landscape connectivity
4. **Interactive web maps** for conservation planning

## Real-World Applications

- **Protected Area Design**: Evidence-based reserve selection
- **Environmental Impact Assessment**: Development project impacts
- **Species Reintroduction**: Suitable habitat identification
- **Climate Adaptation Planning**: Conservation under climate change

## Model Performance Targets

- **Model Accuracy**: AUC >0.8, TSS >0.6 for habitat suitability
- **Spatial Resolution**: 1km resolution for regional analyses
- **Temporal Projections**: Reliable predictions to 2070
- **Multi-species Integration**: Ensemble models for biodiversity hotspots

## Extensions

- **Dynamic Models**: Incorporating population dynamics
- **Dispersal Modeling**: Species movement and colonization
- **Genetic Adaptation**: Local adaptation to environmental conditions
- **Ecosystem Services**: Habitat value for human benefits

## Getting Started

```bash
# Setup environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt

# Download species and environmental data
python src/data_loader.py --species jaguar --region neotropics

# Start with exploration notebook
jupyter notebook notebooks/01_species_occurrence_analysis.ipynb

# Run interactive mapping dashboard
streamlit run src/mapping_dashboard.py
```

## Conservation Biology Concepts

### Species Distribution Modeling
- **Ecological Niche**: Environmental requirements of species
- **Habitat Suitability**: Quality of environment for species survival
- **Presence-Only Data**: Modeling with occurrence records only
- **Pseudo-Absence**: Generating background points for modeling

### Landscape Ecology
- **Habitat Fragmentation**: Breaking up of continuous habitat
- **Connectivity**: Ability of organisms to move between habitats
- **Corridors**: Linear habitat features connecting patches
- **Matrix**: Non-habitat areas surrounding habitat patches

### Climate Change Biology
- **Range Shifts**: Geographic movement of species distributions
- **Climate Refugia**: Areas buffered from climate change
- **Phenological Mismatch**: Timing mismatches in ecological interactions
- **Adaptation vs. Migration**: Species responses to climate change

## Technical Challenges

1. **Spatial Bias**: Uneven sampling effort in occurrence data
2. **Scale Mismatch**: Different resolutions of environmental variables
3. **Model Transferability**: Applying models across space and time
4. **Uncertainty Quantification**: Propagating errors through analyses

## Validation Methods

- **Cross-Validation**: Spatial and temporal data splitting
- **Independent Test Data**: Withheld occurrence records
- **Expert Validation**: Comparison with field knowledge
- **Ensemble Evaluation**: Model agreement and disagreement

## Resources

- [GBIF Species Occurrence Data](https://www.gbif.org/)
- [WorldClim Climate Data](https://www.worldclim.org/)
- [MaxEnt Software](https://biodiversityinformatics.amnh.org/open_source/maxent/)
- [Species Distribution Modeling Guide](https://www.nhm.ac.uk/our-science/our-work/biodiversity/predicts.html)
- [R Package 'dismo'](https://cran.r-project.org/web/packages/dismo/index.html)
