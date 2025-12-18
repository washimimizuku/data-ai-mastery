# Project A: Wildlife Population Modeling with ML

**Time**: 2-3 days | **Difficulty**: Intermediate | **Focus**: Population dynamics and species viability analysis

Build ML models to predict wildlife population trends and extinction risks using ecological data. This project demonstrates practical applications of machine learning for Population Viability Analysis (PVA) and conservation decision-making.

## Learning Objectives

- Apply machine learning to population ecology concepts
- Implement Population Viability Analysis (PVA) models
- Predict extinction risks and population trends
- Create conservation priority ranking systems
- Understand demographic stochasticity and environmental variation

## Tech Stack

- **Python**: Core programming language
- **scikit-learn**: Machine learning algorithms
- **PyTorch**: Deep learning for complex population dynamics
- **pandas/NumPy**: Data manipulation and analysis
- **matplotlib/plotly**: Population trend visualization
- **scipy**: Statistical analysis and optimization
- **Streamlit**: Interactive population modeling dashboard

## Project Structure

```
project-a-wildlife-population-modeling/
├── README.md
├── requirements.txt
├── src/
│   ├── __init__.py
│   ├── data_loader.py          # Population data loading and preprocessing
│   ├── population_models.py    # Demographic and stochastic models
│   ├── ml_models.py           # Machine learning population predictors
│   ├── pva_analysis.py        # Population Viability Analysis
│   ├── risk_assessment.py     # Extinction risk calculations
│   └── dashboard.py           # Interactive modeling interface
├── notebooks/
│   ├── 01_population_data_exploration.ipynb
│   ├── 02_demographic_modeling.ipynb
│   ├── 03_ml_population_prediction.ipynb
│   └── 04_conservation_prioritization.ipynb
├── data/
│   ├── raw/                   # Living Planet Database, species counts
│   ├── processed/             # Cleaned population time series
│   └── models/                # Trained population models
├── config/
│   └── config.yaml            # Model parameters and species settings
└── tests/
    └── test_population_models.py
```

## Implementation Steps

### Day 1: Data Acquisition and Demographic Modeling
1. **Setup Environment**
   - Install scientific computing libraries
   - Download Living Planet Database and GBIF occurrence data
   - Setup species-specific datasets

2. **Data Preprocessing**
   - Clean population time series data
   - Handle missing values and outliers
   - Calculate demographic parameters (birth/death rates)

3. **Classical Population Models**
   - Implement exponential and logistic growth models
   - Build Leslie matrix models for age-structured populations
   - Add demographic and environmental stochasticity

### Day 2: Machine Learning Population Prediction
1. **Feature Engineering**
   - Create lagged population variables
   - Add environmental covariates (climate, habitat)
   - Engineer demographic indicators

2. **ML Model Development**
   - Time series forecasting with LSTM networks
   - Random Forest for population trend classification
   - Ensemble methods combining multiple approaches

3. **Population Viability Analysis**
   - Monte Carlo simulations for extinction probability
   - Minimum viable population size calculations
   - Sensitivity analysis for model parameters

### Day 3: Risk Assessment and Conservation Prioritization
1. **Extinction Risk Modeling**
   - Implement IUCN Red List criteria algorithms
   - Calculate extinction probabilities over time horizons
   - Assess population decline rates and trends

2. **Conservation Prioritization**
   - Multi-criteria decision analysis
   - Cost-effectiveness of conservation interventions
   - Spatial prioritization for habitat protection

3. **Interactive Dashboard**
   - Real-time population modeling interface
   - Scenario analysis for conservation planning
   - Export results for conservation reports

## Key Features

### Population Models
- **Exponential Growth**: Simple population growth models
- **Logistic Growth**: Carrying capacity-limited growth
- **Leslie Matrices**: Age-structured population dynamics
- **Stochastic Models**: Environmental and demographic variation

### Machine Learning Approaches
- **LSTM Networks**: Time series population forecasting
- **Random Forest**: Population trend classification
- **Ensemble Methods**: Combining multiple model predictions
- **Feature Importance**: Identifying key population drivers

### Population Viability Analysis
- **Extinction Probability**: Monte Carlo simulation-based PVA
- **Minimum Viable Population**: Critical population thresholds
- **Time to Extinction**: Expected persistence times
- **Sensitivity Analysis**: Parameter uncertainty assessment

### Conservation Applications
- **Red List Assessment**: Automated IUCN criteria evaluation
- **Priority Species**: Ranking species by extinction risk
- **Intervention Planning**: Cost-effective conservation strategies
- **Monitoring Design**: Optimal sampling for population assessment

## Sample Species

The project includes population data for:
- **Large Mammals**: African elephants, tigers, polar bears
- **Marine Species**: Sea turtles, whales, sharks
- **Birds**: Migratory species, island endemics
- **Amphibians**: Frogs and salamanders (decline focus)

## Expected Outcomes

1. **Accurate population models** with <20% prediction error
2. **Extinction risk assessments** aligned with IUCN evaluations
3. **Conservation priority rankings** for target species
4. **Interactive PVA dashboard** for conservation planning

## Real-World Applications

- **Species Action Plans**: Evidence-based conservation strategies
- **Red List Assessments**: Supporting IUCN evaluations
- **Protected Area Planning**: Population-based reserve design
- **Captive Breeding**: Genetic management and reintroduction planning

## Model Performance Targets

- **Population Trend Accuracy**: >80% correct trend classification
- **Extinction Risk Correlation**: R² >0.7 with expert assessments
- **Prediction Horizon**: Reliable 10-year population forecasts
- **Uncertainty Quantification**: Credible prediction intervals

## Extensions

- **Metapopulation Models**: Spatially structured populations
- **Genetic Factors**: Inbreeding depression and genetic drift
- **Climate Change**: Temperature and precipitation impacts
- **Human Impacts**: Habitat loss and fragmentation effects

## Getting Started

```bash
# Setup environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt

# Download population data
python src/data_loader.py --species elephant,tiger,turtle --years 1970-2023

# Start with exploration notebook
jupyter notebook notebooks/01_population_data_exploration.ipynb

# Run interactive dashboard
streamlit run src/dashboard.py
```

## Conservation Biology Concepts

### Population Ecology
- **Demographic Parameters**: Birth rates, death rates, migration
- **Population Growth**: Exponential, logistic, and complex dynamics
- **Carrying Capacity**: Environmental limits to population size
- **Stochasticity**: Random variation in demographic processes

### Conservation Genetics
- **Effective Population Size**: Genetic diversity maintenance
- **Inbreeding Depression**: Fitness costs of small populations
- **Genetic Drift**: Random changes in allele frequencies
- **Gene Flow**: Migration between populations

### Threat Assessment
- **IUCN Red List Criteria**: Standardized extinction risk categories
- **Population Decline**: Rates and patterns of population change
- **Range Reduction**: Geographic contraction of species distributions
- **Threat Severity**: Impact assessment of human activities

## Resources

- [Living Planet Database](https://www.livingplanetindex.org/)
- [IUCN Red List](https://www.iucnredlist.org/)
- [Population Viability Analysis Guide](https://www.fws.gov/endangered/esa-library/pdf/PVA.pdf)
- [Conservation Biology Textbook](https://www.sinauer.com/conservation-biology-for-all.html)
- [R Package 'popbio'](https://cran.r-project.org/web/packages/popbio/index.html)
