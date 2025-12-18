# Conservation Biology Data Sources

Comprehensive list of publicly available conservation biology datasets used across all projects.

## Species Occurrence & Distribution Data

### Global Databases
- **GBIF**: Global Biodiversity Information Facility - https://www.gbif.org/
- **iNaturalist**: Citizen science observations - https://www.inaturalist.org/
- **eBird**: Cornell Lab bird observations - https://ebird.org/
- **OBIS**: Ocean Biodiversity Information System - https://obis.org/

### Regional Databases
- **ALA**: Atlas of Living Australia - https://www.ala.org.au/
- **BISON**: US Geological Survey - https://bison.usgs.gov/
- **NBN Atlas**: UK National Biodiversity Network - https://nbnatlas.org/

## Population & Monitoring Data

### Population Databases
- **Living Planet Database**: WWF population trends - https://www.livingplanetindex.org/
- **PREDICTS**: Projecting Responses of Ecological Diversity - https://www.nhm.ac.uk/our-science/our-work/biodiversity/predicts.html
- **BioTIME**: Biodiversity time series - https://biotime.st-andrews.ac.uk/

### Camera Trap Data
- **Wildlife Insights**: Global camera trap platform - https://www.wildlifeinsights.org/
- **Snapshot Serengeti**: Zooniverse project - https://www.zooniverse.org/projects/zooniverse/snapshot-serengeti
- **Camera Trap Database**: Smithsonian - https://nationalzoo.si.edu/migratory-birds/camera-traps

## Conservation Status & Threats

### Red List Data
- **IUCN Red List**: Global species assessments - https://www.iucnredlist.org/
- **NatureServe**: North American conservation status - https://www.natureserve.org/
- **Species+**: CITES and CMS species database - https://speciesplus.net/

### Threat Assessments
- **IUCN Threats Classification**: Standardized threat categories
- **Global Forest Watch**: Deforestation alerts - https://www.globalforestwatch.org/
- **Protected Planet**: WDPA protected areas - https://www.protectedplanet.net/

## Environmental & Habitat Data

### Climate Data
- **WorldClim**: Global climate surfaces - https://www.worldclim.org/
- **CHELSA**: High resolution climate data - https://chelsa-climate.org/
- **TerraClimate**: Monthly climate data - https://www.climatologylab.org/terraclimate.html

### Habitat & Land Cover
- **ESA WorldCover**: Global land cover - https://worldcover2020.esa.int/
- **MODIS Land Cover**: NASA land cover products - https://modis.gsfc.nasa.gov/data/dataprod/mod12.php
- **Global Forest Change**: Hansen et al. - https://earthenginepartners.appspot.com/science-2013-global-forest

### Elevation & Topography
- **SRTM**: Shuttle Radar Topography Mission - https://www2.jpl.nasa.gov/srtm/
- **ASTER GDEM**: Global Digital Elevation Model - https://asterweb.jpl.nasa.gov/gdem.asp

## Genetic & Genomic Data

### Sequence Databases
- **GenBank**: NCBI genetic sequences - https://www.ncbi.nlm.nih.gov/genbank/
- **BOLD**: Barcode of Life Data Systems - https://www.boldsystems.org/
- **Dryad**: Research data repository - https://datadryad.org/

### Population Genetics
- **PopSet**: Population study datasets - https://www.ncbi.nlm.nih.gov/popset
- **1000 Genomes**: Human genetic variation - https://www.internationalgenome.org/
- **Conservation Genetics Resources**: Published datasets

## Literature & Scientific Data

### Scientific Literature
- **PubMed**: Biomedical literature - https://pubmed.ncbi.nlm.nih.gov/
- **Web of Science**: Citation database - https://www.webofscience.com/
- **Google Scholar**: Academic search - https://scholar.google.com/
- **bioRxiv**: Biology preprints - https://www.biorxiv.org/

### Conservation Journals
- **Conservation Biology**: Wiley journal
- **Biological Conservation**: Elsevier journal
- **Conservation Letters**: Open access
- **Oryx**: Cambridge University Press

## Image & Media Data

### Wildlife Images
- **iNaturalist**: Citizen science photos - https://www.inaturalist.org/
- **Flickr**: Creative Commons wildlife photos - https://www.flickr.com/
- **Wikimedia Commons**: Free media repository - https://commons.wikimedia.org/

### Camera Trap Images
- **LILA BC**: Microsoft AI for Earth - https://lila.science/
- **Caltech Camera Traps**: Annotated datasets - http://lila.science/datasets/caltech-camera-traps
- **iWildCam**: Kaggle competition data - https://www.kaggle.com/c/iwildcam-2019-fgvc6

## Spatial & GIS Data

### Protected Areas
- **WDPA**: World Database on Protected Areas - https://www.protectedplanet.net/
- **UNEP-WCMC**: Protected area datasets - https://www.unep-wcmc.org/

### Administrative Boundaries
- **Natural Earth**: Free vector and raster map data - https://www.naturalearthdata.com/
- **GADM**: Global administrative areas - https://gadm.org/

## APIs & Real-time Data

### Biodiversity APIs
- **GBIF API**: Species occurrence data - https://www.gbif.org/developer/summary
- **iNaturalist API**: Observation data - https://www.inaturalist.org/pages/api+reference
- **eBird API**: Bird observation data - https://ebird.org/api/keygen

### Conservation APIs
- **IUCN Red List API**: Species assessments - https://apiv3.iucnredlist.org/
- **CITES API**: Trade data - https://api.cites.org/
- **Global Forest Watch API**: Forest change data - https://production-api.globalforestwatch.org/

## Data Processing Tools

### R Packages
- **rgbif**: GBIF data access - https://cran.r-project.org/package=rgbif
- **dismo**: Species distribution modeling - https://cran.r-project.org/package=dismo
- **raster**: Spatial data analysis - https://cran.r-project.org/package=raster

### Python Libraries
- **pygbif**: GBIF Python client - https://pygbif.readthedocs.io/
- **ebird-api**: eBird data access - https://pypi.org/project/ebird-api/
- **biopython**: Bioinformatics tools - https://biopython.org/

## Usage Guidelines

1. **Data Licensing**: Always check and comply with data usage terms
2. **Attribution**: Properly cite data sources in publications
3. **Quality Control**: Validate and clean data before analysis
4. **Ethical Use**: Respect sensitive location data for endangered species
5. **Collaboration**: Engage with data providers and conservation practitioners

## Project-Specific Data

Each project folder contains:
- `data/raw/` - Original downloaded datasets
- `data/processed/` - Cleaned and prepared data
- `data/external/` - Third-party datasets and references
- `scripts/data_download.py` - Automated data retrieval
- `scripts/data_validation.py` - Quality control procedures

## Simulated Data

For projects requiring sensitive or unavailable data:
- **Population models**: Simulated demographic data
- **Genetic data**: Synthetic SNP and microsatellite datasets
- **Camera trap images**: Augmented and synthetic wildlife images
- **Threat assessments**: Simulated literature and news data
