# Environmental Data Sources

Comprehensive list of publicly available environmental datasets used across all projects.

## Satellite & Remote Sensing Data

### Landsat & Sentinel
- **Landsat 8/9**: USGS Earth Explorer - https://earthexplorer.usgs.gov/
- **Sentinel-2**: ESA Copernicus Hub - https://scihub.copernicus.eu/
- **Google Earth Engine**: https://earthengine.google.com/
- **NASA Earthdata**: https://earthdata.nasa.gov/

### Forest & Land Cover
- **Global Forest Change**: Hansen et al. - https://earthenginepartners.appspot.com/science-2013-global-forest
- **MODIS Land Cover**: NASA - https://modis.gsfc.nasa.gov/data/dataprod/mod12.php
- **ESA WorldCover**: https://worldcover2020.esa.int/

## Climate & Weather Data

### Historical Climate
- **NOAA Climate Data**: https://www.ncdc.noaa.gov/data-access
- **ERA5 Reanalysis**: Copernicus Climate Change Service - https://cds.climate.copernicus.eu/
- **Berkeley Earth**: http://berkeleyearth.org/data/
- **GHCN Daily**: https://www.ncdc.noaa.gov/ghcn-daily-description

### Real-time Weather
- **OpenWeatherMap API**: https://openweathermap.org/api
- **NOAA Weather API**: https://www.weather.gov/documentation/services-web-api
- **WeatherAPI**: https://www.weatherapi.com/

## Biodiversity & Species Data

### Audio Recordings
- **Xeno-canto**: Bird sounds - https://www.xeno-canto.org/
- **Macaulay Library**: Cornell Lab - https://www.macaulaylibrary.org/
- **BirdNET**: Audio dataset - https://birdnet.cornell.edu/

### Species Occurrence
- **GBIF**: Global Biodiversity Information Facility - https://www.gbif.org/
- **iNaturalist**: https://www.inaturalist.org/
- **eBird**: Cornell Lab - https://ebird.org/

### Conservation Status
- **IUCN Red List**: https://www.iucnredlist.org/
- **CITES**: https://cites.org/eng/resources/species.html

## Environmental Monitoring

### Air Quality
- **EPA Air Quality**: https://www.epa.gov/outdoor-air-quality-data
- **World Air Quality Index**: https://waqi.info/
- **OpenAQ**: https://openaq.org/

### Water Quality
- **EPA Water Quality Portal**: https://www.waterqualitydata.us/
- **USGS Water Data**: https://waterdata.usgs.gov/nwis

### Ocean Data
- **NOAA Ocean Data**: https://www.nodc.noaa.gov/
- **Copernicus Marine**: https://marine.copernicus.eu/

## Carbon & Emissions Data

### Emission Factors
- **EPA Emission Factors**: https://www.epa.gov/climateleadership/ghg-emission-factors-hub
- **IPCC Guidelines**: https://www.ipcc-nggip.iges.or.jp/
- **DEFRA Conversion Factors**: https://www.gov.uk/government/publications/greenhouse-gas-reporting-conversion-factors-2023

### Carbon Monitoring
- **Global Carbon Atlas**: http://www.globalcarbonatlas.org/
- **OCO-2/OCO-3**: NASA CO2 satellites - https://ocov2.jpl.nasa.gov/

## Synthetic & Simulation Data

### Environmental Sensors
- **Sensor simulation scripts** included in Project E
- **IoT device simulators** for air quality, water quality, weather

### Climate Scenarios
- **CMIP6 Climate Models**: https://esgf-node.llnl.gov/projects/cmip6/
- **IPCC Scenario Database**: https://www.ipcc.ch/report/ar6/wg1/

## Data Access Tools

### APIs & Libraries
- **Google Earth Engine Python API**: https://developers.google.com/earth-engine/guides/python_install
- **GDAL/OGR**: https://gdal.org/
- **Rasterio**: https://rasterio.readthedocs.io/
- **xarray**: http://xarray.pydata.org/
- **Planetary Computer**: https://planetarycomputer.microsoft.com/

### Data Processing
- **Dask**: Parallel computing - https://dask.org/
- **Apache Arrow**: Columnar data - https://arrow.apache.org/
- **Zarr**: Cloud-optimized arrays - https://zarr.readthedocs.io/

## Usage Guidelines

1. **Attribution**: Always cite data sources appropriately
2. **Licensing**: Check data usage terms and licenses
3. **Rate Limits**: Respect API rate limits and terms of service
4. **Local Storage**: Cache frequently used datasets locally
5. **Data Quality**: Validate and clean data before analysis

## Project-Specific Data

Each project folder contains:
- `data/raw/` - Original downloaded datasets
- `data/processed/` - Cleaned and preprocessed data
- `data/external/` - Third-party datasets
- `data_download.py` - Automated data retrieval scripts
