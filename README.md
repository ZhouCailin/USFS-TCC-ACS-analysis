# Tree Cover Inequality in the US

Analyze the distribution and inequality of tree cover in the US 2021 based on USFS Tree Canopy Cover Datasets and ACS.

## Data Souce

- US Tree cover data: [US Forest Service](https://data.fs.usda.gov/geodata/rastergateway/treecanopycover/).
- ACS data: [Dexter — Data EXTractER](https://mcdc.missouri.edu/cgi-bin/broker?_PROGRAM=utils.uex2dex.sas&_SERVICE=MCDC&path=/pub/data/acs2021&dset=ustracts5yr&view=0).
- Census tract shapefiles: [2021 TIGER/Line® Shapefiles: Census Tracts](https://www.census.gov/cgi-bin/geo/shapefiles/index.php?year=2021&layergroup=Census+Tracts).

## How to use
### Preporcessing
1. run `download_tracts.py` to download shapefiles of census tracts in continental states.
2. download `nlcd_tcc_conus_2021_v2021-4.tif` from [US Forest Service](https://data.fs.usda.gov/geodata/rastergateway/treecanopycover/) to work directory.
3. run `utc.py` to generate `tree_cover_tract_data.pkl`
### Analysis
- download acs data from [Dexter — Data EXTractER](https://mcdc.missouri.edu/cgi-bin/broker?_PROGRAM=utils.uex2dex.sas&_SERVICE=MCDC&path=/pub/data/acs2021&dset=ustracts5yr&view=0), select variables related with area, population, race, poverty and income. Modify acs file path in scripts before running.
- `regional_analysis.py` to conduct overview and regional analysis.
- `acs_analysis.py` to conduct socio-demographic analysis based on ACS data.
- `regress.py` to conduct tract-level step-wise regression analysis.
- `plot.py` is for KDE and bar plots.

