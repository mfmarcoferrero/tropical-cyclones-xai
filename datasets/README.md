# Datasets and Models

You can find the datasets used and models generated for this work in the following drive folders:

- **[datasets](https://drive.google.com/drive/folders/13PzFjoqmkzvKffLE2kH322pPMQ-hHtBC?usp=drive_link)**
- **[models](https://drive.google.com/drive/folders/13Tlj1mj-FCb-kdbc1e3iMWI2O1FUv3cV?usp=drive_link)**

## Datasets

All the datasets used in this work cover the period from 1980 to 2022. These datasets result from previous reanalysis and research of the source data. Some of these source data are provided confidentially by the EMCWF to POLIMI (e.g., `tc_act_sind.csv`, `ERA5_MJO_v2.csv`, and `sst_trop_raw.csv`) after a data cleaning process. All other data are downloaded from the official ECMWF platform for the ERA5 observation data.

The datasets used are divided into three categories:

### Local Drivers

These datasets consist of all the meteorological variables related to the target zone considered for the anomaly detection problem.

In this folder, you can find:
- **ERA5_16zones_avg_std_10D.csv**: This dataset contains the values of the meteorological variables for each day considered. These data are downloaded from the ECMWF website and have been previously cleaned and formatted to .csv.

### Global Drivers

These datasets include all the large-scale oscillations that are somehow correlated to tropical cyclone formation.

Here you can find:
- **sst_trop_raw.csv**: Sea surface temperature in different oceanic zones (see thesis document) to describe the El Nino Southern Oscillation phenomenon.
- **ERA5_MJO_v2.csv**: An index to describe the Madden-Julian Oscillation phenomenon.
- **MJO_30D_scaled.csv**: Normalized MJO data. For each of the days considered, a 30-day timeseries of the MJO index is considered.
- **ENSO_30D_scaled.csv**: Normalized ENSO data. For each of the days considered, a 30-day timeseries of the SST is considered.

### Target Labels

These datasets show the presence of cyclones for each day of the time period considered.

The documents in this directory are:
- **tc_act_sind.csv**: Labels that represent the presence/absence of cyclones (S.IndAll) and TC formation (S.IndGen).
- **ibtracs_original.csv**: Original source used to generate `tc_act_sind`. This is the most important dataset that tracks cyclone activities around the globe.
- **ibtracs_original_SI.csv**: Same dataset filtered for the South Indian Ocean zone.
- **ibtracs_Z11.csv**: IBTrACS filtered only for the Z11 target zone (see thesis).
- **ibtracs_hotzone.csv**: IBTrACS filtered only for the six most expressive zones, considering the feature selection process (see thesis).

### Datasets Adopted in Relevant Results

The datasets effectively used in the relevant implementations are:
- `tc_act_sind.csv` for the target labels.
- `MJO_30D_scaled.csv` and `ENSO_30D_scaled.csv` for the global drivers.
- `ERA5_16zones_avg_std_10D.csv` for the local drivers.

## Models

In the models directory, you can find all the black-box models generated and saved in the code of the Jupyter notebooks. These models perform tropical cyclone detection at various time horizons from 1 to 10 days. The implementations include:
- **XGBoost**: A gradient boosting random forest model.
- **LSTM Networks**: Long-Short Term Memory neural networks.
- **Hybrid Models**: These combine two LSTM encoders to generate a compressed representation of the input timeseries and an XGBoost classifier.
