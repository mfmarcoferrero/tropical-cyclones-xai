# Overview

The 16 Zones Datasets are constructed using original NetCDF files downloaded from the ERA5 Copernicus official source. These datasets are compiled into a final CSV file that includes the most relevant meteorological variables. Each variable in the dataset contains time series data for the previous 10 days.

# Data Sources

The original NetCDF files can be found in the shared Google Drive directory. Additionally, CSV files for each specific variable are also provided.

# Data Processing

The `10_FULL_DATASET` notebook is used to merge these partial datasets into the final comprehensive dataset.

# Contents

- **Original NetCDF Files**: Downloaded from ERA5 Copernicus.
- **CSV Files**: Contain specific variables extracted from the NetCDF files.
- **Final CSV File**: Includes all relevant meteorological variables with their 10-day time series.

# Usage

1. **Accessing the Data**:
   - NetCDF files are available in the shared Google Drive directory.
   - Specific variable CSV files are also available in the same directory.

2. **Merging Data**:
   - Use the `10_FULL_DATASET` notebook to merge the partial datasets into the final result.
