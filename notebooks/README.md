# Notebooks

The notebooks implemented in this work cover different chapters of the thesis. Each notebook is well-documented to provide clear insights into the methods and results. Below is an overview of the available notebooks:

- **Data Analysis**:
  - This notebook performs kernel density estimation on the source data to identify the best segmentation for a lower resolution representation of the original ERA5 data.
  - Additionally, it includes an analysis of the distribution of cyclones in the target zone, providing valuable context and insights into the data characteristics.

- **Feature Selection**:
  - A tree-based approach is used in this notebook to identify the meteorological variables that are highly correlated with the target label.
  - This step is crucial for reducing the dimensionality of the dataset and improving the performance and interpretability of the models.

- **Black-box Methods**:
  - This notebook contains implementations of various black-box models used in the study.
  - It also demonstrates the use of the LIME (Local Interpretable Model-agnostic Explanations) technique to explain the predictions made by these black-box models.
  - The focus is on understanding the decision-making process of complex models through local approximations.

- **White-box Models**:
  - This notebook includes the implementation of Bayesian Rule Lists and Decision Trees.
  - These models are inherently interpretable and are used to provide transparent decision rules and insights into the data.
  - The notebook details the training process, evaluation, and interpretation of these white-box models.

Each notebook is designed to be self-contained, allowing you to follow along with the analysis and reproduce the results presented in the thesis. 

For further details on each chapter and the methodologies used, please refer to the thesis document located in the [docs](/docs) directory.
