# CA
Descriptive statistics and machine learning for Anonymous Customer. Data have been randomized for confidentiality.

# Content:

- edl.ipynb: Extract, Transform, Load from the data pipeline.
- eda.ipynb: generate the order reports and a dashboard.
- etl_service.ipynb: Extract, Transform, Load and generate a report of the service cases.
- forecasting.ipynb: timeseries forecasting using FB Prophet, plus an experimentas part.
- data: the folder with data pipelines (data are not visible for confidentiality).
- reports: the folder to dump reports in csv, xlsx formats, and the dashboard images.

The Jupyter notebooks are mostly wrappers which use helper libraries below.

- support:
  -  etl.py: Extract, Transform, Load.
  -  ml.py: ML and DL functions.
  -  orders.py: the Orders class (inherits from pd.DataFrame)
  -  service.py: functions to run service etl.
  -  utils: orders reports and dashboard.
