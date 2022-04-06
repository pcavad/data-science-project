# CA
Descriptive statistics and machine learning for anonymous customer.

# Content:

- edl.ipynb: Extract, Transform, Load from the data pipeline.
- eda.ipynb: order reports and a dashboard.
- etl_service.ipynb: Extract, Transform, Load from service data.
- forecasting.ipynb: timeseries forecasting using FB Prophet, plus an experimentas part.
- data: the folder with data pipelines (empty for confidentiality).
- reports: the folder to dump reports in csv, xlsx formats, and images (dashboard).

The Jupyter notebooks are mostly wrappers which use helper functions.

- support: list of helper functions
  -  etl.py: Extract, Transform, Load.
  -  ml.py: ML and DL functions.
  -  orders.py: the Orders class (inherits from pd.DataFrame)
  -  service.py: functions to run service etl.
  -  utils: orders reports and dashboard.

Data are not visible for confidentiality reasons.
