# Helper libraries

# Content:

- datasetup.py: metadata used during ETL (omitted for confidentiality).
- orders.py: the Orders class (inherits from pd.DataFrame)
- etl.py: Extract, Transform, Load.
- utils: orders reports and dashboard.
  Main assets:
  - load_data: load orders data.
  - validate_input: validate different inputs.
  - make_pivot: generate order header and lineitems reports.
  - make_returns_report: generate service reports.
  - plot_pivot_orders: show the reports inline.
  - plot_dashboard: show the dashboard inline.
- ml.py: ML and DL functions.
- service.py: functions to run service etl.

