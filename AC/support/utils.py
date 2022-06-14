'''
Utility functions for EDA.
'''
# Python
import datetime
import ipywidgets as widgets
import logging
import os
import random # used to scramble data for demo
import time # for performance counter
from functools import wraps # for the decorator

# Thrid part
import matplotlib as mpl
import matplotlib.pyplot as plt
plt.style.use('seaborn')
from matplotlib import dates as mpl_dates # used with formatters
import matplotlib.ticker as ticker # used with formatters
import numpy as np
import pandas as pd
import seaborn as sns
import sqlite3
import tabulate
sns.set_theme(style="darkgrid")

# Support helper functions
from support import orders

def my_perf_counter(orig_func):
    '''
    Decorator to measure the performance.
    '''

    @wraps(orig_func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = orig_func(*args, **kwargs)
        finish = time.perf_counter()
        print(f'Finished in {round(finish-start, 2)} second(s)')
        return result

    return wrapper

def load_data(folders = None, files = None):
    '''
    Load orders, models metadata, service.
    Input:
        folders: dict = {
            main: str >>> main directory
            orders: str >>> orders folder
            service: str >>> service folder
        }
        files: dict = {
            orders: str >>> orders file
            models: str >>> models file
            service: str >>> service file
        }
    Return:
        df: pd.DataFrame >>> orders dataframe
        df_models:  pd.DataFrame >>> models dataframe
        df_service:  pd.DataFrame >>> service dataframe
    '''
    # Assign defaults
    try:
        if folders is None:
            folders = {'main' : 'data',
                    'orders' : 'orders',
                    'service' : 'returns'}
            
        if files is None:
            files = {'orders' : 'Backup_orders_after_etl.csv',
                    'models' : 'models.xlsx',
                    'service' : 'service.csv'}
            
        if folders['main'] is None:
            main_folder = 'data'
        else:
            main_folder = folders['main']

        if folders['orders'] is None:
            orders_folder = 'orders'
        else:
            orders_folder = folders['orders']

        if folders['service'] is None:
            service_folder = 'returns'
        else:
            service_folder = folders['service']

        if files['orders'] is None:
            orders_file = 'Backup_orders_after_etl.csv'
        else:
            orders_file = files['orders']

        if files['models'] is None:
            models_file = 'models.xlsx'
        else:
            models_file = files['models']

        if files['service'] is None:
            service_file = 'service.csv'
        else:
            service_file = files['service']
    except KeyError as e:
        print(f"Key error: {e}")
        return None, None, None

    try:
        # df = pd.read_csv(os.path.join(main_folder, orders_file))
        df_models = pd.read_excel(os.path.join(main_folder, models_file))
        df_service = pd.read_csv(os.path.join(main_folder, service_folder, service_file),
                                 usecols=['distributor','sku', 'processed'])
        
        # using SQLlite instead of csv
        conn = sqlite3.connect(os.path.join(main_folder, 'orders_pipeline.db'))
        with conn:
            c = conn.cursor()
            sql_stm = 'SELECT * FROM orders'
            c.execute(sql_stm)
            orders_table = c.fetchall()
            sql_stm = 'PRAGMA table_info(orders)'
            c.execute(sql_stm)
            cols = [col[1] for col in c.fetchall()]
            df = pd.DataFrame(orders_table, columns = cols)
    except FileNotFoundError as e:
        print(e)
        return None, None, None
    except Exception(e):
        print(e)
        return None, None, None
    else:
        df['order_date']=pd.to_datetime(df['order_date'], format='%Y-%m-%d')
        df['lineitem_date']=pd.to_datetime(df['lineitem_date'], format='%Y-%m-%d')
        # for compatibility with SQLite
        # using map because the flag can be 1, 0, or NaN in order lines (map can propagate Null)
        df['distributor'] = df['distributor'].map({0.0:False, 1.0:True})
        # using astype() becasue it can be either 1 or 0
        df['is_order_header'] = df['is_order_header'].astype('bool') # it can be 0 or 1
        df.fillna(np.nan, inplace = True)
        
    return df, df_models, df_service

def validate_input(d, k, input_type, df_cols = None):
    '''
    Validates inputs for dict event and context and returns a value or None.
    Inputs:
        d: dict >>> dictionay
        k: str >>> dictionay key
        input_type: str >>> data type of dk
        df_cols: pd.DataFrame >>> orders dataframe (verify columns)
    Return:
        bool >>> success/failure
        Optional[input_type] >>> the dk value
        Optional[str] >>> error string
    '''
    if input_type == 'key': # check if dictionary key exists
        try:
            input_value = d[k]
        except KeyError as e:
            print(f"Key error: {e}")
            return False, None, None
    else: # for other validations set the key as None if missing and continue
        try:
            input_value = d[k] # check if dictionary key exists
        except KeyError:
            input_value = None # if not make it and set default as None

    try:

        if input_type == 'dataframe':
            assert isinstance(input_value, pd.DataFrame),\
                f'{k} must be a dataframe'
            assert not input_value.empty, f'{k} is empty'

        if input_type == 'datetime':
            assert isinstance(input_value, datetime.datetime),\
                f'{k} must be datetime.datetime'

        if input_value is not None:

            if input_type == 'int':
                assert isinstance(input_value, int), f'{k} must be int'

            if input_type == 'period':
                assert input_value in ('M','Q', 'Y'),\
                    f'{k} must be M, Q, Y'

            if input_type == 'filter':
                assert isinstance(input_value, pd.Series), f'{k} must be a series'
                assert input_value.dtype == 'bool', f'{k} series type must be bool'

            if input_type == 'column_object':
                for col in input_value:
                    assert col in df_cols.columns[df_cols.dtypes == 'object'].values,\
                        f'{col} is not a valid column of type object'

            if input_type == 'column_float':
                for col in input_value:
                    assert col in df_cols.columns[df_cols.dtypes == 'float'].values,\
                        f'{col} is not a valid column of type float'

            if input_type == 'folder':
                assert os.path.isdir(input_value), f'{k} is not a directory'

    except AssertionError as a:
        return False, None, a
    else:
        return True, input_value, None

def make_pivot(event, context, kind):
    '''
    Main report in pivot table for orders, orderlines,
    and the timeseries with break-down of channels.

    Inputs:
        context: dict = {
            df: pd.DataFrame >>> orders dataframe
            }
        event: dict = {
            date_start: datetime.datetime
            date_end: datetime.datetime
            period: str = 'M' >>> 'Y', 'Q', 'M'
            df_filter: Optional[pd.Series] >>> df['channel'] == 'AVE' (best at order header level)
            drill_down: Optional[list[str]] >>> ['billing_company', 'order_id']
            total_col: Optional[str] >>> 'total_usd' or 'total_quantity'
            roll: int = 12 >>> months to count for the rolling statistic
            dump_path: Optional[str] >>> folder to save to Excel/csv
            }
        kind: 'str' >>> 'orders', 'orderlines', 'timeseries'
    Return:
        pd.DataFrame >>> pivot table
        pd.DataFrame >>> raw data before pivoting
    '''
    # Validate and assign variables
    try:
        res, df, msg = validate_input(context, 'df', 'dataframe')
        if res is False:
            raise Exception(msg)

        res, date_start, msg = validate_input(event, 'date_start', 'datetime')
        if res is False:
             raise Exception(msg)

        res, date_end, msg = validate_input(event, 'date_end', 'datetime')
        if res is False:
             raise Exception(msg)

        res, period, msg = validate_input(event, 'period', 'period')
        if res is False:
            raise Exception(msg)
        if period is None:
            period = 'M'

        res, df_filter, msg = validate_input(event, 'df_filter', 'filter')
        if res is False:
            raise Exception(msg)
        if df_filter is None:
            df_filter = True

        res, drill_down, msg = validate_input(event, 'drill_down', 'column_object', df)
        if res is False:
            raise Exception(msg)
        if drill_down is None:
            if kind == 'orders':
                drill_down = ['billing_company']
            if kind == 'orderlines':
                drill_down = ['lineitem_sku']

        res, total_col, msg = validate_input(event, 'total_col', 'column_float', df)
        if res is False:
            raise Exception(msg)
        if total_col is None:
            total_col = ['total_usd']

        res, roll, msg = validate_input(event, 'roll', 'int')
        if res is False:
            raise Exception(msg)
        if roll is None:
            roll = 6

        res, dump_path, msg = validate_input(event, 'dump_path', 'folder')
        if res is False:
            raise Exception(msg)

    except Exception as e:
        print(e)
        return None, None

    # Assign variables specific to orders
    if kind == 'orders':
        df_filtered = orders.Orders(df).get_orders(
            date_start,
            date_end,
            df_filter
            )
        period_column = 'order_date'
        pivot_index = drill_down
        pivot_columns = ['period']
        pivot_values = 'total_usd'

    # Assign variables specific to orderlines
    if kind == 'orderlines':
        df_filtered = orders.Orders(df).get_orderlines(
            date_start,
            date_end,
            df_filter
            )
        period_column = 'lineitem_date'
        pivot_index = drill_down
        pivot_columns = ['period']
        pivot_values = 'lineitem_quantity'
            
    # Assign variables specific to the timeseries
    if kind == 'timeseries':
        df_filtered = orders.Orders(df).get_timeseries(
            date_start,
            date_end,
            total_col
            )
        period_column = 'order_date'
        pivot_index = 'order_date'
        pivot_columns = 'channel'
        pivot_values = total_col # total_usd or total_quantity

    # Make the filtered dataframe and assert that it is not empty
    try:
        assert not df_filtered.empty, 'no orders data for the selection'
    except AssertionError as a:
        print(a)
        return None, None
    
    # Make the period for re-sampling
    df_filtered['period'] = df_filtered[period_column].dt.to_period(period)

    # add the Billing Company and Channel (option to drill down the orderlines)
    if kind == 'orderlines':
        df_filtered=df_filtered\
        .join(df.loc[df['distributor']==True,['order_id','billing_company', 'channel']]\
        .set_index('order_id').dropna(),on='order_id', how='inner')
            
    # Make the pivot table
    df_pivot = df_filtered\
    .pivot_table(
        values = pivot_values
        , index = pivot_index
        , columns = pivot_columns
        , aggfunc = np.sum
        , margins = True
        , margins_name = 'Total'
        , fill_value = 0
        )
    
    # The timeseries expands the pivot table with more statistics
    if kind == 'timeseries':
        
        # Build the index to resample the data according to period (month)
        # Using the date start/end range in order to fill the months with no business
        new_index =\
        pd.date_range(
            date_start,
            date_end)
            # df_filtered['order_date'].min()
            # , df_filtered['order_date'].max())
        
        # Create the pivot table in which orders are aggregated on a period (monthly basis) for the sale channels
        df_pivot = df_pivot\
        .reindex(new_index, fill_value=0)\
        .resample(period).sum() # re-sample on a monthly basis with SUM
        
        # set index name
        df_pivot.index.name = 'period'

        # Flatten columns and set index columns name
        df_pivot.columns = df_pivot.columns.levels[1]
        
        # Concatenate with different aggregations
        df_pivot = pd.concat(
            [
            df_pivot
            , df_pivot.diff()
                .add_suffix('_diff') # Difference
            , df_pivot.rolling(roll).mean()
                .interpolate() # Interpolate linearly for missing values
                    .add_suffix('_roll') # Rolling mean
            , df_pivot.cumsum()
                .add_suffix('_cum') # Cumulated
            ]
            , axis = 1
            ).fillna(0)
        
        # set index columns name
        df_pivot.columns.name = ''

    # Save xls
    if dump_path:
        df_pivot.to_excel(os.path.join(dump_path, kind + '_pivot.xlsx'))
        df_filtered.to_excel(os.path.join(dump_path, kind + '_raw.xlsx'), index = False)

    return df_pivot, df_filtered

def make_returns_report(event, context):
    '''
    Report of the defect rate by distributor.

    Inputs:
        context: dict = {
            df: pd.DataFrame >>> orders
            df_models: pd.DataFrame >>> models, skus, prices
            df_service: pd.DataFrame >>> returns from distributors
            }
        event: dict = {
            dump_path: Optional[str] >>> folder to save to Excel/csv
            }
    Return:
        df_return_rates_grouped: pd.DataFrame >>> the return rates table
    '''

    # Validate and assign variables
    try:
        res, df, msg = validate_input(context, 'df', 'dataframe')
        if res is False:
            raise Exception(msg)
        res, df_models, msg = validate_input(context, 'df_models', 'dataframe')
        if res is False:
            raise Exception(msg)
        res, df_service, msg = validate_input(context, 'df_service', 'dataframe')
        if res is False:
            raise Exception(msg)
        res, dump_path, msg = validate_input(event, 'dump_path', 'folder')
        if res is False:
            raise Exception(msg)
    except Exception as e:
        print(e)
        return None

    kwargs = {
        'context' : {'df': df},
        'event' : {
            'date_start': datetime.datetime(2016,1,1,0,0,0), # Hard coded - from start
            'date_end': datetime.datetime(2050,12,31,0,0,0), # Hard coded - forever
            'period': 'Y',
            'df_filter': None,
            'drill_down': ['billing_company', 'lineitem_sku'],
            'dump_path': None
            },
        'kind' : 'orderlines'
        }

    # Make the distributors and orders dataframe. Validations made by the underlying function
    df_orders_for_sku, _ = make_pivot(**kwargs)
    if not isinstance(df_orders_for_sku, pd.DataFrame): # if the function call fails it will return None
        return None
    df_orders_for_sku =\
    df_orders_for_sku.loc[df_orders_for_sku.index != 'Total', 'Total']\
        .reset_index()

    # Make the weight to calculate the volume of the returns vs volume of the sales
    df_models['price_index'] = df_models.price / df_models.price.max()

    # Make the distributors and returns dataframe (restrict to processed items)
    df_service_for_sku = df_service\
        .loc[df_service['processed'] == True,:]\
        .groupby(['distributor','sku'], as_index=False).size()

    # Join the datframes
    df_return_rates =\
    df_orders_for_sku\
        .join(
            df_service_for_sku.set_index(['distributor', 'sku'])
            , on=['billing_company', 'lineitem_sku'], how='outer'
            )\
        .join(
            df_models[['sku', 'price_index']].set_index('sku')
            , on='lineitem_sku', how='inner'
            )\
        .sort_values(['billing_company', 'lineitem_sku'])\
        .reset_index(drop=True)

    # Make the weighted sales and returns
    df_return_rates = df_return_rates[df_return_rates.Total.notna()] # removes direct RMA (no matching sales)
    df_return_rates['orders'] = df_return_rates['Total'] * df_return_rates['price_index']
    df_return_rates['service'] = df_return_rates['size'] * df_return_rates['price_index']

    # Calculate the return rate for each distributor
    df_return_rates_grouped =\
        df_return_rates.groupby('billing_company')['orders', 'service'].sum() # Calculate the orders and returns by distributor
    df_return_rates_grouped['return_rate']\
        = df_return_rates_grouped['service'] / df_return_rates_grouped['orders'] * 100 # Calculate the return rate
    df_return_rates_grouped.drop(['orders', 'service'], axis=1, inplace=True) # Just keep the returns rate
    df_return_rates_grouped\
        = df_return_rates_grouped[df_return_rates_grouped['return_rate']!=0] # Remove distributors which never returned defectives
    df_return_rates_grouped.loc['--- Average ---','return_rate']\
        = df_return_rates_grouped['return_rate'].mean()

    # Save
    if dump_path:
        df_return_rates_grouped.to_excel(os.path.join(dump_path, 'return_rates.xlsx'))
        df_return_rates_grouped.to_csv(os.path.join(dump_path, 'return_rates.csv'))

    return df_return_rates_grouped

def show_date():
    '''
    It shows the date pickers.

    Return:
    date_start: datime.date, date_end: datime.date >>> start and end dates (datepicker)
    '''
    date_start = widgets.DatePicker(
        description='Start date',
        disabled=False,
        value = datetime.date(datetime.date.today().year,1,1)
        )

    date_end = widgets.DatePicker(
        description='End date',
        disabled=False,
        value = datetime.date.today()
        )

    display(date_start, date_end)

    return date_start, date_end

def make_datetime(dt):
    '''
    It gets the datepicker as date and makes it into a datetime.

    Input:
        dt: datetime.date
    Returns:
        datetime.datetime >>> with time = 0,0,0
    '''
    return datetime.datetime.combine(dt, datetime.time(0,0,0))

def plot_pivot_orders(event, context):
    '''
    It plots the reports in 3 widgets and dumps xls/csv files.

    Inputs:
        context: dict = {
            df: pd.DataFrame >>> orders dataframe
            df_models: pd.DataFrame >>> models, skus, prices
            df_service: pd.DataFrame >>> returns from distributors
            }
        event: dict = {
            date_start: datetime.datetime
            date_end: datetime.datetime
            df_filter: Optional[pd.Series] >>> df['channel'] == 'AVE'
            period_orders: str = 'M' >>> 'Y', 'Q', 'M'
            drill_down_orders: Optional[list[str]] >>> ['billing_company', 'order_id']
            period_lines: str = 'Y' >>> 'Y', 'Q', 'M'
            drill_down_lines: Optional[list[str]] >>> ['lineitem_model', 'lineitem_sku']
            dump_path: Optional[str] >>> folder to save to Excel/csv
            log_path: Optional[str] >>> folder to save logging with output (for IDE which don't show widgets)
            Jupyter: Optional[bool] >>> Jupyter notebook
            }
    '''

    # Assigning variables and run only key level error validations
    df = validate_input(context,'df', 'key')[1] # returns only the actual value or None
    df_models = validate_input(context,'df_models', 'key')[1]
    df_service = validate_input(context,'df_service', 'key')[1]
    date_start = validate_input(event,'date_start', 'key')[1]
    date_end = validate_input(event,'date_end', 'key')[1]
    df_filter = validate_input(event,'df_filter', 'key')[1]
    period_orders = validate_input(event,'period_orders', 'key')[1]
    drill_down_orders = validate_input(event,'drill_down_orders', 'key')[1]
    period_lines = validate_input(event,'period_lines', 'key')[1]
    drill_down_lines = validate_input(event,'drill_down_lines', 'key')[1]
    dump_path = validate_input(event,'dump_path', 'key')[1]
    jupyter = validate_input(event,'jupyter', 'key')[1]

    # check that the log path folder exists
    res, log_path, msg = validate_input(event, 'log_path', 'folder')
    if res is False:
        raise Exception(msg)
        
    # check if jupyter is bool
    if not isinstance(jupyter, bool):
        jupyter = True

    # For optional values the defaults will be assigned by the undelying functions
    
    # orders dictionaries
    context = {'df': df}
    event = {
        'date_start': date_start,
        'date_end': date_end,
        'period': period_orders,
        'df_filter': df_filter,
        'drill_down': drill_down_orders,
        'dump_path': dump_path
        }
    
    # orders dataframe
    headers, _ = make_pivot(event, context, 'orders')
    if not isinstance(headers, pd.DataFrame):
        raise Exception('Plot pivot orders failed')

    # orders details dataframe
    context = {'df': df}
    event = {
        'date_start': date_start,
        'date_end': date_end,
        'period': period_lines,
        'df_filter': df_filter,
        'drill_down': drill_down_lines,
        'dump_path': dump_path
        }
    lines, _ = make_pivot(event, context, 'orderlines')
    if not isinstance(lines, pd.DataFrame):
        raise Exception('Plot pivot orders lines failed')

    # returns by distributor
    context = {
        'df': df,
        'df_models': df_models,
        'df_service': df_service
        }
    event = {'dump_path': dump_path}
    return_rates = make_returns_report(event, context)
    if not isinstance(return_rates, pd.DataFrame):
        raise Exception('Plot returns failed')

    # Jupyter notebook
    if jupyter:
        out1 = widgets.Output()
        out2 = widgets.Output()
        out3 = widgets.Output()

        tab = widgets.Tab(children = [out1, out2, out3])
        tab.set_title(0, 'Orders')
        tab.set_title(1, 'Order items')
        tab.set_title(2, 'Return rates')
        display(tab)

        with out1:
            display(headers.style.format('{:,.0f}'))

        with out2:
            display(lines.style.format('{:,.0f}'))

        with out3:
            display(return_rates.style.format('{:,.2f}%'))

    # Spyder or IDE
    if log_path:
        log_file_path = os.path.join(log_path, 'eda.log')
        if os.path.isfile(log_file_path):
            os.remove(log_file_path)

        # building the logging handlers
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s:\n:%(message)s')
        file_handler = logging.FileHandler(log_file_path)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        # logging
        logger.info(tabulate.tabulate(headers, headers=headers.columns , tablefmt='psql'))
        logger.info(tabulate.tabulate(lines, headers=lines.columns, tablefmt='psql'))
        logger.info(tabulate.tabulate(return_rates, headers=return_rates.columns, tablefmt='psql'))

@my_perf_counter
def plot_dashboard(event, context):
    '''
    Plot different metrics.

    Inputs:
        context: dict = {
            df: pd.DataFrame >>> orders dataframe
            }
        event: dict = {
            date_start: datetime.datetime
            date_end: datetime.datetime
            period: str = 'M' >>> 'Y', 'Q', 'M'
            window_roll: int = 6 >>> months to count for the rolling mean
            figsize: tuple[int]= (20,6) >>> figure size
            dump_path_figures: Optional[str] >>> folder to save to Excel/csv
    '''
    # Assigning variables and run only key level error validations
    df = validate_input(context,'df', 'key')[1] # returns only the actual value or None
    date_start = validate_input(event,'date_start', 'key')[1]
    date_end = validate_input(event,'date_end', 'key')[1]
    period = validate_input(event,'period', 'key')[1]
    window_roll = validate_input(event,'window_roll', 'key')[1]
    dump_path_figures = validate_input(event,'dump_path_figures', 'key')[1]
    window_roll = validate_input(event, 'window_roll', 'key')[1]
    res, figsize, _ = validate_input(event, 'figsize', 'key')
    if res is False:
        raise Exception()
    if not figsize:
        figsize = (20,6)

    # Function to calculate the months difference between 2 dates
    def diff_month(d1: datetime.datetime, d2: datetime.datetime):
        return (d1.year - d2.year) * 12 + d1.month - d2.month

    # If the start date is less than 6 months ago then set to date end less 6 months
    if diff_month(date_end, date_start) < 6:
        date_start = date_end - datetime.timedelta(days=180)
        
    # colormap
    cmap = 'Accent'

    # Channels and timeseries along the same dates of the report
    kwargs1 = {
        'event': {
            'date_start': date_start
            , 'date_end': date_end
            , 'total_col': ['total_usd']
            , 'period': period
            , 'roll': window_roll
        },
        'context': {
            'df': df
        },
        'kind': 'timeseries'
    }
    tseries_ds, tseries_raw_ds = make_pivot(**kwargs1)
    channels_ds = tseries_raw_ds['channel'].unique()

    # Channels and timeseries from day one
    kwargs2 = {
        'event': {
            'date_start': datetime.datetime(2016,1,1,0,0,0) # Hard coded, since beginning
            , 'date_end': date_end
            , 'total_col': ['total_usd']
            , 'period': period
            , 'roll': window_roll
        },
        'context': {
            'df': df
        },
        'kind': 'timeseries'
    }
    tseries_df, tseries_raw_df = make_pivot(**kwargs2)
    channels_df = tseries_raw_df['channel'].unique()

    # Channels and timeseries by quantity rom day one
    kwargs3 = {
        'event': {
            'date_start': datetime.datetime(2016,1,1,0,0,0) # Hard coded, since beginning
            , 'date_end': date_end
            , 'total_col': ['total_quantity']
            , 'period': period
            , 'roll': window_roll
        },
        'context': {
            'df': df
        },
        'kind': 'timeseries'
    }
    tseries_df_qty, tseries_raw_df_qty = make_pivot(**kwargs3)
    
    # Channels and timeseries lagged 1 year backward
    kwargs4 = {
        'event': {
            'date_start': date_start - datetime.timedelta(days=365)
            , 'date_end': date_end - datetime.timedelta(days=365)
            , 'total_col': ['total_usd']
            , 'period': period
            , 'roll': window_roll
        },
        'context': {
            'df': df
        },
        'kind': 'timeseries'
    }
    tseries_d1y, tseries_raw_d1y = make_pivot(**kwargs4)

    # Dataframe with trend of unit price
    unit_price_orderline_df = pd.DataFrame(
            data = tseries_df[['Total']] / tseries_df_qty[['Total']])

    # time start string "MM-YYYY"
    time_start = str(date_start.month) + '-' + str(date_start.year)
    
    # Plot figures
    args = [tseries_ds,
            unit_price_orderline_df,
            channels_ds,
            time_start,
            window_roll,
            figsize,
            cmap]
    fig1 = plot_total_sales(*args)
    
    args = [tseries_ds,
            tseries_d1y,
            # tseries_df,
            channels_ds,
            # channels_df,
            time_start,
            figsize,
            cmap]
    fig2 = plot_cumulated_sales(*args)

    args = [tseries_df,
            channels_df,
            time_start,
            window_roll,
            figsize,
            cmap]
    fig3 = plot_rolling_mean(*args)

    args = [tseries_ds,
            channels_ds,
            time_start,
            figsize,
            cmap]
    fig4 = plot_differential_sales(*args)

    args = [df,
            date_start,
            date_end,
            time_start,
            figsize,
            'Blues']
    fig5 = plot_distributors(*args)

    args =[df,
           date_start,
           date_end,
           time_start,
           figsize,
           'Blues']
    fig6 = plot_sku_models(*args)

    # Dumping plots to figures
    if dump_path_figures:
        plots = [(fig1, 'plot_total_sales'),
                 (fig2, 'plot_cumulated_sales'),
                 (fig3, 'plot_rolling_mean'),
                 (fig4, 'plot_differential_sales'),
                 (fig5, 'plot_distributors'),
                 (fig6, 'plot_sku_models')]

        for f, p in plots:
            f.savefig(os.path.join(dump_path_figures, p + '.jpg'))

def plot_total_sales(tseries_ds,
                     unit_price_orderline_df,
                     channels_ds,
                     time_start,
                     window_roll,
                     figsize,
                     cmap):
    '''
    Plottig sales total and by sale channel.
    '''
    # Total sales from date start
    fig, axs = plt.subplots(1,2, figsize = figsize)
    tseries_ds.loc[:, channels_ds]\
        .plot(title=f'Monthly sales by sale channel from {time_start}'
              , ax = axs[0]
              , colormap = cmap)
    tseries_ds.loc[:, 'Total']\
        .plot(c = 'yellowgreen'
              , label = 'Total sales'
              , ax = axs[1])
    axs_1_twinx = axs[1].twinx()
    unit_price_orderline_df.loc[time_start:, 'Total']\
        .plot(c = 'black'
              , label = 'Unit Price - mean value'
              , ax = axs_1_twinx)
    axs[1].set_title(f'Monthly total sales from {time_start} vs unit price rolling mean ({window_roll} months)')
    axs[1].legend(loc='lower right')
    axs_1_twinx.legend(loc='upper left')
    for ax in axs:
        ax.set_xlabel("")
        ax.yaxis.set_major_formatter(ticker.StrMethodFormatter('{x:,.0f}'))
    fig.autofmt_xdate()
    plt.show()

    return fig

def plot_cumulated_sales(tseries_ds,
                        tseries_d1y,
                        # tseries_df,
                        channels_ds,
                        # channels_df,
                        time_start,
                        figsize,
                        cmap):
    '''
    Plottig cumulated sales total and by sale channel.
    '''
    # Cumulated sales from date start by sale channel
    fig, axs = plt.subplots(1,2, figsize = figsize)
    tseries_ds.loc[:,[c + '_cum' for c in channels_ds]]\
        .plot(title=f'Monthly Sales cumulated by sale channel from {time_start}'
              , ax = axs[0]
              , colormap = cmap)
    axs[0].legend(channels_ds)
    
    # Cumulated total sales as compared to the previous period
    axs[1].plot(tseries_ds.index,
                tseries_ds.loc[:, 'Total_cum'].values,
                '-',
                c = 'black',
                label = 'Current period')
    axs[1].plot(tseries_ds.index,
                tseries_d1y.loc[:, 'Total_cum'].values,
                '--',
                c = 'yellowgreen',
                label = 'Previous year')
    axs[1].set_title(f'Monthly Sales cumulated from {time_start} vs previous year')
    axs[1].legend()
    axs[1].xaxis.set_major_formatter(mpl_dates.DateFormatter('%b %Y'))
    for ax in axs:
        ax.set_xlabel("")
        ax.yaxis.set_major_formatter(ticker.StrMethodFormatter('{x:,.0f}'))
    fig.autofmt_xdate()
    plt.show()

    return fig

def plot_differential_sales(tseries_ds,
                            channels_ds,
                            time_start,
                            figsize,
                            cmap):
    '''
    Plottig sales difference total and by sale channel.
    '''
    
    # For the add_value_labels function
    min_label = tseries_ds.loc[:,'Total_diff'].min()
    
    # Monthly sales difference from date start by sale channel
    fig, axs = plt.subplots(1,2, figsize = figsize)
    tseries_ds.loc[:,[c + '_diff' for c in channels_ds]]\
        .plot(
            title=f'Monthly Sales difference by sale channel from {time_start}'
            , ax = axs[0]
            , kind = 'bar'
            , colormap = cmap)
    axs[0].legend(channels_ds)

    # Monthly total sales difference from date start
    tseries_ds.loc[:,'Total_diff']\
        .plot(
            title=f'Monthly total sales difference from {time_start}'
            , ax = axs[1]
            , kind = 'bar'
            , colormap = cmap)
    for ax in axs:
        ax.set_xlabel("")
        ax.set_xticklabels(tseries_ds.index.month_name())
        ax.set_yticklabels([])
        add_value_labels(ax, symbol = '$',min_label = min_label)
    fig.autofmt_xdate()
    plt.show()

    return fig

def plot_rolling_mean(tseries_df,
                    channels_df,
                    time_start,
                    window_roll,
                    figsize,
                    cmap):
    '''
    Plottig sales rolling mean total and by sale channel.
    '''
    # Monthly rolling mean from from date start  by sale channel
    fig, axs = plt.subplots(1,2, figsize = figsize)
    tseries_df\
    .loc[time_start:,[c + '_roll' for c in channels_df]]\
        .plot(
            title=f'Monthly sales rolling mean ({window_roll} months) by sale channel from {time_start}'
              ,  ax = axs[0]
              , colormap = 'tab20b')
    axs[0].legend(channels_df)

    # Monthly total rolling mean from from date start 
    ax = tseries_df\
    .loc[time_start:,'Total_roll']\
        .plot(title=f'Monthly total sales rolling mean ({window_roll} months) from {time_start}'
              , ax = axs[1]
              , colormap = 'tab20b')
    for ax in axs:
        ax.set_xlabel("")
        ax.set_xticklabels(tseries_df.loc[time_start:, :].index.month_name())
        ax.yaxis.set_major_formatter(ticker.StrMethodFormatter('{x:,.0f}'))
    fig.autofmt_xdate()
    plt.show()
    
    return fig

def plot_distributors(df,
                    date_start,
                    date_end,
                    time_start,
                    figsize,
                     cmap):
    '''
    Plotting top distributors.
    '''
    # Show top distributors from date start
    context = {'df': df}
    event = {
          'date_start': date_start
        , 'date_end': date_end
        , 'period': 'M'
        , 'drill_down': ['billing_company']
        }
    top_distributors, _ = make_pivot(event, context, 'orders')
    
    fig = plt.figure(figsize = figsize)
    distributors_cmap = plt.cm.get_cmap(cmap)
    top_distributors = top_distributors.loc[top_distributors.index != 'Total', 'Total']\
        .nlargest(10)
    colors = distributors_cmap([x / max(top_distributors) for x in top_distributors])
    ax = top_distributors.plot(kind='bar', color = colors)
    ax.set_title(f'Top distributors from {time_start}')
    add_value_labels(ax, symbol = '$')
    ax.set_xlabel('')
    ax.set_xticklabels(ax.get_xticklabels(), rotation = 45, horizontalalignment = 'right')
    ax.set_yticklabels([0])
    plt.show()
    
    return fig

def plot_sku_models(df,
                   date_start,
                   date_end,
                   time_start,
                   figsize,
                   cmap):
    '''
    Plotting top skus and models for distributors.
    '''
    # Show top skus from date start
    context = {'df': df}
    event = {
          'date_start': date_start
        , 'date_end': date_end
        , 'period': 'M'
        , 'drill_down': ['lineitem_sku']
        }
    top_sku, _ = make_pivot(event, context, 'orderlines')
    fig, axs = plt.subplots(1,2, figsize = figsize)
    sku_cmap = plt.cm.get_cmap(cmap)
    top_sku = top_sku.loc[top_sku.index != 'Total','Total']\
        .nlargest(10)
    colors = sku_cmap([x / max(top_sku) for x in top_sku])
    top_sku.plot(kind='bar', ax = axs[0], color = colors)
    axs[0].set_title(f'Top 10 skus for distributors from {time_start}')
    axs[0].set_xlabel('')
    axs[0].set_xticklabels(axs[0].get_xticklabels(), rotation = 45)
    axs[0].set_yticklabels([0])
    add_value_labels(axs[0])
    
    # Show top models from date start
    context = {'df': df}
    event = {
          'date_start': date_start
        , 'date_end': date_end
        , 'period': 'M'
        , 'drill_down': ['lineitem_model']
        }
    top_models, _ = make_pivot(event, context, 'orderlines')
    models_cmap = plt.cm.get_cmap(cmap)
    top_models = top_models.loc[top_models.index != 'Total', 'Total']\
        .nlargest(10)
    colors = models_cmap([x / max(top_models) for x in top_models])
    top_models.plot(kind='bar', ax = axs[1], color = colors)
    axs[1].set_title(f'Best models for distributors from {time_start}')
    axs[1].set_xlabel('')
    axs[1].set_xticklabels(axs[1].get_xticklabels(), rotation = 45)
    axs[1].set_yticklabels([0])
    add_value_labels(axs[1])
    plt.show()

    return fig
        
def add_value_labels(ax, spacing=5,symbol='',min_label=0):
    """
    Helper function which I re-used. Add labels to the end of each bar in a bar chart.
    """

    # For each bar: Place a label
    for rect in ax.patches:
        # Get X and Y placement of label from rect.
        y_value = rect.get_height()
        x_value = rect.get_x() + rect.get_width() / 2

        # Number of points between bar and label. Change to your liking.
        space = spacing
        # Vertical alignment for positive values
        va = 'bottom'

        # If value of bar is negative: Place label below bar
        if y_value < 0:
            # Invert space to place label below
            space *= -1
            # Vertically align label at top
            va = 'top'

        # Use Y value as label and format number with one decimal place
        label = "{}{:,.0f}".format(symbol,y_value)

        #If value is low don't show the bar
        if y_value < min_label or y_value == 0:
            label = ''

        # Create annotation
        ax.annotate(
            label,                      # Use `label` as label
            (x_value, y_value),         # Place label at end of the bar
            xytext=(0, space),          # Vertically shift label by `space`
            textcoords="offset points", # Interpret `xytext` as offset in points
            ha='center',                # Horizontally center label
            va=va)                      # Vertically align label differently for
                                        # positive and negative values.
            

def randomizer(df, *args):
    '''
    It makes sensitive data random to demo a live case.
    Inputs:
        df: pd.DataFrame >>> orders dataframe
        *args: str >>> list of string fields to scramble
    Return:
        df: pd.DataFrame >>> orders dataframe scrambled
    ''' 
    # keep the same randomization
    random.seed(40)
    
    # update function to scramble the strings
    def update_series(arr_of_unique_values):
        new_arr = [(s, ''.join(random.sample(s,len(s)))) for s in [str(b) for b in arr_of_unique_values] if s != 'nan']
        return pd.DataFrame(new_arr, columns=['original', 'transformed'])

    # scramble all object type columns
    for col in args:
        if df[col] .dtypes == 'object':
            df_for_update = update_series(df.loc[df.distributor == True, col].unique())
            df[col] = df.join(df_for_update.set_index('original'), on=col, how='left').loc[:, 'transformed']

    # it makes float columns (except total_quantity) by a random value from a uniform distribution [0,1]
    for col in args:
        if col != 'total_quantity' and df[col] .dtypes == 'float64':
            df[col] = pd.Series(np.round(df[col] * np.random.rand(len(df[col]), ), 2))

    return df