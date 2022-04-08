'''
ETL utility module.
'''
# Python
import os
import re

# Thrid part
from forex_python.converter import CurrencyRates
import numpy as np
import pandas as pd
import sqlite3

# support
from support import datasetup
from support.utils import validate_input
from support.orders import Orders

def e_t_l (context, event):
    '''
    Runs ETL to load and transform order information according to logic.
    It will update the orders_pipeline.db SQLite database orders table with new order_ids.
    It will re-generate or not the corresponding csv file depending by re_generate_orders_csv.

    Inputs:
        context: dict
            data: str >>> folder with all the data pipelines
            orders_filepath: str >>> the filepath to the orders data
            orders_file: str >>> the name of the file with orders data
            rates_file: str >>> the name of the file with currency rates
            to_replace_company_names: list[str] = None list of company names to standardize
            value_company_names: list[str] = None the standardized companies names
            stores: str >>> the Shopify stores separated by a "|"
            channels: list[str] = None >>> the sale channels
        event: dict
            re_generate_rates: bool >>> generate a new currency rates file
            re_generate_orders_csv: bool >>> generate a new orders_file or just append to the SQL table
    Return:
        df: pd.DataFrame >>> the orders dataframe with headers and lines
    '''
    # Assign variables and run key validations
    try:
        data = validate_input(context, 'data', 'key')[1]
        orders_filepath = validate_input(context, 'orders_filepath', 'key')[1]
        orders_file = validate_input(context, 'orders_file', 'key')[1]
        rates_file = validate_input(context, 'rates_file', 'key')[1]
        to_replace_company_names = validate_input(context, 'to_replace_company_names', 'key')[1]
        value_company_names = validate_input(context, 'value_company_names', 'key')[1]
        stores = validate_input(context, 'stores', 'key')[1]
        channels = validate_input(context, 'channels', 'key')[1]
        re_generate_rates = validate_input(event, 're_generate_rates', 'key')[1]
        re_generate_orders_csv = validate_input(event, 're_generate_orders_csv', 'key')[1]
    except Exception as e:
        print(e)
        return None


    # Assign defaults for None inputs
    if to_replace_company_names is None:
        to_replace_company_names = []
    if value_company_names is None:
        value_company_names = []
    if channels is None:
        channels = []
    if stores is None:
        stores = datasetup.stores
    if re_generate_rates is None:
        re_generate_rates = False
    if re_generate_orders_csv is None:
        re_generate_orders_csv = False

    # Type verification for inputs which passed through key validation
    try:
        assert isinstance(to_replace_company_names, list), 'to_replace_company_names is not a list'
        assert isinstance(value_company_names, list), 'value_company_names is not a list'
        assert isinstance(channels, list), 'channels is not a list'
        assert isinstance(re_generate_rates, bool), 're-generate rates must be bool'
    except AssertionError as a:
        print(a)
        return None

    # load files into a dataframe
    df = pd.DataFrame()

    try:
        files_csv = [f for f in os.listdir(os.path.join(data, orders_filepath)) if re.search('.csv', f)]
        for f in files_csv:
            print(f)
            df1 = pd.read_csv(os.path.join(data, orders_filepath, f))
            df1['store'] = re.search(stores, f).group() # on all order lines 
            df = pd.concat([df, df1], axis=0)
    except FileNotFoundError:
        print('File not found')
        return None
    except Exception as e:
        print(e)
        return None
    else:
        del df1 # delete temporary dataframe
        df.reset_index(drop=True,inplace=True) # reset df index
        df.columns = [c.lower() for c in df.columns] # make df column names lower case
        df.columns = [re.sub(' ', '_', c) for c in df.columns] # replace space with _

    # Enforce unique order identifier
    df['order_id'] = df['name'] + '-' + df['store'] # on all order lines

    # Drop cancelled and refunded orders (headers and lines)
    df.drop(df.loc[
        (
            df['order_id'].isin(df.loc[df['cancelled_at'].notna(),'order_id']) | 
            df['order_id'].isin(df.loc[df['financial_status'] == 'refunded','order_id'])
        )].index, axis=0, inplace=True)
    df.reset_index(drop=True,inplace=True) # reset df index

    # Remove un-used columns
    df.drop([
        'accepts_marketing',
        'cancelled_at',
        'device_id',
        'discount_code',
        'duties',
        'email',
        'employee',
        'id',
        'lineitem_compare_at_price',
        'lineitem_requires_shipping',
        'lineitem_taxable',
        'location',
        'name',
        'next_payment_due_at',
        'note_attributes',
        'notes',
        'outstanding_balance',
        'payment_reference',
        'payment_terms_name',
        'phone',
        'receipt_number',
        'refunded_amount',
        'risk_level',
        'source',
        'tax_1_name',
        'tax_1_value',
        'tax_2_name',
        'tax_2_value',
        'tax_3_name',
        'tax_3_value',
        'tax_4_name',
        'tax_4_value',
        'tax_5_name',
        'tax_5_value',
        'taxes',
        'vendor'
        ],axis=1,inplace=True)

    # Make order header flag
    df['is_order_header'] = pd.Series([True if pd.notna(c) else False for c in df.currency]) # Currency is NaN in line

    # Make order lines model, amount, unit price, json columns
    df['lineitem_model'] = df['lineitem_name'].str.split('-',n=1).str[0]
    df['lineitem_amount'] = df['lineitem_quantity'] * df['lineitem_price'] - df['lineitem_discount']
    df['lineitem_unit_price'] = df['lineitem_amount'] / df['lineitem_quantity']
    df['lineitems_json'] = np.nan

    # Make region, not for the order lines
    df['region'] = df.billing_country.apply(make_region)

    # Make distributor, not for the order lines
    df['distributor'] = df.apply(lambda x: make_distributor(x['is_order_header'], x['tags']), axis=1)

    # Update billing company distributors in which the company is on the billing name, not for the order lines
    df['billing_company']=\
        df.apply(lambda x: update_billing_company_for_distributors(x['billing_company'],
                                                                   x['billing_name'],
                                                                   x['distributor']), axis=1)
    # Standardize distributors' names
    df['billing_company'].replace(to_replace_company_names,value_company_names,inplace=True,regex=True)
    # Strips trailing blanks
    df['billing_company'] = df['billing_company'].str.strip()

    # Make source, not for the order lines
    df['source'] = df.apply(lambda x: make_source(x['billing_company'], x['billing_name']), axis=1)

    # Make order header and lines dates
    df[['order_date', 'lineitem_date']] =    pd.DataFrame(
    [make_order_dates(x[0], x[1], x[2]) for x in\
    df[['created_at', 'fulfilled_at', 'is_order_header']].values])

    # Make the currency rates
    df_rates = make_currency_rates(df, os.path.join(data, rates_file), re_generate_rates)
    # join the rates dataframe with the df dataframe
    df = df.merge(df_rates,on=['order_date','currency'],how='left')

    # Make total USD, not for the order lines
    df['total_usd'] = df.apply(lambda x: make_total_usd(x['currency'], x['currency_rate'], x['total']), axis=1)

    # Make total quantity, not for the order lines
    liq = df.groupby('order_id')['lineitem_quantity'].sum()
    df['total_quantity'] = df.apply(lambda x: make_total_qty(x['order_id'], x['is_order_header'], liq), axis = 1)
    del liq

    # Make sale channels, not for the order lines
    df['channel'] = df.apply(lambda x: make_channels(channels, x['tags'], x['distributor']), axis = 1)

    # Sort columns
    df = df.reindex([
        'order_id',
        'order_date',
        'created_at',
        'financial_status',
        'paid_at',
        'payment_method',
        'fulfillment_status',
        'fulfilled_at',
        'distributor',
        'billing_name',
        'billing_company',
        'billing_address1',
        'billing_address2',
        'billing_street',
        'billing_city',
        'billing_province',
        'billing_province_name',
        'billing_country',
        'billing_zip',
        'billing_phone',
        'currency',
        'currency_rate',
        'discount_amount',
        'shipping',
        'subtotal',
        'total',
        'total_usd',
        'total_quantity',
        'is_order_header',
        'lineitem_date',
        'lineitem_sku',
        'lineitem_name',
        'lineitem_model',
        'lineitem_quantity',
        'lineitem_price',
        'lineitem_discount',
        'lineitem_amount',
        'lineitem_unit_price',
        'lineitem_fulfillment_status',
        'shipping_name',
        'shipping_company',
        'shipping_address1',
        'shipping_address2',
        'shipping_street',
        'shipping_city',
        'shipping_province',
        'shipping_province_name',
        'shipping_country',
        'shipping_zip',
        'shipping_phone',
        'shipping_method',
        'store',
        'channel',
        'region',
        'source',
        'tags',
        'lineitems_json'
        ], axis=1)
    
    # Save the orders table to the orders_pipeline.db SQLite
    import_new_orders(df, data)
        
    # Save to csv and return dataframe
    if re_generate_orders_csv:
        df.to_csv(os.path.join(data, orders_file),index=False)

    return df

# Helper functions

def import_new_orders(df, data):
    '''
    Imports into the orders_pipeline.db (SQLite) orders from the input dataframe 
    (=from the csv files which are in the working area).
    Inputs:
        df: pd.DataFrame >>> the orders dataframe with headers and lines
        data: str >>> folder with all the data pipelines
    '''
    try:
        # opens the database
        conn = sqlite3.connect(os.path.join(data, 'orders_pipeline.db'))
        with conn:
            c = conn.cursor()
            # create the table if it doesn't exist
            sql_stm = 'CREATE TABLE IF NOT EXISTS orders {}'.format(str(tuple(df)))
            c.execute(sql_stm)
            # deleting from the database orders which remained in the working area (csv files)
            order_ids = df['order_id'].unique()
            sql_stm = 'DELETE FROM orders WHERE order_id IN {}'.format(tuple(order_ids))
            c.execute(sql_stm)
            conn.commit()
            # selecting orders which are already in the staging area and don't need to be imported again
            sql_stm = 'SELECT DISTINCT order_id FROM orders'
            order_ids = [oid[0] for oid in c.execute(sql_stm)]
            df_update = df[~df.order_id.isin(order_ids)]
            # add the lieitems column
            count_added_rows = df_update.shape[0]
            # inserting orders from the working area
            for row in df_update.values:
                sql_stm = 'INSERT INTO orders VALUES ' + str(tuple(row)).replace("nan", "Null")
                c.execute(sql_stm)
                conn.commit()
            # updating lineitems
            my_orders = Orders(df_update)
            for order_id in df_update['order_id'].unique():
                try:
                    dfl = my_orders.get_order_lines(order_id)
                    # building a dictionary for each order header in which the lineitem attributes (e.g. sku) are lists
                    ol = {'lineitems':{}}
                    for col in dfl.columns:
                        l = []
                        for i in dfl[col]:
                            l.append(str(i))
                        ol['lineitems'][col] = l
                    sql_stm = "UPDATE orders SET lineitems_json = \"{}\" WHERE order_id = '{}' AND is_order_header = 1".format(ol, order_id)
                    c.execute(sql_stm)
                    conn.commit()
                except sqlite3.Error as e:
                    print(order_id, ': ', e)
                    continue
            # checking the number of orders in the database
            sql_stm = 'SELECT COUNT(*) FROM orders'
            c.execute(sql_stm)
            count_total_rows = c.fetchone()
    except sqlite3.Error as e:
        print(e)
    else:
        print(f'{count_added_rows} new records imported in orders db file.')
        print(f'{count_total_rows} total records.')

def make_region(bc):
    '''
    Input:
        bc: str >>> billing country
    Return:
        region: Optional[str]
    '''
    if bc in ('US', 'CA'):
        region = 'NAM' # North America
    elif bc == bc: # Test is r is not NaN
        region = 'INT' # International
    else:
        region = np.nan # Order lines or undefined

    return region

def make_distributor(oh, t):
    '''
    Inputs:
        oh: bool >>> is order header
        t: str >>>> tags
    Return:
        distributor: Optional[bool]
    '''
    if re.search('[Dd]istributor', str(t)) and oh: # Header with distributor
        distributor = True
    elif oh:                  # Header without distributor
        distributor = False
    else:                     # Order line
        distributor = np.nan

    return distributor

def update_billing_company_for_distributors(bc, bn, d):
    '''
    Inputs:
        bc: str >>> billing company
        bn: str >>> billing name
        d: Optional[bool] >>> distributor flag
    Return:
        bc: str
    '''
    if pd.notna(d) and pd.isna(bc):
        bc = str(bn)
    elif pd.notna(d) and pd.notna(bc):
        bc = str(bc)

    return bc

def make_source(bc, bn):
    '''
    Inputs:
        bc: str >>> billing company
        bn: str >>> billing name
    Return:
        source: Optional[str]
    '''
    if pd.notna(bc): # B2B
        source = 'B2B'
    elif pd.isna(bc) and pd.notna(bn): # Direct
        source = 'DIR'
    else:
        source = np.nan # Order lines

    return source

def make_order_dates(ca, fa, oh):
    '''
    Inputs:
        fa: object >>> fulfillment date
        ca: object >>> created date
        oh: bool >>> is order header
    Return:
        order_date, order_line_date Optional[dict]
    '''
    if pd.notna(fa): # If the fulfilment date exists in the order header
        order_date = re.split('\s', fa)[0] # Save the format YYYY-MM-DD
    elif oh: # If the fulfilment date is null use the created date
        order_date = re.split('\s', ca)[0] # Save the format YYYY-MM-DD
    else:
        order_date = np.nan # Order lines

    order_line_date = re.split('\s', ca)[0] # Order lines have only created date

    return {'ohd': order_date, 'old': order_line_date}

def make_total_usd(c, r, t):
    '''
    Inputs:
        c: str >>> currency
        r: float >>> exchange rate
        t: float >>> total in non-USD currency
    Return:
        total_usd: Optional[float]
    '''
    if pd.notna(c) and c != 'USD': # EUR or GBP
        total_usd = t * r
    elif pd.notna(c) and c == 'USD': # USD
        total_usd = t
    else:
        total_usd = np.nan # Order lines

    return total_usd

def make_total_qty(oid, oh, liq):
    '''
    Inputs:
        oid: str >>> order_id
        oh: bool >>> is order header
        liq: pd.Series >>> line item quantity by order id
    Return:
        total_qty: Optional[int] >>> total quantity by order id
    '''

    # Apply total quantity to order header, not to order lines
    total_qty = int(liq[oid]) if oh else np.nan

    return total_qty

def make_channels(ch, t, d):
    '''
    Inputs:
        ch: list[str] >>> channels
        t: list[str] >>> tags
        d: Optional[bool] >>> distributor flag
        oh: bool >>> is order header
    Return:
        channel: Optional[str] sale channel
    '''
    for c in ch:
        if d and re.search(str(c).lower(), str(t).lower()): # A channel match in order header
            return c # return and exit the function
    if d is True:  # Headquarter sale to distributor
        return 'HQ' # return and exit the function

    return np.nan # Any other case: order line or sale not to distributor

def make_currency_rates(
    df
    , rates_filepath
    , re_generate_rates=False
    ):
    '''
    Update the currency rates.

    Inputs:
        df: pd.DataFrame >>> the orders dataframe
        rates_filepath: str >>> the path of the currency rates csv file
        re_generate_rates: bool >>> re-generate the entire file (True) or update (False)
    Return:
        df_rates: pd.DataFrame >>> the currency rates dataframe for non-USD orders
    '''

    # To update take from the csv file and add rate for dates after the last update
    if re_generate_rates is False:
        try:
            df_rates = pd.read_csv(rates_filepath,usecols=['order_date','currency_rate','currency'])
        except FileNotFoundError:
            print('File not found')
            return None
        except Exception as e:
            print(e)
            return None
        else:
            date_start = df_rates['order_date'].max() # Most recent exchange rate
    else:
        date_start = '2016-01-01' #to re-generate start from beginning
        
    # Dataframe with order dates after the most recent exchange rate
    date_currency_index = df.loc[
        (df['currency'].notna()) # The currency is only in the order header
        & (df['currency'] != 'USD') # Get exchange rates for currencies other than USD
        & (df['order_date'] > date_start) # Get exchange rates for dates which are not in the file
        ,['order_date','currency']
        ]

    # Make order_date into datetime format for use with the currency API
    date_currency_index['order_date'] = pd.to_datetime(date_currency_index['order_date'])

    # Drop duplicates (save only one day and currency touple for each order)
    date_currency_index = date_currency_index.drop_duplicates()\
    .sort_values('order_date', ascending=False)\
    .reset_index(drop=True)

    # Run the currency API
    try:
        cr = CurrencyRates()
    except Exception as e:
        print(e)
        return None

    # Make a list with all the new exchange rates
    rate_index = [cr.convert(c, 'USD', 1, d) for d,c in date_currency_index.values]

    # Put together the dataframe with updates
    df_rates_update = pd.DataFrame({
        'order_date':date_currency_index['order_date']
        ,'currency':date_currency_index['currency']
        ,'currency_rate':rate_index
        })

    # Format the order_date back to YYYY-MM-DD
    df_rates_update['order_date'] = df_rates_update['order_date'].dt.strftime('%Y-%m-%d')

    # Merge the update from date_start
    if re_generate_rates is False:
        df_rates = pd.concat([df_rates,df_rates_update],axis=0, ignore_index=True) # Updating df_rates
    else:
        df_rates = df_rates_update # Re-generating df_rates

    # Sort and reset index of df_rates
    df_rates.sort_values('order_date',ascending=False,inplace=True)
    df_rates.reset_index(drop=True)

    # Save the rates to csv for future updates
    df_rates.to_csv(rates_filepath,columns=['order_date','currency_rate','currency'])

    return df_rates
