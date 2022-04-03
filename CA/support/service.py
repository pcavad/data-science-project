'''
Service utility functions.
'''
# Python
import logging
import os

# Thrid part
import numpy as np
import pandas as pd
pd.set_option('display.max_columns', 15)
pd.set_option('display.max_rows', None)
pd.options.display.float_format = '{:,.2f}'.format
import re
import tabulate
import warnings
warnings.filterwarnings('ignore', module='pandas')

def make_service(
    data: str = 'data'
    , returns_filepath: str = 'returns'
    , models_file: str = 'models.xlsx'
    , to_replace_descriptions: list[str] = None
    , value_classified: list[str] = None
    , log_path: str = 'reports'
    ):
    '''
    Reads the RMA files and returns the service dataframe.

    Inpuuts:
        data: str >>> folder with orders data
        returns_filepath: str >>> path to the authorized returns files
        models_file: str >>> Excel file with metadata
        to_replace: list[str] texts to trap for auto classification
        value: list[str] classified defects
        log_path: Optional[str]
                >>> folder to save logging with output (for IDE which don't show widgets)
    Return:
        df: pd.DataFrame >>> history of the service cases
    '''
    # load files into a dataframe
    df = pd.DataFrame()

    # exception class
    class BreakIt(BaseException):
        '''
        Use to handle exit loop and function.
        '''
        pass
    exit_function = False

    try:
        for d in os.listdir(os.path.join(data, returns_filepath)):
            if os.path.isdir(os.path.join(data, returns_filepath, d)): # For each directory
                for f in os.listdir(os.path.join(data, returns_filepath, d)):
                    if bool(re.search('.*.xlsx', f)): # If it is a file Excel
                        df1 = pd.read_excel(os.path.join(data, returns_filepath, d, f)\
                                            , header=None, usecols='A:D',
                                            names=['sku', 'name', 'serial_number', 'defect'])
                        df1['distributor'] = d # Update distributor name

                        df1['file'] = f # File name
                        if bool(re.search('^[_]', f)): # If the RMA wasn't returned yet search for a '_' at beginning of the file name
                            df1['processed'] = False
                        else:
                            df1['processed'] = True
                        if bool(re.search('20[0-9]{2}Q[1-4]', f)):
                            df1['return_period'] =\
                                re.search('20[0-9]{2}Q[1-4]', f).group() # Return period
                        else:
                            raise BreakIt
                        df = pd.concat([df, df1], axis=0)
                        del df1 # delete temporary dataframe
                        df.reset_index(drop=True,inplace=True) # reset df index
    except BreakIt:
        exit_function = True
        print(f, 'Wrong date code')
    except Exception as e:
        print(f, e)

    if exit_function:
        return None

    # Make standardized model and price index (for future processing)
    df_sku = pd.read_excel(os.path.join(data,models_file)) # load metadata with sku/models
    df_sku.set_index('sku', inplace=True)
    df['model'] = df.join(df_sku, on='sku', how='left').loc[:,'model']
    df.drop('name', axis=1, inplace=True) # The name is replaced by model (standardised)

    # Make standardized defect description
    df['defect_classified'] =\
        df.defect.apply(lambda x: str(x.lower()))\
            .replace(to_replace_descriptions,value_classified, regex=True)

    # Make serial number upper case (for joins)
    df['serial_number'] = df.serial_number.str.upper()

    # Search for tags in the description of the defect (e.g. [repaired]) and put it in a 'tags' column
    def tags(t):
        if bool(re.search('\[.+\]', t)):
            return re.search('\[.+\]', t).group()[1:-1] # Remove []
        return np.nan
    
    df['tags'] = df.defect.apply(tags)

    # Sort columns
    df = df.reindex(
        columns=[
        'sku'
        , 'model'
        , 'serial_number'
        , 'return_period'
        , 'defect'
        , 'defect_classified'
        , 'distributor'
        , 'processed'
        , 'tags'
        , 'file'
        ]
        )

    # Run checks
    print('### new ###\n')
    display(df[df.processed == False].sort_values('return_period')) # Check only the new records

    print('### sku mismatch ###\n')
    display(df.loc[(df.model.isna()) & (df.processed is False)])

    def bad_serial(sn):
        '''
        Checks the serial number for wrong format
        Arguments:
            sn (str): serial number
        Return:
            bool (True = serial is bad)
        '''
        if pd.notna(sn):
            sn = str(sn)
            if bool(re.match('[0-9]{4}X*\s\w[0-9]{5}', sn))\
                | bool(re.match('[0-9]{2}S[0O][0-9]{5}', sn)):
                return False # Serial is ok
            return True # Serial is bad
        return True # Serial is Null

    print('### bad serial number ###\n')  # Check only the new records
    display(df.loc[(df.serial_number.apply(bad_serial)) & (df.processed is False)])

    print('### duplicated serial number ###\n')
    display(df.loc[df.serial_number.duplicated(keep=False)]\
            .sort_values('serial_number')) # Check all records

    print('### not classified ###\n')
    display(df.loc[(df.defect_classified.isna())
                   & (df.processed == False)]) # Check only the new records

    def classification_length(dc: str):
        '''
        Checks if the auto classification failed.
    
        Inputs:
            dc: str >>> description of the defect
        Return:
            bool >>> True = bad calssification
        '''
        if pd.notna(dc):
            if len(str(dc)) > 10:
                return True
            return False

    print('### bad classification ###\n')
    display(df[(df.defect_classified.apply(classification_length))
               & (df.processed == False)]) # Check only the new records

    # save to csv
    df.to_csv(os.path.join(data, returns_filepath, 'service.csv'))

    if log_path:
        log_file_path = os.path.join(log_path, 'service.log')
        if os.path.isfile(log_file_path):
            os.remove(log_file_path)
        logging.basicConfig(filename = log_file_path, level = logging.INFO)
        logging.info('===== new =====\n')
        logging.info(tabulate.tabulate(df[df.processed == False].sort_values('return_period'),
                          headers=df.columns, tablefmt='psql'))
        logging.info('===== sku mismatch =====\n')
        logging.info(tabulate.tabulate(df.loc[(df.model.isna()) & (df.processed == False)],
                     headers=df.columns, tablefmt='psql'))
        logging.info('===== bad serial number =====\n')
        logging.info(tabulate.tabulate(
            df.loc[(df.serial_number.apply(bad_serial)) & (df.processed == False)]
                      , headers=df.columns, tablefmt='psql'))
        logging.info('===== duplicated serial number =====\n')
        logging.info(tabulate.tabulate(
            df.loc[df.serial_number.duplicated(keep=False)].sort_values('serial_number')
                      , headers=df.columns, tablefmt='psql'))
        logging.info('===== not classified =====\n')
        logging.info(tabulate.tabulate(
            df.loc[(df.defect_classified.isna()) & (df.processed == False)]
                      , headers=df.columns, tablefmt='psql'))
        logging.info('===== bad classification =====\n')
        logging.info(tabulate.tabulate(
            df[(df.defect_classified.apply(classification_length)) & (df.processed == False)]
                      , headers=df.columns, tablefmt='psql'))

    return df
