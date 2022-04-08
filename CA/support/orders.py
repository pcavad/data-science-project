#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 21 17:22:06 2022

@author: paolocavadini
"""

import datetime
import json
import os
import pandas as pd

class Orders(pd.DataFrame):
    '''
    The Orders data type.
    '''
    def __init__(self
            , data=None, index=None, columns=None, dtype=None, copy=None
            , my_header_mask: list[str] = None
            , my_columns_mask: list[str] = None):

        super().__init__(data, index, columns, dtype, copy)
        
        # sets columns to defaults if None
        if my_header_mask:
            self.header_mask = my_header_mask
        else:
            self.header_mask = Orders.header_mask
            
        if my_columns_mask:
            self.columns_mask = my_columns_mask
        else:
            self.columns_mask = Orders.columns_mask
    
    header_mask = [
        'order_id'
        , 'billing_company'
        , 'billing_name'
        , 'fulfillment_status'
        , 'order_date'
        , 'created_at'
        , 'paid_at'
        , 'fulfilled_at'
        , 'currency_rate'
        , 'currency'
        , 'subtotal'
        , 'shipping'
        , 'total'
        , 'total_usd'
        , 'total_quantity'
        , 'source'
        , 'region'
        , 'channel'
        , 'distributor'
        ]

    columns_mask = [
        'order_id'
        , 'lineitem_date'
        , 'lineitem_sku'
        , 'lineitem_name'
        , 'lineitem_model'
        , 'lineitem_quantity'
        , 'lineitem_unit_price'
        , 'lineitem_amount'
        ]
    
    def get_order_header(self, order_id: str):
        '''
        The order header for a specific order_id.
        '''
        return self.loc[(self['order_id'] == order_id) 
                        & (self['is_order_header'] == True), self.header_mask]

    def get_order_lines(self, order_id: str):
        '''
        The order lines for a specific order_id.
        '''
        return self.loc[self['order_id'] == order_id, self.columns_mask]
    
    def get_order_json(self, order_id: str, reports: str = 'reports'):
        '''
        Dumps the order header and line in json format.
        '''
        dfh = self.loc[(self['order_id'] == order_id) 
                        & (self['is_order_header'] == True), self.header_mask]
        oh = {}
        for col in dfh.columns:
            oh[col] = str(dfh[col].item())

        dfl = self.loc[self['order_id'] == order_id, self.columns_mask]
        ol = {}
        for col in dfl.columns:
            l = []
            for i in dfl[col]:
                l.append(str(i))
            ol[col] = l
            
        order_dict = {}
        order_dict['header'] = oh
        order_dict['lineitems'] = ol
        
        with open(os.path.join(reports, order_id + '.json'), 'w') as f:
            json.dump(order_dict,f)
            
        return order_dict
    
    def get_orders(self
                  , date_start: datetime.datetime
                  , date_end: datetime.datetime
                  , df_filter: pd.Series = None):
        '''
        Get orders by date.
        '''
        if df_filter is None:
            filt_ad = True
        elif df_filter is True:
            filt_ad = True
        else:
            filt_ad = df_filter

        return self.loc[
            (self.is_order_header == True)
            & (self.distributor == True)
            & (date_start <= self.order_date)
            & (date_end >= self.order_date)
            & filt_ad
            , self.header_mask]
    
    def get_orderlines(self
                  , date_start: datetime.datetime
                  , date_end: datetime.datetime
                  , df_filter: pd.Series = None):
        '''
        Get orderlines by date.
        '''
        if df_filter is None:
            filt_ad = True
        elif df_filter is True:
            filt_ad = True
        else:
            filt_ad = self['order_id'].isin(self.loc[df_filter, 'order_id'])
            
        return self.loc[
            (self['order_id'].isin(
                self.loc[self.distributor == True, 'order_id']))
            & (date_start <= self.lineitem_date)
            & (date_end >= self.lineitem_date)
            & filt_ad
            , self.columns_mask]
    
    def get_timeseries(self
                  , date_start: datetime.datetime
                  , date_end: datetime.datetime
                  , total_column: list[str] = ['total_usd']):
        '''
        Get timeseries by date.
        '''
        return self.loc[
            (self.is_order_header == True)
            & (self.distributor == True)
            & (date_start <= self.order_date)
            & (date_end >= self.order_date)
            , ['order_date', 'channel'] + total_column]
    
    @classmethod
    def set_header_mask(cls, my_header: list[str]):
        '''
        Set a new list of header columns.
        '''
        cls.header_mask = my_header
    
    @classmethod
    def set_lineitems_mask(cls, my_lineitems: list[str]):
        '''
        Set a new list of lineitems columns.
        '''
        cls.columns_mask = my_lineitems
        
    def __repr__(self):
        '''
        Friendly representation.
        '''
        return(f'The orders dataframe {self.shape}')