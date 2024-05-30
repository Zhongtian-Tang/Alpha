import os
import sys

import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict

from basefactor import fetcher

class Generator():
    def __init__(self, startdate, enddate):
        self.startdate = startdate
        self.enddate = enddate
        self.path = '../data/cache/BASEDATA'
        self.years = np.arange(int(startdate) // 10000, int(enddate) // 10000 + 1)
        self.base_data_list = [
            "ASHARE", "CAP", "OPEN", "HIGH", "LOW", "CLOSE", "VWAP",\
            "VWAPRET", "VOL", "ISZT", "ISTP", "WIND01", "DAYS", "STOCKS", "IRE500" 
        ]
          
    
    @classmethod
    def pivot_data(self, df, name):
        """
        Pivot DataFrame got from fetch_data().
        """
        df = df.pivot_table(index='tradedate', columns='wind_code', values=name, aggfunc='mean')
        return df.to_numpy()
    
    def create_folder(self, year):
        """Create folder for each year.
        """
        datapath = Path(self.path+f"/{year}")
        datapath.mkdir(parents=True, exist_ok=True)
        print(f"Folder {year} created.")


    def save_base_data(self, year):
        """
        Fetch basic data from local JYDB database.
        """
        start_date = str(year) + '-01-01'
        end_date = str(year) + '-12-31'
        dailydata = fetcher.get_tradedate_data('jydb.dailydata', start_date, end_date,\
                                               'ashares', 'firstindustrycode', 'ifsuspend',\
                                               'changepct', 'totalmv', 'avgprice',\
                                               'openprice', 'highprice', 'lowprice',\
                                               'closeprice', 'turnovervolume')\
                                               .sort_values(by=['tradedate', 'wind_code'])
        
        dstockdata = fetcher.get_market_data('jydb.dstock', start_date, end_date,\
                                             'VWAP', 'MAXUPORDOWN', 'VAL_FLOATMV')\
                                            .sort_values(by=['tradedate', 'wind_code'])
        # Remove stocks with suffix '.BJ'
        dstockdata = dstockdata[~dstockdata['wind_code'].str.endswith('.BJ')]
        
        indexdata = fetcher.get_tradedate_data('jydb.index_quote_daily', start_date, end_date,
                                               'closeprice').sort_values(by=['tradedate', 'wind_code'])
        
        base_data_dict = defaultdict()
        base_data_dict['ASHARE'] = self.pivot_data(dailydata, 'ashares')
        base_data_dict['WIND01'] = self.pivot_data(dailydata, 'firstindustrycode')
        base_data_dict['ISTP'] = self.pivot_data(dailydata, 'ifsuspend')
        base_data_dict['VWAPRET'] = self.pivot_data(dailydata, 'changepct')
        base_data_dict['VOL'] = self.pivot_data(dailydata, 'turnovervolume')
        base_data_dict['OPEN'] = self.pivot_data(dailydata, 'openprice')
        base_data_dict['HIGH'] = self.pivot_data(dailydata, 'highprice')
        base_data_dict['LOW'] = self.pivot_data(dailydata, 'lowprice')
        base_data_dict['CLOSE'] = self.pivot_data(dailydata, 'closeprice')
        base_data_dict['VWAP'] = self.pivot_data(dailydata, 'avgprice')
        base_data_dict['CAP'] = self.pivot_data(dstockdata, 'VAL_FLOATMV')
        base_data_dict['ISZT'] = self.pivot_data(dstockdata, 'MAXUPORDOWN')
        base_data_dict['DAYS'] = pd.Series(dailydata['tradedate'].\
                                           unique()).astype(str).str.\
                                           replace('-','').astype(int).values
        base_data_dict['STOCKS'] = np.sort(np.array(dailydata['wind_code'].unique()))
        base_data_dict['IRE500'] = indexdata[indexdata['wind_code'] == '000001.SH']['closeprice']

        return base_data_dict

    
    def execute_save(self):
        for year in self.years:
            self.create_folder(year)
            temp_data = self.save_base_data(year)
            for key in self.base_data_list:
                np.save(f"{self.path}/{year}/{key}.npy", temp_data[key])
                print(f"Year {year} items {key} saved. shape is {temp_data[key].shape}")
            print(f"Year {year} saved.")
        


        
        


