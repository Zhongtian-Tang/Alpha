import os
import sys

import numpy as np
import pandas as pd

from basefactor import fetcher

class Generator():
    def __init__(self, startdate, enddate):
        self.startdate = startdate
        self.enddate = enddate
        self.path = '../data/cache/BASEDATA'
        self.year = self.startdate[:4]    
    
    @classmethod
    def pivot_data(self, df):
        """
        Pivot DataFrame got from fetch_data().
        """
        columname = df.columns
        df = df.pivot_table(index=columname[0], columns=columname[1], values=columname[-1])
        return df.to_numpy()
    
    def fetch_data(self):
        """
        Fetch basic data from local JYDB database.
        """
        data_list = ["ASHARE", "WIND01", "ISZT", "VWAP", "VWAPRET", "ISDT"]
        dailydata = fetcher.get_tradedate_data('jydb.dailydata', self.startdate, self.enddate,\
                                               'ashares', 'firstindustrycode', 'ifsuspend',\
                                               'avgprice', 'changepct', 'iftradable')\
                                                .set_index(['tradedate', 'wind_code']).sort_index()
        for i in range(len(data_list)):
            temp = dailydata.iloc[:, i].reset_index()
            temp = self.pivot_data(temp)
            np.save(self.path + f"/{self.year}/" + data_list[i] + ".npy", temp)
        
        codes = dailydata.index.get_level_values(1).unique().to_numpy()
        np.save(self.path + f"/{self.year}" + "/STOCKS.npy", codes)
        
        return 0


        
        


