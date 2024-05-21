import os

import numpy as np
import pandas as pd
from sqlalchemy import create_engine


def update_datelist(begT, endT, dback=0, dnext=0):
    """Update the date list based on the beginning and ending dates
       and save it to the cache folder.

    Parameters
    ----------
    begT : str
        The beginning date
    endT : str
        The ending date
    dback : int
        The shift days before first trading day
    dnext : int
        The shift days after last trading day
    
    Returns
    -------
    int
        0 if the function runs successfully
    """
    db_connection_str = "oracle+cx_oracle://rejy:jcFXLzL10@10.224.6.3:1522/?service_name=orcl"
    engine = create_engine(db_connection_str)
    """ Query the trading date list from the database """
    start_date = '2000-12-31'
    sql = f"""
        SELECT TradingDate FROM JYDB.QT_TradingDayNew
        WHERE TradingDate >= to_date('{start_date}', 'YYYY-MM-DD')
        AND SecuMarket = 83 AND IfTradingDay = 1
        ORDER BY TradingDate
        """
    date_list = pd.read_sql(sql, engine)
    date_list['tradingdate'] = date_list['tradingdate'].unique().astype(str)
    trading_days = date_list['tradingdate'].replace('-', '', regex=True).astype(int)

    beg_index = len(trading_days[trading_days < int(begT)]) - dback
    end_index = len(trading_days[trading_days <= int(endT)]) + dnext
    np.save('../data/cache/BASEDATA/traingdays.npy', trading_days[beg_index:end_index])
    return 0

def get_datelist(begT, endT, dback=0, dnext=0):
    """Get the date list based on the beginning and ending dates
    """
    if os.path.exists('../data/cache/BASEDATA/traingdays.npy'):
        trading_days = np.load('../data/cache/BASEDATA/traingdays.npy')
        beg_index = len(trading_days[trading_days < int(begT)]) - dback
        end_index = len(trading_days[trading_days <= int(endT)]) + dnext
        if end_index <= len(trading_days):
            return trading_days[beg_index:end_index]
    
    db_connection_str = "oracle+cx_oracle://rejy:jcFXLzL10@10.224.6.3:1522/?service_name=orcl"
    engine = create_engine(db_connection_str)
    """ Query the trading date list from the database """
    sql = f"""
        SELECT TradingDate FROM JYDB.QT_TradingDayNew
        WHERE TradingDate >= to_date('{begT}', 'YYYYMMDD')
        AND TradingDate <= to_date('{endT}', 'YYYYMMDD')
        AND SecuMarket = 83 AND IfTradingDay = 1
        ORDER BY TradingDate
        """
    date_list = pd.read_sql(sql, engine)
    trading_days = date_list['tradingdate'].unique().astype(str).replace('-', '', regex=True).astype(int)

    beg_index = len(trading_days[trading_days < int(begT)]) - dback
    end_index = len(trading_days[trading_days <= int(endT)]) + dnext

    return trading_days[beg_index:end_index]

def get_symbols(startdate, enddate):
    """Get the symbols based on the beginning and ending dates
    """
    newpath = '../data/cache/'
    years = np.arange(int(startdate) // 10000, int(enddate) // 10000 + 1)
    symbols = np.load(
        newpath + 'BASEDATA/' + str(years[len(years) - 1]) + '/STOCKS.npy'
        )
    return symbols

def get_startdi(startdate, enddate, actdays):
    """Get the start date index based on the beginning and ending dates
    """
    days = get_datelist(startdate, enddate, 0, 0)
    startdi = loadcache(actdays[0], actdays[-1], "DAYS").tolist().index(days[0])
    enddi = startdi + len(days) - 1
    return startdi, enddi

def loadcache(startdate, enddate, dataitem, source='BASEDATA'):
    """
    Load the cache data in specified range of dates. This function is used to deal with data with different dimensions.
    Use same strategy to deal with the data with different dimensions when concatenating multy-year data.

    Parameters
    ----------
    startdate : str
        The beginning date
    enddate : str
        The ending date
    dataitem : str
        The data item to be loaded, e.g "STOCKS", "VOLUME".
    source : str, optional
        The source of the data, e.g. "BASEDATA", "FACTOR". The default is 'BASEDATA'.
    
    Returns
    ----------
    np.array
        The data in the specified range of dates
    
    Examples
    ----------
    Load the data of 'STOCKS' between 2018 and 2020:
    - ndim = 1:
        2018: [1, 2, 3, 4, 5]
        2019: [6, 7, 8, 9, 10]
        2020: [11, 12, 13, 14, 15]
        result: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
    
    - ndim = 2:
        2018: [[1, 2], [3, 4]]
        2019: [[5, 6, 7], [8, 9, 10]]
        2020: [[11, 12, 13], [14, 15, 16]]
        result: [[1, 2, NaN], [3, 4, NaN], [5, 6, 7], [8, 9, 10], [11, 12, 13], [14, 15, 16]]
    
    - ndim = 3:
        2018: [[[100, 101]], [[200, 201]]]
        2019: [[[110, 111], [210, 211], [310, 311]]]
        2020: [[[120, 121], [220, 221], [320, 321]]]
        result: [[[100, 101], [NaN, NaN], [NaN, NaN]], [[200, 201], [NaN, NaN], [NaN, NaN]],
                  [[110, 111], [210, 211], [310, 311]], [[120, 121], [220, 221], [320, 321]]]
    """
    newpath = "../data/cache/"
    if source != "" and not source.endswith("/"):
        source += "/"
    years = np.arange(int(startdate) // 10000, int(enddate) // 10000 + 1)
    instruments = len(np.load(newpath + "BASEDATA/" + str(years[-1]) + '/STOCKS.npy'))
    
    if dataitem == "STOCKS":
        return np.load(newpath + source + str(years[-1]) + '/' + dataitem + '.npy')
    
    alldata = []
    baseoffsets = 0
    for yr in range(len(years)):
        year = years[yr]
        data = np.load(newpath + source + str(year) + "/" + dataitem + '.npy')
        if data.ndim == 1:
            alldata.append(data)
        elif data.ndim == 2:
            """
            Match the number of obesevations with the missing stocks
            (which are not in the current year but in the last year)
            """
            adddata = np.full([data.shape[0], instruments - data.shape[1]], np.nan)
            alldata.append(np.concatenate((data, adddata), axis=1))
        elif data.ndim == 3:
            if dataitem == "offsets" or dataitem == "OFFSETS":
                if yr > 0:
                    addbase = np.load(
                        newpath + source + str(years[yr - 1]) + "/" + "RECORDNUM.npy"
                    )[-1]
                    baseoffsets += addbase
                data += baseoffsets
            alldata = np.full(
                [data.shape[0], data.shape[1], instruments - data.shape[2]], np.nan
            )
            alldata.append(np.concatenate((data, adddata), axis=2))
        
    if alldata[0].ndim <= 2:
        return np.concatenate(alldata, axis=0)
    else:
        return np.concatenate(alldata, axis=1)

def IndNeu_help(group, xlist):
    """
    Help function for neutralize the factors in the groups.

    Parameters
    ----------
    group : np.array
        The group array, should be the same length as xlist
    xlist : np.array
        The factor array to be neutralized
    
    Returns
    ----------
    np.array
        The neutralized factor array

    """
    keys = pd.Series(group).dropna().unique()
    for gg in keys:
        xlist[group == gg] = xlist[group == gg] - np.mean(xlist[group == gg])
    xlist[np.isnan(group)] = np.nan

    return xlist


def IndNeutralize(xmatrix, groupmatrix=[]):
    """
    Neutralize the factors in the groups.

    Parameters
    ----------
    xmatrix : np.array
        The factor matrix to be neutralized
    groupmatrix : np.array, optional
        The group matrix, each element is the array with the same rows as xmatrix, represents the row group data. The default is [].
    
    Returns
    ----------
    np.array
        The neutralized factor matrix
    """
    if groupmatrix == []:
        xmatrix = (xmatrix.T - np.nanmean(xmatrix, axis=1)).T
        return xmatrix
    for di in range(xmatrix.shape[0]):
        xmatrix[di] = IndNeu_help(groupmatrix[di], xmatrix[di])
    return xmatrix 



