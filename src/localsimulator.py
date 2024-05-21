import logging
import multiprocessing as mp
import os
from collections import defaultdict
from pathlib import Path

import basicfunc # src/basicfunc.py
import numpy as np
import pandas as pd
import xarray as xr

logger = logging.getLogger(__name__)
# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

class Simulator:
    def __init__(self):
        self.group = None
        self.date_list = None
        self.bt_date = None
        self.isdt = None
        self.iszt = None
        self.vwap_ret = None
        self.cpu_num = int(0.5 * mp.cpu_count())

    def scalebook(self, alpha):
        """Scale the book data by alpha
        """
        alpha_pos = np.where(alpha>0, alpha, 0)
        alpha_neg = np.where(alpha<0, alpha, 0)

        alpha_pos_value = (alpha_pos.T / np.nansum(np.abs(np.array(alpha_pos)), axis=1)).T * 10e6
        alpha_neg_value = (alpha_neg.T / np.nansum(np.abs(np.array(alpha_neg)), axis=1)).T * 10e6

        alpha_tmp = np.where(alpha>0, alpha_pos_value, alpha)
        alpha_res = np.where(alpha_tmp<0, alpha_neg_value, alpha_tmp)

        return alpha_res
    
    def keepzdt(self, alpha, startdi, enddi, iszt, isdt):
        """Decide whether to drop the stocks stop trading
        """
        iszdt = iszt + isdt
        alpha[0][iszdt[startdi] == 1] = np.nan
        for di in range(startdi + 1, enddi + 1):
            alpha[di - startdi][iszdt[di] == 1] = alpha[di - startdi - 1][
                iszdt[di] == 1
            ]
        return alpha
    
    def simu(
            self,
            alphas,
            startdate,
            enddate,
            filename,
            colnames,
            flag=0.0,
            group_detail="off",
            stockwise_export="off",
            vwap="VWAP",
    ):
        alphas_cnt = len(alphas)
        self.actdays = basicfunc.get_datelist(startdate, enddate, 1, -1)
        self.iszt = basicfunc.loadcache(self.actdays[0], enddate, "ISZT1", "BASEDATA")
        self.date_list = basicfunc.loadcache(self.actdays[0], enddate, "DAYS", "BASEDATA")
        self.isdt = basicfunc.loadcache(self.actdays[0], enddate, "ISDT1", "BASEDATA")
        self.vwap_ret = basicfunc.loadcache(self.actdays[0], enddate, vwap + "RET", "BASEDATA")
        self.index_ret = basicfunc.loadcache(self.actdays[0], enddate, "I500RFET", "BASEDATA")
        self.group = basicfunc.loadcache(self.actdays[0], enddate, "WIND01", "BASEDATA")
        self.startdi = self.date_list.tolist().index(self.actdays[1])
        self.enddi = self.startdi + len(self.actdays) - 1

        self.tickers = basicfunc.loadcache(
            self.actdays[0], self.actdays[-1], "STOCKS", "BASEDATA"
        )

        pnllist = []
        results = []
        loglist = ""

        pool = mp.Pool(process = self.cpu_num)
        for j in range(alphas_cnt):
            tempname = filename + "_" + colnames[j]
            tempflag = flag

            res = pool.apply_async(
                self._simu,
                (
                    j,
                    self.startdi,
                    self.enddi,
                    tempname,
                    tempflag,
                    group_detail,
                    stockwise_export,
                    alphas[j].copy(),
                ),
            )
            results.append(res)

        templogs = [res.get() for res in results]
        pool.close()
        pool.join()

        for idx, msg, resstr in templogs:
            if msg != "":
                loglist = f"Alpha {colnames[idx]} Exception: \n{msg}"
                # loglist += " ".join(["\nAlpha", colnames[idx], "\n", msg])
            pnllist += [resstr]
        return loglist
    
    def _simu(
            self,
            j,
            startdi,
            enddi,
            filename,
            costflag, 
            group_detail,
            stockwise_export,
            alpha,
    ):
        histdays = enddi - startdi + 1
        self.bt_date = self.date_list[startdi:enddi + 1].copy()

        raw_alpha = self.keepzdt(alpha.copy(), startdi, enddi, self.iszt, self.isdt)
        alpha = self.keepzdt(alpha, startdi, enddi, self.iszt, self.isdt)
        alpha = self.scalebook(alpha)
        palpha = alpha[0 : enddi - startdi + 1].copy()
        palpha[palpha <= 0] = np.nan
        nAlpha = alpha[0 : enddi - startdi + 1].copy()
        nAlpha[nAlpha > 0] = np.nan
        alpha[np.isnan(alpha)] = 0
        palpha[np.isnan(palpha)] = 0
        nAlpha[np.isnan(nAlpha)] = 0
        



