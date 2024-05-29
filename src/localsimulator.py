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
        """Scale the alpha to 10e6. This function accept a array and transform the alpha to real
        position value.
        """
        alpha_pos = np.where(alpha>0, alpha, 0)
        alpha_neg = np.where(alpha<0, alpha, 0)

        # Add a small number to avoid the zero division
        alpha_pos_value = (alpha_pos.T / (np.finfo(float).eps + np.nansum(np.abs(np.array(alpha_pos)), axis=1))).T * 10e6
        alpha_neg_value = (alpha_neg.T / (np.finfo(float).eps + np.nansum(np.abs(np.array(alpha_neg)), axis=1))).T * 10e6

        # Use the buy and sell value to replace the original alpha value
        alpha_tmp = np.where(alpha>0, alpha_pos_value, alpha)
        alpha_res = np.where(alpha_tmp<0, alpha_neg_value, alpha_tmp)

        return alpha_res
    
    def keepzdt(self, alpha, startdi, enddi, iszt, isdt):
        """Decide whether to drop the stocks stop trading
        """
        iszdt = iszt + isdt

        # If the stock is not trading, set the alpha to nan 
        alpha[0][iszdt[startdi] == 1] = np.nan

        for di in range(startdi + 1, enddi + 1):
            # If the stock is not trading, set the alpha to the previous day
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
        self.iszt = basicfunc.loadcache(self.actdays[0], enddate, "ISZT", "BASEDATA")
        self.date_list = basicfunc.loadcache(self.actdays[0], enddate, "DAYS", "BASEDATA")
        self.isdt = basicfunc.loadcache(self.actdays[0], enddate, "ISTP", "BASEDATA")
        self.vwap_ret = basicfunc.loadcache(self.actdays[0], enddate, vwap + "RET", "BASEDATA") / 100
        self.index_ret = basicfunc.loadcache(self.actdays[0], enddate, "IRE500", "BASEDATA")
        self.group = basicfunc.loadcache(self.actdays[0], enddate, "WIND01", "BASEDATA")
        self.startdi = self.date_list.tolist().index(self.actdays[1])
        self.enddi = self.startdi + len(self.actdays) - 1

        self.tickers = basicfunc.loadcache(
            self.actdays[0], self.actdays[-1], "STOCKS", "BASEDATA"
        )

        pnllist = [] # Store the pnl result for each alpha
        results = [] # Store async for each alpha simulation result
        loglist = ""

        pool = mp.Pool(processes=self.cpu_num)
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
        self.bt_date = self.date_list[startdi : enddi + 1].copy()

        raw_alpha = self.keepzdt(alpha.copy(), startdi, enddi, self.iszt, self.isdt)
        alpha = self.keepzdt(alpha, startdi, enddi, self.iszt, self.isdt)
        alpha = self.scalebook(alpha)
        palpha = alpha[0 : enddi - startdi + 1].copy()
        palpha[palpha <= 0] = np.nan
        nalpha = alpha[0 : enddi - startdi + 1].copy()
        nalpha[nalpha > 0] = np.nan
        alpha[np.isnan(alpha)] = 0.0
        palpha[np.isnan(palpha)] = 0.0
        nalpha[np.isnan(nalpha)] = 0.0
        logresult = ""
        recordstr = ""

        ## check dump part and maxwagt part
        """
        Calculate the sum of the stocks that are actually traded everyday
        and the days that the traded stocks are less than 50
        """
        dumpum = np.sum(~(alpha[0 : enddi - startdi + 1] == 0), axis=1)
        dumdays = np.where(dumpum <= 50)[0]
        for i in range(dumdays.shape[0]):
            logresult = (
                logresult
                + "only dump "
                + str(dumpum[dumdays[i]])
                + " stocks on "
                + str(self.date_list[startdi + dumdays[i]])
                + "\n"
            )

        # check the situation that one stock has over weight in the trading day
        ta = np.nanmax(palpha, axis=1) / np.nansum(palpha, axis=1)
        tb = np.nanmax(nalpha, axis=1) / np.nansum(nalpha, axis=1)
        ta[np.isinf(ta)] = 0.0
        tb[np.isinf(tb)] = 0.0
        tadays = np.where(ta > 0.1)[0]
        tbdays = np.where(tb > 0.1)[0]

        
        for i in range(tadays.shape[0]):
            logresult += (
                " Max weight "
                + str(ta[tadays[i]])
                + " on "
                + str(self.date_list[startdi + tadays[i]])
                + "\n"
            )

        for i in range(tbdays.shape[0]):
            logresult += (
                " Max weight "
                + str(-tb[tbdays[i]])
                + " on "
                + str(self.date_list[startdi + tbdays[i]])
                + "\n"
            )


        # * Calculate the pnl and other metrics everyday
        retmatrix = np.r_[
            np.full((1, alpha.shape[1]), 0.0),
            alpha[0, histdays - 1] * self.vwap_ret[startdi + 1 : enddi + 1],
        ]

        holdpnl = np.r_[
            np.nansum(np.full((2, alpha.shape[1]), 0.0), axis=1),
            np.nansum(np.minimum(alpha[1:histdays-1], alpha[0 : histdays - 2])*self.vwap_ret[startdi+2 : enddi+1], axis=1),
        ]

        tradepnl = np.nansum(retmatrix, axis=1) - holdpnl # The potential change cost
        trade_cap = np.around(
            np.r_[
                np.nansum(np.abs(alpha[0])),
                np.nansum(np.abs(alpha[1:histdays] - alpha[0: histdays - 1]), axis=1),
            ],
            decimals=2,
        ) # The capital used for trading, sum may less than 2e6 since some stocks are not trading
        delta = alpha[1:histdays] - alpha[0 : histdays - 1]
        pdelta = palpha[1:histdays] - palpha[0 : histdays - 1]
        ndelta = nalpha[1:histdays] - nalpha[0 : histdays - 1]
        poscost = np.r_[
            trade_cap[0] * 0.0002 * costflag, # 0.0002 is the transaction cost
            np.nansum(np.where(delta > 0, delta, 0), axis=1) * 0.0002 * costflag,
        ]
        negcost = np.r_[
            0.0, np.nansum(np.where(delta < 0, delta, 0), axis=1) * -0.0012 * costflag
        ]
        poscost2 = np.r_[
            0.0, np.nansum(np.where(pdelta > 0, pdelta, 0), axis=1) * 0.0002 * costflag
        ]
        negcost2 = np.r_[
            0.0, np.nansum(np.where(pdelta < 0, pdelta, 0), axis=1) * -0.0012 * costflag
        ]
        poscost3 = np.r_[
            0.0, np.nansum(np.where(ndelta > 0, ndelta, 0), axis=1) * 0.0002 * costflag
        ]
        negcost3 = np.r_[
            0.0, np.nansum(np.where(ndelta < 0, ndelta, 0), axis=1) * -0.0012 * costflag
        ]
        pospnl = np.around(
            np.r_[
                0.0,
                np.nansum(
                    np.where(alpha[0 : histdays - 1] > 0, retmatrix[1:], 0), axis=1
                ),
            ]
            - poscost2
            - negcost2,
            decimals=2,
        )
        negpnl = np.around(
            np.r_[
                0.0,
                np.nansum(
                    np.where(alpha[0 : histdays - 1] < 0, retmatrix[1:], 0), axis=1
                ),
            ]
            - poscost3
            - negcost3,
            decimals=2,
        )

        # * Use xarray to calc IC now(behavior like df.corr but faster 300x)
        IC = np.around(
            np.r_[
                0.0,
                xr.corr(
                    xr.DataArray(alpha[0 : enddi - startdi], dims=("date", "ticker")),
                    xr.DataArray(
                        self.vwap_ret[startdi + 1 : enddi + 1], dims=("date", "ticker")
                    ),
                    dim="ticker",
                ),
            ],
            decimals=6,
        )

        inx_ret = np.around(
            np.r_[0.0, self.index_ret[startdi + 1 : enddi + 1]], decimals=6
        )
        
        # Coverage Rate
        poscov = np.around(np.nansum(alpha > 0, axis=1), decimals=1)
        negcov = np.around(np.nansum(alpha < 0, axis=1), decimals=1)

        longsize = np.around(np.nansum(palpha, axis=1), decimals=1)
        shortsize = np.around(np.nansum(nalpha, axis=1), decimals=1)
        pnl = np.around(np.nansum(retmatrix, axis=1) - poscost - negcost, decimals=2)
        ret = np.around(pnl / np.nansum(abs(alpha), axis=1), decimals=6)
        ret[np.isnan(ret)] = 0.0

        if group_detail == "on":
            temprank = pd.DataFrame(raw_alpha[0 : enddi - startdi]).rank(
                pct = True, axis=1
            )
            rankpnl = []
            for i in range(10):
                lower_bound = i / 10
                upper_bound = (i + 1) / 10
                pnl_per_group = np.around(
                    np.r_[
                        0.0,
                        np.nansum(
                            np.where(
                                (temprank > lower_bound) & (temprank <= upper_bound),
                                self.vwap_ret[startdi + 1 : enddi + 1],
                                0,
                            ),
                            axis=1,
                        ),
                    ],
                    decimals=6,
                )
                rankpnl.append(pnl_per_group)
            
            recordstr += (
                "date, pnl, longsize, shortsize, ret, holdpnl, tradepnl, poscov, negcov, pospnl, negpnl, IC, index_ret,"
                + "rankpnl1, rankpnl2, rankpnl3, rankpnl4, rankpnl5, rankpnl6, rankpnl7, rankpnl8, rankpnl9, rankpnl10\n"
                )
        else:
            recordstr += (
                "date, pnl, longsize, shortsize, ret, holdpnl, tradepnl, poscov, negcov, pospnl, negpnl, IC, index_ret\n"
            )

        for di in range(startdi, enddi + 1):
            # date, pnl, longsize, shortsize, ret, holdpnl, tradepnl, poscov, negcov, pospnl, negpnl, IC, index_ret
            recordstr += (
                f"{int(self.bt_date[di - startdi])},"
                f"{pnl[di - startdi]},"
                f"{longsize[di - startdi]},"
                f"{shortsize[di - startdi]},"
                f"{ret[di - startdi]},"
                f"{holdpnl[di - startdi]},"
                f"{tradepnl[di - startdi]},"
                f"{poscov[di - startdi]},"
                f"{negcov[di - startdi]},"
                f"{pospnl[di - startdi]},"
                f"{negpnl[di - startdi]},"
                f"{IC[di - startdi]},"
                f"{inx_ret[di - startdi]}"
            )

            if "group_detail" == "on":
                recordstr += (
                    f",{rankpnl[0][di - startdi]},"
                    f"{rankpnl[1][di - startdi]},"
                    f"{rankpnl[2][di - startdi]},"
                    f"{rankpnl[3][di - startdi]},"
                    f"{rankpnl[4][di - startdi]},"
                    f"{rankpnl[5][di - startdi]},"
                    f"{rankpnl[6][di - startdi]},"
                    f"{rankpnl[7][di - startdi]},"
                    f"{rankpnl[8][di - startdi]},"
                    f"{rankpnl[9][di - startdi]}"
                )

            recordstr += "\n"

        pnl_folder_path = Path(filename).parent.absolute()
        pnl_filname = Path(filename).absolute()

        if stockwise_export == "on":
            alpha_filename = pnl_folder_path / filename.split("/")[-1].replace(
                "PNL", "ALPHA"
            )

            ret_filname = pnl_folder_path / filename.split("/")[-1].replace(
                "PNL", "RET"
            )

            retmat_filename = pnl_folder_path / filename.split("/")[-1].replace(
                "PNL", "RETMAT"
            )

            # alpha index use previous day's date(self.actdays)
            alpha_csv = pd.DataFrame(
                raw_alpha[0 : histdays - 1],
                index=self.actdays[:-1],
                columns=self.tickers,
            )
            ret_csv = pd.DataFrame(
                self.vwap_ret[startdi + 1 : enddi + 1],
                index=self.bt_date[1:],
                columns=self.tickers,
            )
            # retmatrix index keep the same as `vwap_ret`'s (self.bt_date)
            retmat_csv = pd.DataFrame(
                retmatrix[1:],
                index=self.bt_date[1:],
                columns=self.tickers,
            )
            alpha_csv.to_csv(f"{alpha_filename}.csv", float_format="%.6f")
            ret_csv.to_csv(f"{ret_filname}.csv", float_format="%.6f")
            retmat_csv.to_csv(f"{retmat_filename}.csv", float_format="%.6f")

        with open(f"{pnl_filname}.csv", "w+") as f:
            f.write(recordstr)
        
        return j, logresult, recordstr

        



