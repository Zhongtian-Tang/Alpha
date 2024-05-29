import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

import inspect
import logging
from collections import defaultdict

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

import basicfunc
import summary
from localsimulator import Simulator
from operation import Operation


class FactorBase:

    def __init__(self):
        self.delay = 1
        self.histdays = 10 # need histdays >= delay, ensure that data from startdate to histdays is loaded at least once
        self.alpha_list = []
        self.alpha_num = 0

    def generate_features(self, raw_feat: np.ndarray) -> np.ndarray:
        """
        Generate features from raw `BASEDATA`.

        Slice ndarray by start/end index, the slicing index is from `startdi - delay` to `enddi - delay + 1`.

        Parameters
        ----------
        raw_feat : np.ndarray
            The raw features from `BASEDATA`.

        Returns
        ----------
        numpy.ndarray
            The sliced feature array by fixed rule.
        """
        cur_start = self.startdi - self.delay
        cur_end = self.enddi - self.delay + 1
        if raw_feat.ndim == 3:
            return raw_feat[:, cur_start:cur_end, :]
        return raw_feat[cur_start:cur_end, :]
    
    def preprocess_alpha(self, method: str) -> None:
        """Apply data preprocessing methods to alpha ndarray.

        Parameters
        ----------
        method : str
            The name of preprocessing method.

        Raises
        ----------
        AttributeError
            If the method is not supported.
        TypeError
            If the method is not callable.
        """
        try:
            method_to_call = getattr(Operation, method)
        except AttributeError:
            raise ValueError(f"Method '{method}' is not supported.")
        
        if not callable(method_to_call):
            raise ValueError(f"Method '{method}' is not callable.")
        
        groupdata = basicfunc.loadcache(
            self.actdays[0], self.actdays[-1], "WIND01", "BASEDATA"
            )
        
        for i in range(self.alpha.shape[0]):
            if method == "ind_neutralize":
                self.alpha[i] = Operation.ind_neutralize(
                    self.alpha[i], groupdata[self.startdi - 1 : self.enddi]
                    )
            else:
                self.alpha[i] = method_to_call(self.alpha[i])
        
        logging.info(f"{method.capitalize()} applied.")

    def get_constants(self):
        """Generate constants for alpha calculation.

        Get days, tickers, etc. using `basicfunc` functions.
        """
        self.actdays = basicfunc.get_datelist(
            self.start_date, self.end_date, self.histdays, -1
            )
        
        self.tickers = basicfunc.loadcache(
            self.actdays[0], self.actdays[-1], "STOCKS", "BASEDATA"
        ).shape[0]

        self.startdi, self.enddi = basicfunc.get_startdi(
            self.start_date, self.end_date, self.actdays
        )

    def load_data(self):
        """Load data specified in `need_fields` and `need_hdim_fields` from cache.
        """
        self.need_data = defaultdict()
        self.need_hdim_data = defaultdict()

        """ Traversal data in `need_fields`"""
        if hasattr(self, "need_fields"):
            for field in self.need_fields:
                raw_pv_feat = basicfunc.loadcache(
                    self.actdays[0],
                    self.actdays[-1],
                    field.upper(),
                    "BASEDATA",
                )
                self.need_data[field.upper()] = self.generate_features(raw_pv_feat)
        
        """ Traversal data in `need_hdim_fields`"""
        if hasattr(self, "need_hdim_fields"):
            for key, fields in self.need_hdim_fields.items():
                for field in fields:
                    raw_hdim_feat = basicfunc.loadcache(
                        self.actdays[0],
                        self.actdays[-1],
                        field.upper(),
                        key,
                    )
                    self.need_hdim_data[field.upper()] = self.generate_features(
                        raw_hdim_feat
                    )
    
    def calculate_alpha(self):
        """Calculate alphas by traversal `alpha_list` and apply data
        preprocessing methods.
        """

        """ Traversal alphas in sub class `FactorPool`"""
        subclass_methods = inspect.getmembers(self, predicate=inspect.ismethod)
        sub_methods_list = [name for name, _ in subclass_methods]

        """ Traversal alphas in base class `FactorBase`"""
        baseclass_methods = inspect.getmembers(FactorBase, predicate=inspect.isfunction)
        base_methods_list = [name for name, _ in baseclass_methods]
        
        """Get unique method defined in `FactorPool.py`"""
        self.alpha_list = [
            item for item in sub_methods_list if item not in base_methods_list
        ]
        self.alpha_num = len(self.alpha_list)

        logging.info(
            f"{self.alpha_num} alphas will be calculated (Simsummary in alphabetical order)."
            )
        
        """ Init empty alpha ndarray with nans """
        self.alpha = np.full(
            [self.alpha_num, self.enddi - self.startdi + 1, self.tickers],
            np.nan
            )

        """ Traversal alphas in `alpha_list`"""
        for i in range(self.alpha_num):
            alpha_method = getattr(self, self.alpha_list[i], None)
            if alpha_method is not None and callable(alpha_method):
                """Set value in corresponding places."""
                temp_data = alpha_method()
                self.alpha[i] = temp_data.copy()

        logging.info("Alphas calculated.")

        """ Apply data preprocessing methods to alpha ndarray"""

        if self.winsorize:
            self.preprocess_alpha("winsorize")
        if self.zscore:
            self.preprocess_alpha("zscore")
        if self.ind_neu:
            self.preprocess_alpha("ind_neutralize")
        if self.fillna:
            self.preprocess_alpha("fillna")


    def local_simu(self):
        """Local simulation on pnl files.
        
        Generate pnl files for each alpha:

        - pnl/alpha_pv_sample (shape: days * tickers)

        - pnl/retmat_pv_sample (shape: days * tickers)

        - pnl/pnl_pv_sample (shape: days * indicator[12 or 22])

        """

        """ Set the path to save pnl files"""
        pnlpath = Path(self.path)
        pnlpath.mkdir(parents=True, exist_ok=True)
        pnlpath_ret = (
            f"{self.path}/{'PNL' if self.prefix == '' else f'{self.prefix}_PNL'}"
        )

        """ Generate pnl related files for each alpha."""
        logging.info("Generating PNL files...")
        simulator_instance = Simulator()
        log_ret = simulator_instance.simu(
            self.alpha.copy(),
            self.start_date,
            self.end_date,
            pnlpath_ret,
            self.alpha_list,
            1.0,
            self.group_detail,
            self.stockwise_export,
        )
        if len(log_ret):
            logging.warning(log_ret)

        pnlfiles = []
        """ Generate summary log to terminal."""
        for pnl in self.alpha_list:
            simfile_path = f"{pnlpath_ret}_{pnl}.csv"
            pnlfiles.append(simfile_path)
            # summary.simsummary(simfile_path, group_detail=self.group_detail)


    def execute_alphaflow(
            self,
            path: str = "./pnl",
            group_detail: str = "off",
            stockwise_export: str = "off",
            prefix: str = "",           
    ) -> None:
        """Execute workflow or pipeline related to alpha factors.

        Within the following steps:

        1. Prepare base data from ``data/cache``

        2. Calculate factors and apply data preprocessing methods.

        3. Local simulation on generated files.

        Parameters
        ----------
        path : str, optional
            The folder path save ``pnl`` files. By default, "./pnl".
        group_detail : str, optional
            Whether to generate pnl files for each group. Set to '`on`' to enable. By default, "off".
        stockwise_export : str, optional
            Whether to generate daily ``ALPHA`` and ``RETMAT`` files for each stock. Set to '`on`' to enable. By default, "off".
        prefix : str, optional
            The prefix add to ``{path}/{prefix}_{file_type}_{factor_name}``. By default, "".
        """

        """Setup parameters"""
        self.group_detail = group_detail
        self.stockwise_export = stockwise_export
        self.prefix = prefix
        self.path = path
        
        """Prepare base data."""
        self.get_constants()
        self.load_data()
        logging.info("Base data loaded.")
        
        """Calculate alphas and apply data preprocessing methods."""
        self.calculate_alpha()
        
        """Local simulation"""
        self.local_simu()
        
        

        
