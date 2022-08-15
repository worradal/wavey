from copy import deepcopy
import csv
from distutils.ccompiler import gen_lib_options
from glob import glob
from os.path import join
from __future__ import annotations
from typing import Tuple, Optional

import numpy as np
from scipy.fft import fft, ifft
from tqdm.auto import tqdm
import pandas as pd

from .exceptions import DataError

class Data:
    def __init__(self, in_dir: str, ftype: str='raman'):
        self._x, self._y = None, None
        for fpath in tqdm(glob(join(in_dir, '*.csv'))):
            x, y = self._load_data(fpath=fpath, ftype='raman')
            if self._x is None:
                self._x = x
            elif self._x.shape != x.shape:
                raise DataError('Different x-axis size between files')
            if self._y is None:
                self._y = y
            elif self._y.shape[0] != y.shape[0]:
                raise DataError('Different y-axis length between files')
            else:
                self._y = np.concatenate((self._y, y), axis=-1)
        self._init_data = (deepcopy(self._x), deepcopy(self._y))
        return self
    
    @property
    def x(self):
        return self._x
    
    @property
    def y(self):
        return self._y

    def _load_data(self, fpath: str, ftype: str) -> Tuple[np.ndarray, np.ndarray]:
        if ftype.lower() == 'raman':
            x_name = 'Raman Shift'
            x_col = -1
            y_name = 'Dark Subtracted #1'
            y_col = -1
            useful_data_start = False
            data = {'x': [], 'y': []}
            with open(fpath, newline='') as csvfile:
                reader = csv.reader(csvfile, delimiter=',')
                for row in reader:
                    if useful_data_start:
                        assert x_col != -1 and y_col != -1
                        try:
                            y_val = float(row[y_col])
                        except ValueError as e:
                            print(type(e))
                            y_val = np.nan
                        data['y'].append(y_val)
                        
                        try:
                            x_val = float(row[x_col])
                        except ValueError as e:
                            x_val = np.nan
                        data['x'].append(x_val)

                    if x_name in row:
                        useful_data_start = True
                        x_col = row.index(x_name)
                        y_col = row.index(y_name)
 
        else:
            raise NotImplementedError(f'{ftype} ftype not supported')

        data_df = pd.DataFrame(data).dropna()
        return data_df['x'].values.reshape((-1, 1)), data_df['y'].values
    

    def fourier_transform(self):
        for row in range(self._x.shape[0]):
            self._x[row] = fft(self._x[row])
        return self
    
    def weight(self, fpath):
        weights_df = pd.read_csv(fpath).iloc[0]
        n_time_samples = weights_df.shape[1]
        if n_time_samples != self._y.shape[-1]:
            raise DataError(f'Number of wights {n_time_samples} does not match number of y-samples {self._y.shape[-1]}')
        weights = weights_df.to_numpy().reshape(n_time_samples, -1)  # (n_time_samples, 1)
        for w_id, w in enumerate(weights):
            self._y[:, w_id] *=  w
        return self

    def inverse_fourier_transform(self):
        for row in range(self._x.shape[0]):
            self._x[row] = ifft(self._x[row])
        return self
    
    def save_to(self, fpath):
        df = pd.DataFrame(np.concatenate(self._x, self._y), axis=-1)
        print('Writing to ', fpath)
        df.to_csv(fpath)

