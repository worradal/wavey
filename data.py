from __future__ import annotations
from copy import deepcopy
import csv
from glob import glob
from os.path import join
from typing import Tuple

import numpy as np
from scipy.fft import fft, ifft
from tqdm.auto import tqdm
import pandas as pd

from exceptions import DataError

class Data:
    def __init__(self, in_dir: str, num_time_points: int, ftype: str='raman') -> Data:
        """
        Data structure to hold the spectrum. The data is stored in the form of x containing 
        wavenumbers or raman shifts of shape (num_sampling_points, 1) and y containing the 
        responses of shape (num_sampling_points, num_time_points). 
        
        Note:
            The responses are averaged across a third axis of length num_repitions calculated as:
            total_num_files // num_time_points.
        """
        self.num_time_points = num_time_points
        all_files = sorted(glob(join(in_dir, '*.csv')))
        num_repeats = len(all_files) // num_time_points
        self._x, self._y = None, None
        full_y = None
        time_point = 0
        for fpath in tqdm(all_files):
            x, y = self._load_data(fpath=fpath, ftype=ftype)

            if time_point == 0 or time_point == num_time_points:
                full_y = y
                if self._x is None:
                    self._x = x
                time_point = 0
            
            else:
                if self._x.shape != x.shape:
                    raise DataError('Different x-axis size between files')
                if full_y.shape[0] != y.shape[0]:
                    raise DataError(f'Different y-axis length between files. '
                                    f'Current length {self.full_y.shape[0]} but got {y.shape[0]} for file {fpath}')
                full_y = np.concatenate((full_y, y), axis=-1)
            
            if full_y.shape[-1] == num_time_points:
                if self._y is None:
                    self._y = full_y
                else:
                    self._y += full_y
                full_y = None

            time_point += 1
        
        self._y /= num_repeats
        self._init_data = (deepcopy(self._x), deepcopy(self._y))
    
    @property
    def x(self):
        return self._x
    
    @property
    def y(self):
        return self._y

    def _load_data(self, fpath: str, ftype: str) -> Tuple[np.ndarray, np.ndarray]:
        """Loads data from the file."""
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
        return data_df['x'].values.reshape((-1, 1)), data_df['y'].values.reshape((-1, 1))
    

    def fourier_transform(self):
        for row in range(self._y.shape[0]):
            self._y[row] = fft(self._y[row])
        return self
    
    def weight(self, fpath):
        weights_df = pd.read_csv(fpath)
        all_weights = weights_df['weights'].values
        if len(all_weights) != self._y.shape[1]:
            raise DataError(f'Number of weights {len(weights_df)} does not match length of y-samples along time axis {self._y.shape[1]}')
        for w_id, w in enumerate(all_weights):
            self._y[:, w_id] *=  w
        return self

    def inverse_fourier_transform(self):
        for row in range(self._y.shape[0]):
            self._y[row] = ifft(self._y[row])
        return self
    
    def save_to(self, fpath):
        df = pd.DataFrame(np.concatenate((self._x, self._y), axis=-1))
        print('Writing to ', fpath)
        df.to_csv(fpath)

