import csv
from typing import Tuple

import numpy as np
import pandas as pd


class Data:
    def __init__(self, fpath: str, ftype: str='raman'):
        self.x, self.y = self._load_data(fpath=fpath, ftype='raman')
    
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
        return data_df['x'].values, data_df['y'].values

