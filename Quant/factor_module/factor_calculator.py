import pandas as pd
import numpy as np
import time

from Quant.data.mysql_processor import mysql_processor
from Quant.decorator.factor_cal_decorators import rolling_window_decorator,stack_matrix_decorator

import numpy as np
from functools import wraps
'''
def _max_value(sub_data_matrix: np.ndarray) -> np.ndarray:
    # 传入窗口内数据，返回最大值
    sub_data_matrix = sub_data_matrix.T
    factor_matrix = np.max(sub_data_matrix, axis=1)
    return factor_matrix'''

def _max_value(data_matrix: np.ndarray) -> np.ndarray:
    factor_matrix = np.max(data_matrix, axis=2)
    return factor_matrix

class factor_calculator:
    def __init__(self, data:pd.DataFrame, start_date:str, window:int):
        self.start_date = start_date
        self.data = data
        self.window = window
    
    @stack_matrix_decorator
    def max_value(self):
        return _max_value(self.data_matrix)
    



if __name__ == '__main__':
    connection = mysql_processor()
    data = connection.get_data_for_factor('ETH','2024-01-01 00:00:00','2024-07-02 00:00:00',200)
    data = data[['DateTime','Close']]
    factor_module = factor_calculator(data,'2024-01-01 00:00:00',10)
    factor_module.max_value()
