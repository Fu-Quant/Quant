import pandas as pd
import numpy as np
import time
from math import *

from Quant.data.mysql_processor import mysql_processor
from Quant.decorator.factor_cal_decorators import rolling_window_decorator,stack_matrix_decorator

import numpy as np
from functools import wraps


class yield_calculator():
    @staticmethod
    def get_yield(data:pd.DataFrame, end_date:str, windows:list):
        end_date_index = data[data['DateTime'] == end_date].index.values[0]
        data = data.sort_values(by='DateTime')

        out_matrix = np.zeros((end_date_index + 1, len(windows)))

        for i in range(end_date_index + 1):

            for j in range(len(windows)):
                
                out_matrix[i, j] = data.loc[i + windows[j] - 1,'Close'] / data.loc[i,'Close'] - 1

        #print(f'yield array \n {out_matrix}')
        #print(data.tail(30))
        
        return out_matrix
    
    @staticmethod
    def get_yield_for_long(data:pd.DataFrame, end_date:str, windows:list, slippage:float):
        end_date_index = data[data['DateTime'] == end_date].index.values[0]
        data = data.sort_values(by='DateTime')

        out_matrix = np.zeros((end_date_index + 1, len(windows)))

        for i in range(end_date_index + 1):

            for j in range(len(windows)):
                
                out_matrix[i, j] = (data.loc[i + windows[j] - 1,'Close'] * (1 - slippage)) / (data.loc[i,'Close'] * (1 + slippage)) - 1

        #print(f'yield array \n {out_matrix}')
        #print(data.tail(30))
        
        return out_matrix
    
    def get_yield_for_short(data:pd.DataFrame, end_date:str, windows:list, slippage:float):
        end_date_index = data[data['DateTime'] == end_date].index.values[0]
        data = data.sort_values(by='DateTime')

        out_matrix = np.zeros((end_date_index + 1, len(windows)))

        for i in range(end_date_index + 1):

            for j in range(len(windows)):
                
                out_matrix[i, j] = (data.loc[i,'Close'] * (1 - slippage)) / (data.loc[i + windows[j] - 1,'Close'] * (1 + slippage)) - 1

        #print(f'yield array \n {out_matrix}')
        #print(data.tail(30))
        
        return out_matrix
    


