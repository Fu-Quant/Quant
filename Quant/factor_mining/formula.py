import numpy as np
import pandas as pd

from Quant.data.mysql_processor import mysql_processor

class formula():

    @staticmethod
    def add(data_matrix_1:np.array, data_matrix_2:np.array):
        return data_matrix_1 + data_matrix_2
    
    @staticmethod
    def subcontract(data_matrix_1:np.array, data_matrix_2:np.array):
        return data_matrix_1 - data_matrix_2
    
    @staticmethod
    def multiply(data_matrix_1:np.array, data_matrix_2:np.array):
        return data_matrix_1 * data_matrix_2
    
    @staticmethod
    def divide(data_matrix_1:np.array, data_matrix_2:np.array):
        return data_matrix_1 / (data_matrix_2 + 1e-8)
    
    @staticmethod
    def max(data_matrix_1:np.array):
        if data_matrix_1.ndim == 3:
            return np.max(data_matrix_1, axis = 2)
        return data_matrix_1
    
    @staticmethod
    def min(data_matrix_1:np.array):
        if data_matrix_1.ndim == 3:
            return np.min(data_matrix_1, axis = 2)
        return data_matrix_1
    
    @staticmethod
    def sum(data_matrix_1:np.array):
        if data_matrix_1.ndim == 3:
            return np.sum(data_matrix_1, axis = 2)
        return data_matrix_1
    
    @staticmethod
    def mean(data_matrix_1:np.array):
        if data_matrix_1.ndim == 3:
            return np.mean(data_matrix_1, axis = 2)
        return data_matrix_1
    
    @staticmethod
    def std(data_matrix_1:np.array):
        if data_matrix_1.ndim == 3:
            return np.std(data_matrix_1, axis = 2)
        return data_matrix_1
    
    @staticmethod
    def delta_sub(data_matrix_1:np.array):
        return data_matrix_1[:,:,0] - data_matrix_1[:,:,-1]
    
    @staticmethod
    def delta_add(data_matrix_1:np.array):
        return data_matrix_1[:,:,0] + data_matrix_1[:,:,-1]
    
    @staticmethod
    def delta_multiply(data_matrix_1:np.array):
        return data_matrix_1[:,:,0] * data_matrix_1[:,:,-1]
    
    @staticmethod
    def delta_divide(data_matrix_1:np.array):
        return data_matrix_1[:,:,0] / (data_matrix_1[:,:,-1] + 1e-8)
    
    @staticmethod
    def decay_linear(data_matrix_1:np.array):
        length = data_matrix_1.shape[2]
        for i in range(length):
            data_matrix_1[:,:,i] = (length - i) * data_matrix_1[:,:,i]
        data_matrix_1 = np.sum(data_matrix_1, axis=2)
        data_matrix_1 = data_matrix_1 /((length + 1) * length/2 + 1e-8)
        return data_matrix_1
    
    @staticmethod
    def square(data_matrix_1:np.array):
        return data_matrix_1 ** 2
    
    @staticmethod
    def log(data_matrix_1:np.array):
        return np.log(np.abs(data_matrix_1))
    
    @staticmethod
    def ln(data_matrix_1:np.array):
        return np.log(np.abs(data_matrix_1))/(np.log(2) + 1e-8)
    
    @staticmethod
    def abs(data_matrix_1:np.array):
        return np.abs(data_matrix_1)
    
    @staticmethod
    def negative(data_matrix_1:np.array):
        return -data_matrix_1
    
    @staticmethod
    def sign(data_matrix_1:np.array):
        return np.sign(data_matrix_1)
    
    @staticmethod
    def sqrt(data_matrix_1:np.array):
        return np.sqrt(np.abs(data_matrix_1))
    
    @staticmethod
    def cube(data_matrix_1:np.array):
        return data_matrix_1**3
    
    @staticmethod
    def cbrt(data_matrix_1:np.array):
        return np.abs(data_matrix_1)**(1/3)
    
    @staticmethod
    def inv(data_matrix_1:np.array):
        return 1/(data_matrix_1 + 1e-8)
    
def run_function_for_signal_matrix(data_matrix_1:np.array, function_name:str):
    # 创建信号字典
    # 创建信号字典
    signal_dict = {
        'add': formula.add,
        'subcontract': formula.subcontract,
        'multiply': formula.multiply,
        'divide': formula.divide,
        'max': formula.max,
        'min': formula.min,
        'sum': formula.sum,
        'mean': formula.mean,
        'std': formula.std,
        'delta_sub': formula.delta_sub,
        'delta_add': formula.delta_add,
        'delta_multiply': formula.delta_multiply,
        'delta_divide': formula.delta_divide,
        'decay_linear': formula.decay_linear,
        'square': formula.square,
        'log': formula.log,
        'ln': formula.ln,
        'abs': formula.abs,
        'negative': formula.negative,
        'sign': formula.sign,
        'sqrt': formula.sqrt,
        'cube': formula.cube,
        'cbrt': formula.cbrt,
        'inv': formula.inv
    }

    data_matrix = signal_dict[function_name](data_matrix_1)
    if np.any(np.isnan(data_matrix)):
        raise ValueError('Invalid data of NaN')

    return data_matrix

def run_function_for_double_matrix(data_matrix_1:np.array, data_matrix_2:np.array, function_name:str):
    # 创建信号字典  
    signal_dict = {
        'add': formula.add,
        'subcontract': formula.subcontract,
        'multiply': formula.multiply,
        'divide': formula.divide,
    }
    
    data_matrix = signal_dict[function_name](data_matrix_1, data_matrix_2)
    return data_matrix
