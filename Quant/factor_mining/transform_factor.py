import pandas as pd
import numpy as np

from Quant.utils.matrix_utils import matrix
from Quant.factor_mining.formula import run_function_for_signal_matrix
from Quant.data.mysql_processor import mysql_processor
from Quant.data.data_processor import fill_datanan
from Quant.factor_module.yield_calculator import yield_calculator
from Quant.factor_module.factor_test import factor_test
from Quant.factor_mining.preprocessor import standardize_dataframe, factor_preprocessor

class TransformFactor(factor_test):
    def __init__(self, bitcoin_data:pd.DataFrame, start_date:str, end_date:str):
        self.max_transform = 10
        self.bitcoin_data = bitcoin_data
        self.bitcoin_data = standardize_dataframe(bitcoin_data, 20)
        self.start_date = start_date
        self.end_date = end_date
        self.signal_sign_list = ['square', 'log', 'ln', 'abs', 'negative', 'sqrt', 'cube', 'cbrt', 'inv']
        self.time_series_sign_list = ['delta_sub','delta_add','delta_multiply','delta_divide','decay_linear']
        self.factor_list = []
        super().__init__()

    def create_transform_list(self):
        'create transform list for signal factor'
        columns_list = self.bitcoin_data.columns
        transform_number = np.random.randint(1, self.max_transform)
        column_name = columns_list[np.random.randint(1, len(columns_list))]
        signal_transform_l = [column_name]

        for i in range(transform_number):

            transform_type = np.random.choice(self.signal_sign_list)
            signal_transform_l.append(transform_type)
        
        self.factor_list.append(signal_transform_l)
    
    def transform_data(self, signal_transform_l):
        'transform data according to transform list'
        data_matrix = np.array(self.bitcoin_data[signal_transform_l[0]])
        factor_eva = self.signal_linear_factor_test(np.array(self.bitcoin_data[signal_transform_l[0]]), self.yield_matrix, [1,3,5,10,15,20,25,30,35,40])
        #print(f'corr_eva:\n {factor_eva}')
        for i in range(1, len(signal_transform_l)):
            data_matrix = run_function_for_signal_matrix(data_matrix, signal_transform_l[i])
        return data_matrix

    def run(self):
        yield_data = data_connector.get_data_for_return('BTC',start_date,end_date,40)
        yield_data = fill_datanan(yield_data)
        self.yield_matrix = yield_calculator.get_yield(yield_data, end_date, [1,3,5,10,15,20,25,30,35,40])

        for i in range(10):
            self.create_transform_list()
            factor_matrix = self.transform_data(self.factor_list[i])
            factor_matrix = factor_preprocessor.signal_median_mad_processor(factor_matrix)  
            factor_eva = self.signal_linear_factor_test(factor_matrix, self.yield_matrix,[1,3,5,10,15,20,25,30,35,40])
            #print(f'factor_eva:\n {factor_eva}')

if __name__ == '__main__':
    data_connector = mysql_processor()
    max_window = 20
    start_date = '2023-01-01 00:00:00'
    end_date = '2023-06-02 00:00:00'
    bitcoin_data = data_connector.get_data_for_factor('BTC',start_date,end_date,max_window)
    bitcoin_data = fill_datanan(bitcoin_data)
    transform_factor = TransformFactor(bitcoin_data, start_date, end_date).run()
