import numpy as np
import pandas as pd
import re 
import time

from Quant.data.mysql_processor import mysql_processor
from Quant.utils.matrix_utils import matrix
from Quant.factor_mining.formula import run_function_for_signal_matrix, run_function_for_double_matrix
from Quant.factor_mining.preprocessor import standardize_dataframe, factor_preprocessor
from Quant.data.data_processor import fill_datanan
from Quant.factor_module.yield_calculator import yield_calculator
from Quant.factor_module.factor_test import factor_test

class formula_tree():
    def __init__(self, start_date:str, end_date:str, bitcoin_data:pd.DataFrame, max_window:int):
        self.start_date = start_date
        self.end_date = end_date
        self.bitcoin_data = bitcoin_data
        #self.bitcoin_data = standardize_dataframe(bitcoin_data, max_window)
        self.sign_list = ['add', 'subcontract', 'multiply', 'divide']
        self.series_sign_list = ['max', 'min', 'sum', 'mean', 'std']
        self.time_series_sign_list = ['delta_sub','delta_add','delta_multiply','delta_divide',
                             'decay_linear']
        self.signal_sign_list = ['square', 'log', 'ln', 'abs', 'negative', 'sign', 'sqrt', 'cube', 'cbrt', 'inv']
        self.column_names = self.bitcoin_data.columns[1:]
        self.formula = ''
        self.max_window = max_window
    
    def fill_formula_list(self, now_index:int, value:str):
        if now_index > len(self.formula_list) - 1:
            target_length = int(2 ** np.ceil(np.log2(now_index + 1)) - 1)
            self.formula_list = self.formula_list + [0] * (target_length - len(self.formula_list) + 1)
        
        self.formula_list[int(now_index)] = value
        
    def sign_fun(self, mode:int, now_index:int, direction:str):
        select_sign_or_value = np.random.rand(1)
        if select_sign_or_value < 0.5:#选择sign
            select_time_or_signal = np.random.rand(1)
            if select_time_or_signal < 0.5 or mode == 2:#选择time
                random_time = np.random.randint(1, len(self.time_series_sign_list))
                value = self.time_series_sign_list[random_time]
                data_matrix = self.evaluate(value, now_index, direction)
                data_matrix = run_function_for_signal_matrix(data_matrix, self.time_series_sign_list[random_time])
                return data_matrix
            elif select_time_or_signal >= 0.5:#选择signal
                random_sign = np.random.randint(0, len(self.signal_sign_list))
                value = self.signal_sign_list[random_sign]
                data_matrix = self.evaluate(value, now_index, direction)
                data_matrix = run_function_for_signal_matrix(data_matrix, self.signal_sign_list[random_sign])
                return data_matrix
        elif select_sign_or_value >= 0.5:#选择value
            if mode == 1:
                random_value = np.random.randint(0, len(self.column_names))
                value = self.column_names[random_value]
                self.fill_formula_list(value, now_index)
                self.formula += value
                data_matrix = matrix.trans_dataframe_to_matrix(self.bitcoin_data, self.column_names[random_value], self.start_date)
                return data_matrix

            elif mode == 2:
                size = np.random.randint(1, len(self.column_names))
                interge_list = np.arange(len(self.column_names))
                random_values = np.random.choice(interge_list, size, replace=False)
                select_time = np.random.randint(1, self.max_window)
                self.formula += ','.join(self.column_names[random_values] + f'({select_time})')
                self.fill_formula_list(','.join(self.column_names[random_values] + f'({select_time})'), now_index)
                data_matrix = matrix.trans_dataframe_to_frame_window(self.bitcoin_data, self.column_names[random_values], self.start_date, select_time)
                return data_matrix
            
    def evaluate(self, value, last_index:int, direction:str):
         if direction == 'left':
             self.fill_formula_list(value, last_index)
             now_index = 2 * last_index
         elif direction == 'right':
             self.fill_formula_list(value, last_index)
             now_index = 2 * last_index + 1

         if value in self.sign_list:
            self.formula += value + '(('
            data_matrix_1 = self.sign_fun(1, now_index, 'left')
            self.formula += '),('
            data_matrix_2 = self.sign_fun(1, now_index, 'right')
            self.formula += '))'
            data_matrix = run_function_for_double_matrix(data_matrix_1, data_matrix_2, value)
            return data_matrix
         elif value in self.series_sign_list:
            self.formula += value + '(('
            data_matrix_1 = self.sign_fun(2, now_index, 'left')  
            data_matrix = run_function_for_signal_matrix(data_matrix_1, value)              
            self.formula += ')'
            return data_matrix
         elif value in self.time_series_sign_list:
             self.formula += value + '('
             select_columns = np.random.randint(0, len(self.column_names))
             select_time = np.random.randint(2, self.max_window)
             select_sign_or_not = np.random.rand(1)
             data_matrix = matrix.trans_dataframe_to_series_window(self.bitcoin_data, self.column_names[select_columns], self.start_date, select_time)
             if select_sign_or_not < 0.5:
                select_signal = np.random.randint(0, len(self.signal_sign_list))
                value = self.signal_sign_list[select_signal] + '(' + self.column_names[select_columns] + f'({select_time})' + ')'
                data_matrix = run_function_for_signal_matrix(data_matrix, self.signal_sign_list[select_signal])
             elif select_sign_or_not >= 0.5:
                value = self.column_names[select_columns] + f'({select_time})'
             self.formula += value +')'
             self.fill_formula_list(value, now_index)
             return data_matrix
         elif value in self.signal_sign_list:
             self.formula += value + '('
             select_column = np.random.randint(0, len(self.column_names))
             select_signal = np.random.randint(0, len(self.signal_sign_list))
             value = self.signal_sign_list[select_signal] + f'({self.column_names[select_column]})'
             self.formula += value + ')'
             self.fill_formula_list(value, now_index)
             data_matrix = matrix.trans_dataframe_to_matrix(self.bitcoin_data, self.column_names[select_column], self.start_date)
             data_matrix = run_function_for_signal_matrix(data_matrix, self.signal_sign_list[select_signal])
             return data_matrix
         
    def construct_formula_tree(self, formula_value: str, last_index: int, direction: int):
        left_now_index = 2 * last_index
        right_now_index = 2 * last_index + 1
        
        if formula_value in self.sign_list:
            selection = np.random.randint(1, 3)

            if selection == 1:
                formula_value = self.series_sign_formula_fun(left_now_index)
                self.construct_formula_tree(formula_value, left_now_index, 'left')

                selection = np.random.randint(1, 3)
                if selection == 1:
                    formula_value = self.time_series_sign_formula_fun(right_now_index)
                elif selection == 2:
                    formula_value = self.signal_sign_formula_fun(right_now_index, True)

                self.construct_formula_tree(formula_value, right_now_index, 'right')

            elif selection == 2:
                formula_value = self.signal_sign_formula_fun(left_now_index, False)
                self.construct_formula_tree(formula_value, left_now_index, 'left')
                
                selection = np.random.randint(1, 3)
                if selection == 1:
                    formula_value = self.time_series_sign_formula_fun(right_now_index)
                elif selection == 2:
                    formula_value = self.signal_sign_formula_fun(right_now_index, True)

                self.construct_formula_tree(formula_value, right_now_index, 'right')
        
        if formula_value in self.series_sign_list:

            selection = np.random.randint(1, 3)
            if selection == 1:
                formula_value = self.time_series_sign_formula_fun(left_now_index)
            elif selection == 2:
                formula_value = self.signal_sign_formula_fun(left_now_index, False)    
                self.construct_formula_tree(formula_value, left_now_index, 'left')

            selection = np.random.randint(1, 3)
            if selection == 1:
                formula_value = self.time_series_sign_formula_fun(right_now_index)
            elif selection == 2:
                formula_value = self.signal_sign_formula_fun(right_now_index, False)           
                self.construct_formula_tree(formula_value, right_now_index, 'left')
        
        if formula_value in self.signal_sign_list:
            pro = np.random.rand(1)

            if pro < 0.5:
                formula_value = self.signal_sign_formula_fun(left_now_index, False)
                self.construct_formula_tree(formula_value, left_now_index, 'left')
            else:
                self.signal_sign_formula_fun(left_now_index, True)
            
    def series_sign_formula_fun(self, now_index):
        selection_index = np.random.randint(0, len(self.series_sign_list))
        self.fill_formula_list(now_index, self.series_sign_list[selection_index])

        return self.series_sign_list[selection_index]

    def sign_formula_fun(self,now_index):
        selection_index = np.random.randint(0, len(self.sign_list))
        self.fill_formula_list(now_index, self.sign_list[selection_index])

        return self.sign_list[selection_index]

    def time_series_sign_formula_fun(self, now_index):
        selection_index = np.random.randint(0, len(self.time_series_sign_list))
        time_length = str(np.random.randint(2, 20))
        formula_value = self.time_series_sign_list[selection_index]
        self.fill_formula_list(now_index, formula_value)

        formula_value = self.column_names[np.random.randint(0, len(self.column_names))] + f'({time_length})'
        self.fill_formula_list(now_index * 2, formula_value)

        return formula_value

    def signal_sign_formula_fun(self, now_index, is_end):
        if is_end == True:
            selection_index = np.random.randint(0, len(self.signal_sign_list))
            self.fill_formula_list(now_index, self.signal_sign_list[selection_index])

            selection_index = np.random.randint(0, len(self.column_names))
            self.fill_formula_list(now_index * 2, self.column_names[selection_index])

            return self.column_names[selection_index]
        
        elif is_end == False:
            selection_index = np.random.randint(0, len(self.signal_sign_list))
            self.fill_formula_list(now_index, self.signal_sign_list[selection_index])
        
            return self.signal_sign_list[selection_index]


    def cal_formula_tree(self, formula_list:list, route_list:list, data_matrix:np.array, now_index:int):
        next_left_index = now_index * 2 
        next_right_index = now_index * 2 + 1
        
        if next_left_index < len(route_list) - 1:

            if route_list[next_left_index] == 0 and formula_list[next_left_index] != 0:
                #遍历左结点
                data_matrix_1 = self.cal_formula_tree(formula_list, route_list, None, next_left_index)
        
        if next_right_index < len(route_list) - 1:

            if route_list[next_right_index] == 0 and formula_list[next_right_index] != 0:
  
                #遍历右结点
                data_matrix_2 = self.cal_formula_tree(formula_list, route_list, None, next_right_index)
        
        formula_value = re.sub(r'[^a-zA-Z]', '',formula_list[now_index])
        if formula_value in self.column_names:
            time_length = re.sub(r'[^\d+]', '', formula_list[now_index])
            if len(time_length) > 0:
                time_length = int(time_length)
                data_matrix = matrix.trans_dataframe_to_series_window(self.bitcoin_data, formula_value, self.start_date, time_length)
            else:
                data_matrix = matrix.trans_dataframe_to_matrix(self.bitcoin_data, formula_list[now_index], self.start_date)

        elif formula_list[now_index] in self.series_sign_list or \
            formula_list[now_index] in  self.time_series_sign_list or \
            formula_list[now_index] in self.signal_sign_list:
            data_matrix = run_function_for_signal_matrix(data_matrix_1, formula_list[now_index])

        elif formula_list[now_index] in self.sign_list:
            data_matrix = run_function_for_double_matrix(data_matrix_1, data_matrix_2, formula_list[now_index])

        route_list[now_index] = 1

        return data_matrix
    def build_formula_tree(self):
        self.formula_list = [0]
        selection_index = np.random.randint(0, len(self.sign_list))
        value = self.sign_list[selection_index]
        self.construct_formula_tree(value, 1, None)
        self.fill_formula_list(1, value)
        return self.formula_list
    
    def calculate_formula_tree(self, formula_list:list):
        route_list = [0] * (len(formula_list) + 1)
        data_matrix = self.cal_formula_tree(formula_list, route_list, None, 1)

        return data_matrix

if __name__ == '__main__':
    data_connector = mysql_processor()
    max_window = 20
    windows = [1,3,5,10,15,20,25,30,35,40]
    start_date = '2023-01-01 00:00:00'
    end_date = '2023-01-02 00:00:00'
    bitcoin_data = data_connector.get_data_for_factor('BTC',start_date,end_date,int(max_window * 2))
    bitcoin_data = fill_datanan(bitcoin_data)
    yield_data = data_connector.get_data_for_return('BTC',start_date,end_date,40)
    yield_data = fill_datanan(yield_data)
    yield_matrix = yield_calculator.get_yield(yield_data, end_date, windows)
    formula_generator = formula_tree(start_date,end_date,bitcoin_data, max_window)
    #foumula_list = formula_generator.build_formula_tree()
    #data_matrix = formula_generator.calculate_formula_tree(foumula_list)
    #factor_matrix = factor_preprocessor.signal_median_mad_processor(data_matrix)
    #factor_test_module = factor_test()
    #factor_eva = factor_test_module.signal_linear_factor_test(factor_matrix, yield_matrix, windows)
    #print(f'factor_eva:\n {factor_eva}')

    formula = [0, 'multiply', 'max', 'ln', 'cbrt', 'delta_multiply', 'Volume', 0, 'abs', 0, 'Close(18)', 0, 0, 0, 0, 0, 'Volume', 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    m = formula_generator.calculate_formula_tree(formula)
    print(m)