import pandas as pd
import numpy as np
import time

from Quant.data.mysql_processor import mysql_processor
from Quant.factor_mining.genetics import genetics_algo
from Quant.data.data_processor import fill_datanan
from Quant.factor_module.yield_calculator import yield_calculator
from Quant.data.data_processor import create_calender
from Quant.factor_module.linear_factor_module.linear_factor_processor import processor
from Quant.evaluation.evaluation import evaluator
from Quant.utils.date_utils import datetime_utils

from sklearn.linear_model import LinearRegression

class backtest_module():
    def __init__(self, start_date, end_date, frequency, formula_tree, window) -> None:
        self.start_date = start_date
        self.end_date = end_date
        self.window = window
        self.formula_tree = formula_tree
        self.frequency = frequency
        self.coin = 'BTC'
        self.slippage = 0.00001
        self.commission = (0.0005)/10
    
    def create_module(self):
        self.data_module = mysql_processor()
        self.genetics_module = genetics_algo()
        self.evaluator = evaluator()

    def create_matrix_for_backtest(self):
        self.create_module()
        start_date = str(pd.to_datetime(self.start_date) - pd.Timedelta(minutes=self.window))
        factor_matrix = self.genetics_module.get_factor_matrix(start_date, self.end_date, self.formula_tree)
        #factor_dataframe = pd.DataFrame(factor_matrix)
        #factor_dataframe.to_csv('/Users/fu/Desktop/factor_test.csv',index=False)

        yield_data = self.data_module.get_data_for_return(self.coin,start_date, self.end_date,10)
        yield_data = fill_datanan(yield_data)
        #yield_data.to_csv('/Users/fu/Desktop/yield_test.csv',index=False)
        yield_matrix = yield_calculator.get_yield(yield_data, end_date=self.end_date, windows = [10])
        long_yield_matrix = yield_calculator.get_yield_for_long(yield_data, end_date=self.end_date, windows = [10], slippage=self.slippage)
        short_yield_matrix = yield_calculator.get_yield_for_short(yield_data, end_date=self.end_date, windows = [10], slippage=self.slippage)
        date_matrix = datetime_utils.get_date_matrix(yield_data, end_date=self.end_date)

        train_data_matrix = np.zeros((int((len(factor_matrix) - self.window) / self.frequency), 1, self.window, factor_matrix.shape[1]))
        predict_data_matrix = np.zeros((int((len(factor_matrix) - self.window) / self.frequency), factor_matrix.shape[1]))
        yield_train_data_matrix = np.zeros((int((len(factor_matrix) - self.window) / self.frequency), 1, self.window))
        yield_test_data_matrix = np.zeros((int((len(factor_matrix) - self.window) / self.frequency), 1))
        long_yield_test_data_matrix = np.zeros((int((len(factor_matrix) - self.window) / self.frequency), 1))
        short_yield_test_data_matrix = np.zeros((int((len(factor_matrix) - self.window) / self.frequency), 1))
        date_test_matrix = []

        for i in range(0, len(factor_matrix) - self.window - 1, self.frequency):
            
            train_data_matrix[int(i / self.frequency), 0, :, :] = factor_matrix[i:i + self.window, :]
            predict_data_matrix[int(i / self.frequency), :] = factor_matrix[i + self.window, :]
            yield_train_data_matrix[int(i / self.frequency), 0, :] = yield_matrix[i:i + self.window, 0]
            yield_test_data_matrix[int(i / self.frequency), 0] = yield_matrix[i + self.window, 0]
            long_yield_test_data_matrix[int(i / self.frequency), 0] = long_yield_matrix[i + self.window, 0]
            short_yield_test_data_matrix[int(i / self.frequency), 0] = short_yield_matrix[i + self.window, 0]
            date_test_matrix.append(date_matrix[i + self.window])

        return train_data_matrix, yield_train_data_matrix, predict_data_matrix, yield_test_data_matrix, long_yield_test_data_matrix, short_yield_test_data_matrix, date_test_matrix

    def linear_predictor(self,train_factor_matrix, yield_train_matrix, test_factor_matrix):
        train_factor_matrix, test_factor_matrix = processor.pca_for_rolling_backtest(train_factor_matrix, test_factor_matrix)
        if test_factor_matrix.ndim == 1:
            test_factor_matrix = test_factor_matrix.reshape(-1, test_factor_matrix.shape[0])
        model = LinearRegression()
        
        # 训练模型
        model.fit(train_factor_matrix, yield_train_matrix)
        
        # 进行预测
        predicted_yield = model.predict(test_factor_matrix)
        
        return predicted_yield

    def rolling_backtest(self):
        result_dict = []
        train_data_matrix, yield_train_data_matrix, predict_data_matrix, yield_test_data_matrix, long_yield_test_data_matrix, short_yield_test_data_matrix, date_test_matrix = self.create_matrix_for_backtest()
        
        for i in range(len(predict_data_matrix)):
            train_factor_matrix = train_data_matrix[i, 0, :]
            test_factor_matrix = predict_data_matrix[i]
            yield_train_matrix = yield_train_data_matrix[i]
            yield_train_matrix = yield_train_matrix.reshape(yield_train_matrix.shape[1],1)
            
            date_test = date_test_matrix[i]
            yield_test = yield_test_data_matrix[i].flatten()
            long_yield = long_yield_test_data_matrix[i].flatten()
            short_yield = short_yield_test_data_matrix[i].flatten()
            single_factor = self.linear_predictor(train_factor_matrix, yield_train_matrix, test_factor_matrix)

            if single_factor > 0:
                result_dict.append({
                    'datetime':date_test,
                    'yield': yield_test[0],
                    'yield_including_slippage':long_yield[0],
                    'yield_including_commission': yield_test[0] - self.commission,
                    'yield_including_commission_and_slippage':long_yield[0] - self.commission,
                    'direction': 'long'
                })
            elif single_factor < 0:
                result_dict.append({
                    'datetime':date_test,
                    'yield': -yield_test[0],
                    'yield_including_slippage':short_yield[0],
                    'yield_including_commission': - yield_test[0] - self.commission,
                    'yield_including_commission_and_slippage': short_yield[0] - self.commission,
                    'direction': 'short'
                })

        self.evaluator.eva(result_dict, self.start_date, self.end_date)


if __name__ == '__main__':
    start_date = '2023-01-01 00:00:00'
    end_date = '2023-06-02 00:00:00'
    backtest = backtest_module(start_date, end_date, 10, [[0,'add','Close','Close']], 240)
    backtest.rolling_backtest()