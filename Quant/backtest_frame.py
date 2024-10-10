from Quant.factor_mining.genetics import genetics_algo
from Quant.factor_module.linear_factor_module.linear_factor_processor import processor
from Quant.factor_module.factor_test import factor_test
from Quant.data.mysql_processor import mysql_processor
from Quant.factor_module.yield_calculator import yield_calculator
from Quant.data.data_processor import fill_datanan
from Quant.factor_module.backtest_module import backtest_module

def frame():
    genetics_module = genetics_algo()
    start_date = '2023-01-01 00:00:00'
    end_date = '2023-06-02 00:00:00'
    formula_trees = genetics_module.run(start_date, end_date,num_trees = 1000)
    factor_matrix,formula_trees = genetics_module.concat_array(formula_trees)
    print(factor_matrix)

    #pca
    '''pca_matrix = processor.pca(factor_matrix)
    print(pca_matrix)
    factor_test_module = factor_test()
    data_connector = mysql_processor()
    yield_data = data_connector.get_data_for_return('BTC', start_date, end_date, 10)
    yield_data = fill_datanan(yield_data)
    yield_matrix = yield_calculator.get_yield(yield_data, end_date='2023-06-02 00:00:00', windows = [10])
    eva = factor_test_module.multiple_linear_factor_eva(pca_matrix, yield_matrix)
    print(eva)
    residual_matrix = processor.extract_nonpca_factors(factor_matrix, pca_matrix)
    eva = factor_test_module.multiple_linear_factor_eva(residual_matrix, yield_matrix)
    print(eva)
    correlation_residual_matrix = processor.calculate_correlation_matrix(residual_matrix)
    correlation_pca_matrix = processor.calculate_correlation_matrix(pca_matrix)

    processor.plot_heatmap(correlation_residual_matrix)
    processor.plot_heatmap(correlation_pca_matrix)'''
    start_date = '2023-06-02 00:00:00'
    end_date = '2023-12-31 00:00:00'
    backtest = backtest_module(start_date, end_date, 10, formula_trees, 240)
    backtest.rolling_backtest()

frame()
