import pandas as pd
import numpy as np

from scipy.stats import spearmanr, pearsonr
import matplotlib.pyplot as plt

class factor_test():
    def signal_linear_factor_test(self, factor_array:np.array, yield_array:np.array, windows:list) -> pd.DataFrame:
        '单个因子的多周期测试'
        eva_dict = []
        if factor_array.ndim == 2:
            factor_array = factor_array.flatten()
        if yield_array.ndim == 2:
            for i in range(yield_array.shape[1]):
                signal_yield_array = yield_array[:,i].flatten()
                RankIC, RankIC_pvalue = spearmanr(factor_array, signal_yield_array)
                IC, IC_pvalue = pearsonr(factor_array, signal_yield_array)
                r_squared = self.linear_regression_matrix(factor_array, signal_yield_array)
                eva_dict.append({'index': f'future {windows[i]}minutes',
                                 'RankIC': RankIC,
                                 'RankIC_pvalue': RankIC_pvalue,
                                 'IC': IC,
                                 'IC_pvalue': IC_pvalue,
                                 'R_squared': r_squared})
        return pd.DataFrame(eva_dict)
    
    def multiple_linear_factor_eva(self, factor_array:np.array, yield_array:np.array):

        eva_dict = []

        if yield_array.ndim == 2:
            yield_array = yield_array.flatten()

        for i in range(factor_array.shape[1]):
            factor_array_1 = factor_array[:,i]

            if factor_array_1.ndim == 2:
                factor_array_1 = factor_array_1.flatten()

            RankIC, RankIC_pvalue = spearmanr(factor_array_1, yield_array)
            IC, IC_pvalue = pearsonr(factor_array_1, yield_array)
            eva_dict.append({ 'RankIC': RankIC,
                                 'RankIC_pvalue': RankIC_pvalue,
                                 'IC': IC,
                                 'IC_pvalue': IC_pvalue})
            
        return pd.DataFrame(eva_dict)

    def linear_regression_matrix(self, x, y):
        n = len(x)

        # 计算必要的统计量
        sum_x = np.sum(x)
        sum_y = np.sum(y)
        sum_xy = np.sum(x * y)
        sum_x_squared = np.sum(x**2)
        
        # 计算斜率 a
        a_numerator = n * sum_xy - sum_x * sum_y
        a_denominator = n * sum_x_squared - sum_x**2
        a = a_numerator / a_denominator
        
        # 计算截距 b
        b = (sum_y - a * sum_x) / n
        y_pred = a * x + b
        r_squared = self.calculate_r_squared(y, y_pred)
        return r_squared
    
    def calculate_r_squared(self, y, y_pred):
        # 计算实际值的平均值
        y_mean = np.mean(y)
        
        # 计算总平方和 SST
        sst = np.sum((y - y_mean)**2)
        
        # 计算残差平方和 SSE
        sse = np.sum((y - y_pred)**2)
        
        # 计算 R 方
        r_squared = 1 - (sse / sst)
        
        return r_squared

class plot:
    def plot_heatmap(self, data_matrix: np.array):
        plt.imshow(data_matrix, cmap='viridis', interpolation='nearest')

        # 添加颜色条
        plt.colorbar()

        # 添加坐标轴标签
        plt.xlabel('Columns')
        plt.ylabel('Rows')

        # 添加标题
        plt.title('Heatmap of 2D Array')

        # 显示图形
        plt.show()