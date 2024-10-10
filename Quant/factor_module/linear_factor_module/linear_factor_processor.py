from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
class processor:
    @staticmethod
    def pca(data_matrix:np.array):
        
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(data_matrix)
        for i in range(2,10):
            pca = PCA(n_components=i)
            transform_data = pca.fit_transform(scaled_data)
            variance_ratios = pca.explained_variance_ratio_
            cumulative_variance = np.cumsum(variance_ratios)
            if cumulative_variance[-1] > 0.99:
                break

        return transform_data
    
    def pca_for_rolling_backtest(data_matrix_1:np.array, data_matrix_2:np.array):
        if data_matrix_2.ndim == 1:
            data_matrix_2 = data_matrix_2.reshape(-1,data_matrix_2.shape[0])
        data_matrix = np.concatenate((data_matrix_1, data_matrix_2), axis=0)
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(data_matrix)
        for i in range(2,7):
            pca = PCA(n_components=i)
            transform_data = pca.fit_transform(scaled_data)
            variance_ratios = pca.explained_variance_ratio_
            cumulative_variance = np.cumsum(variance_ratios)
            if cumulative_variance[-1] > 0.999:
                break

        transform_train_data = transform_data[:-1]
        test_data = transform_data[-1]

        return transform_train_data, test_data
        
    
    @staticmethod
    def extract_nonpca_factors(factor_matrix:np.array, pca_matrix:np.array):
        scaler = StandardScaler()
        factor_matrix = scaler.fit_transform(factor_matrix)
        num_columns = factor_matrix.shape[1]
        residuals = []
                
        for i in range(num_columns):
            # 使用多项式特征作为自变量，每一列作为因变量训练模型
            lr = LinearRegression()
            lr.fit(pca_matrix, factor_matrix[:, i])
            
            # 预测每一列
            predictions = lr.predict(pca_matrix)
            
            # 计算残差
            residual = factor_matrix[:, i] - predictions
                  
            residuals.append(residual)
        
        # 将残差转换为二维数组
        residuals_array = np.array(residuals).T
        
        return residuals_array
    
    @staticmethod
    def calculate_correlation_matrix(data_matrix: np.array):
        correlation_matrix = np.corrcoef(data_matrix.T)
        return correlation_matrix
    
    @staticmethod
    def plot_heatmap(correlation_matrix):
        plt.figure(figsize=(10, 8))
        sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="coolwarm", square=True, cbar_kws={"shrink": .5})
        plt.title("Correlation Heatmap")
        plt.show()
