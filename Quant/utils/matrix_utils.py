import pandas as pd
import numpy as np
import time

class matrix():

    def trans_dataframe_to_matrix(dataframe:pd.DataFrame, column:str, start_date:str):
        'used for calculating factor'
        start_data_index = dataframe[dataframe['DateTime'] == start_date].index.values[0]
        
        data_matrix = np.asarray(dataframe.loc[start_data_index - 1:len(dataframe) - 2,column:column])
        #print('原始dataframe\n',dataframe.loc[start_data_index:,:])
        
        return data_matrix


    @staticmethod
    def trans_dataframe_to_series_window(dataframe:pd.DataFrame, column:str, start_date:str, window:int):
        'transfrom dataframe to multiple matrix'
        start_data_index = dataframe[dataframe['DateTime'] == start_date].index.values[0]
        data_matrix = np.asarray(dataframe.loc[start_data_index - window:len(dataframe) - 1,column:column])

        new_shape = (data_matrix.shape[0] - window, data_matrix.shape[1], window)

        # 计算新视图的步长
        new_strides = (data_matrix.strides[0], 1, data_matrix.strides[0])

        # 创建滑动窗口视图
        data_matrix = np.lib.stride_tricks.as_strided(data_matrix, shape=new_shape, strides=new_strides)

        #print('原始dataframe\n',dataframe.loc[start_data_index - window + 1:,:])
        #print(dataframe.loc[start_data_index - window:start_data_index,:])

        return data_matrix
    
    @staticmethod
    def trans_dataframe_to_frame_window(dataframe:pd.DataFrame, columns:list, start_date:str, window:int):
       # 查找 start_date 在 DataFrame 中的位置
        start_data_index = dataframe[dataframe['DateTime'] == start_date].index.values[0]
        #print('原始dataframe\n', dataframe.loc[start_data_index - window + 1:, :])

        # 选择从 start_date 开始的窗口数据
        dataframe_slice = dataframe.loc[start_data_index - window:len(dataframe) - 2, columns]

        data_matrix = dataframe_slice.values
        out_matrix = np.zeros((len(data_matrix) - window + 1, 1, data_matrix.shape[1] * window))
        for i in range(window - 1, len(data_matrix)):
            out_matrix[i - window + 1, 0, :] = data_matrix[i - window + 1:i + 1, :].flatten()
        
        return out_matrix
