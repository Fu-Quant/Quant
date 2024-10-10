import time
import numpy as np

from functools import wraps

def rolling_window_decorator(func):
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        start_time = time.time()
        # 保证数据按日期排序
        self.data = self.data.iloc[self.start_date-self.window + 1:]
        self.data = self.data.sort_values(by='DateTime').drop(columns='DateTime')
        self.data_matrix = np.asarray(self.data)
        
        # 创建输出矩阵
        out_matrix = np.zeros_like(self.data_matrix)

        for i in range(self.window - 1, len(self.data_matrix)):
            if i + 1 <= len(self.data_matrix) - 1:
                self.sub_data_matrix = self.data_matrix[i - self.window + 1:i + 1]
                out_matrix[i] = func(self, *args, **kwargs)
                #print(out_matrix)
                #print(out_matrix[i])
                #time.sleep(10)
            elif i + 1 == len(self.data_matrix):
                self.sub_data_matrix = self.data_matrix[i - self.window + 1:]
                out_matrix[i] = func(self, *args, **kwargs)
        
        out_matrix = out_matrix[self.window - 1:]
        end_time = time.time()
        elapsed_time = end_time - start_time  # 计算耗时
        #print(f"Elapsed time: {elapsed_time:.6f} seconds")
        #print('因子矩阵: \n',out_matrix)
        #print(out_matrix.shape)
        return out_matrix
    
    return wrapper

def stack_matrix_decorator(func):
    @wraps(func)
    def wrapper(self,*args, **kwargs):
        start_time = time.time()
        self.start_date_index = self.data[self.data['DateTime'] == self.start_date].index.values[0]
        self.data = self.data.iloc[self.start_date_index - self.window + 1:]
        print(self.data)
        self.data = self.data.sort_values(by='DateTime').drop(columns='DateTime')
        self.data_matrix = np.asarray(self.data)

        new_shape = (self.data_matrix.shape[0] - self.window + 1, self.data_matrix.shape[1], self.window)

        # 计算新视图的步长
        new_strides = (self.data_matrix.strides[0], self.data_matrix.strides[1], self.data_matrix.strides[0])

        # 创建滑动窗口视图
        self.data_matrix = np.lib.stride_tricks.as_strided(self.data_matrix, shape=new_shape, strides=new_strides)

        out_matrix = func(self, *args, **kwargs)

        end_time = time.time()
        elapsed_time = end_time - start_time  # 计算耗时
        #print(f"Elapsed time: {elapsed_time:.6f} seconds")
        #print('因子矩阵: \n',out_matrix)
        #print(out_matrix.shape)
        return out_matrix
    
    return wrapper
            


