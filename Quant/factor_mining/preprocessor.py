import numpy as np

def standardize_matrix(data_matrix:np.array, window:int):##未检查
    dimensions = data_matrix.ndim
    
    new_shape = (data_matrix.shape[0] - window + 1, data_matrix.shape[1], window)

    # 计算新视图的步长
    new_strides = (data_matrix.strides[0], data_matrix.strides[1], data_matrix.strides[0])

    # 创建滑动窗口视图
    rolling_data_matrix = np.lib.stride_tricks.as_strided(data_matrix, shape=new_shape, strides=new_strides)
    mean_matrix = np.mean(rolling_data_matrix, axis=2)
    standardize_matrix = np.std(rolling_data_matrix, axis=2)
    data_matrix = (data_matrix - mean_matrix)/standardize_matrix

    return data_matrix

def standardize_dataframe(dataframe, window):
    columns = dataframe.columns.tolist()
    if 'DateTime' in columns:
        new_dataframe = dataframe.drop(columns='DateTime')
        columns.remove('DateTime')
    else:
        new_dataframe = dataframe.copy()

    mean_dataframe = new_dataframe.rolling(window=window).mean()
    #print(mean_dataframe)
    std_dataframe = new_dataframe.rolling(window=window).std()#数据和numpy算的不一样
    #print(std_dataframe)
    new_dataframe = (new_dataframe - mean_dataframe) / (std_dataframe + 1e-8)
    dataframe.loc[:,columns] = new_dataframe.loc[:,columns]
    dataframe = dataframe.iloc[window:]
    #print(f'Dataframe normalized: {dataframe}')

    return dataframe

class factor_preprocessor:

    @staticmethod
    def signal_median_mad_processor(data_matrix:np.array):
        median_value = np.median(data_matrix)
        mad = np.median(np.abs(data_matrix - median_value))
        #print(median_value,mad,data_matrix - median_value)
        data_matrix[data_matrix > median_value + 3 * 1.4826 * mad] = median_value + 3 * 1.4826 * mad
        data_matrix[data_matrix < median_value - 3 * 1.4826 * mad] = median_value - 3 * 1.4826 * mad
        return data_matrix
