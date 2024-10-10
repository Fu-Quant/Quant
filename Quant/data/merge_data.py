from Quant.data.mysql_processor import mysql_processor
import pandas as pd
import numpy as np

class merge_data(mysql_processor):
    def __init__(self) -> None:
        super().__init__()
    def get_merge_data(self, coin_codes:list, start_date:str, end_date:str):

        bitcoin_datas = pd.DataFrame()

        for coin_code in coin_codes:
            bitcoin_data = self.get_data(coin_code, start_date, end_date)
            new_columns = {
                'DateTime': 'DateTime',
                'Open': f'{coin_code}_Open',
                'High': f'{coin_code}_High',
                'Low': f'{coin_code}_Low',
                'Close': f'{coin_code}_Close',
                'Volume': f'{coin_code}_Volume'
            }
            bitcoin_data = bitcoin_data.rename(columns = new_columns)

            if not bitcoin_datas.empty:
                if not bitcoin_datas['DateTime'].equals(bitcoin_data['DateTime']):
                    raise Exception('DateTime is not same')
                else:
                    bitcoin_data = bitcoin_data.drop(columns = 'DateTime')
                
            bitcoin_datas = pd.concat([bitcoin_datas, bitcoin_data], axis = 1)

        #print(bitcoin_datas)
    
        return bitcoin_datas
    
    def get_multiple_frequency_data(self, coin_code, start_date, end_date, frequency):
        "get multiple frequency data"
        bitcoin_data = self.get_data_for_factor(coin_code, start_date, end_date, frequency)
        data_matrix = np.asarray(bitcoin_data)
        new_shape = (data_matrix.shape[0] - frequency + 1, data_matrix.shape[1], frequency)

         # 计算新视图的步长
        new_strides = (data_matrix.strides[0], data_matrix.strides[1], data_matrix.strides[0])

        # 创建滑动窗口视图
        data_matrix = np.lib.stride_tricks.as_strided(data_matrix, shape=new_shape, strides=new_strides)
        dtype = [('DateTime','object'),('Open','float64'),('High','float64'),
                 ('Low','float64'),('Close','float64'),('Volume','float64')]
        out_matrix = np.zeros((data_matrix.shape[0], 1), dtype=dtype)

        high_matrix = data_matrix[:, 2, :]
        low_matrix = data_matrix[:, 3, :]
        volume_matrix = data_matrix[:, 5, :]

        out_matrix['DateTime'] = data_matrix[:, 0, -1].reshape(-1, 1)       
        out_matrix['Open'] = data_matrix[:, 1, 0].reshape(-1, 1)
        out_matrix['High'] = np.max(high_matrix, axis=1).reshape(-1, 1)
        out_matrix['Low'] = np.min(low_matrix, axis=1).reshape(-1, 1)
        out_matrix['Close'] = data_matrix[:, 4, -1].reshape(-1, 1)
        out_matrix['Volume'] = np.sum(volume_matrix, axis=1).reshape(-1, 1)

        field_names = out_matrix.dtype.names
        columns = []

        for filed_name in field_names:
            columns.append(out_matrix[filed_name])

        out_matrix = np.column_stack(columns)
        out_DataFrame = pd.DataFrame(out_matrix, columns=['DateTime', 'Open', 'High', 'Low', 'Close', 'Volume'])
        #print(out_DataFrame)

        return out_DataFrame
    
if __name__ == '__main__':
    merge_data = merge_data()
    merge_data.get_multiple_frequency_data('BTC', '2021-01-01 00:00:00', '2021-01-02 00:00:00',5)
    