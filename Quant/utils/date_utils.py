import numpy as np
import pandas as pd

from datetime import datetime

class datetime_utils:

    def is_valid_datetime_format(date_str):
        try:
            format_str='%Y-%m-%d %H:%M:%S'

            # 尝试解析日期字符串
            datetime.strptime(date_str, format_str)
            return True
        
        except ValueError:
            return False
    
    def get_date_matrix(data:pd.DataFrame, end_date:str):
        end_date_index = data[data['DateTime'] == end_date].index.values[0]
        data = data.sort_values(by='DateTime')

        out_matrix = []

        for i in range(end_date_index + 1):
                
            out_matrix.append(data.loc[i,'DateTime'])
        
        return out_matrix