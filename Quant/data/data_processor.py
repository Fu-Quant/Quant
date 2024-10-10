import pandas as pd
import numpy as np
def create_calender(start:str, end:str, start_window:int, end_window:int) -> pd.DataFrame:
    'create dataframe with time range'
    if start_window >= 0:
        start = pd.to_datetime(start) + pd.Timedelta(minutes=start_window)
    elif start_window < 0:
        start = pd.to_datetime(start) - pd.Timedelta(minutes=abs(start_window))      
    if end_window >= 0:
        end = pd.to_datetime(end) + pd.Timedelta(minutes=end_window)
    elif end_window < 0:
        end = pd.to_datetime(end) - pd.Timedelta(minutes=abs(end_window))
    time_list = pd.date_range(start=start, end=end, freq='T')
    data_frame = pd.DataFrame({'DateTime': time_list})
    data_frame['DateTime'] = data_frame['DateTime'].astype(str)
    return data_frame

def fill_datanan(data_frame:pd.DataFrame) -> pd.DataFrame:
    'merge dataframe with date dataframe and fill nan values'
    date_dataframe = create_calender(data_frame.loc[0,'DateTime'], data_frame.loc[len(data_frame) - 1, 'DateTime'], 0, 0)
    data_frame = pd.merge(date_dataframe, data_frame, on='DateTime', how='left')
    nan_frame = data_frame[data_frame.isna().any(axis=1)]
    #print(f'show nan dataframe {nan_frame}')
    data_frame['Volume'] = data_frame['Volume'].fillna(0)
    data_frame = data_frame.fillna(method = 'ffill')
    #print(f'Data fillna complete {data_frame}')
    return data_frame

