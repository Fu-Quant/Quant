import pandas as pd
from re import sub

from Quant.data.mysql_connector import connect_mysql
from Quant.utils.date_utils import datetime_utils
from Quant.decorator.date_decorators import with_db_connection
from Quant.data.data_processor import create_calender
    
class mysql_processor(connect_mysql):
    def __init__(self):
        super().__init__()

    def tranform_types(self,types:list):#将types列表中的数据转换为mysql中的数据类型
        type_mapping = {
            'int': 'INT',
            'integer': 'INT',
            'long': 'BIGINT',
            'float': 'FLOAT',
            'double': 'DOUBLE',
            'str': 'VARCHAR(255)',
            'string': 'VARCHAR(255)',
            'bool': 'BOOLEAN',
            'boolean': 'BOOLEAN',
            'datetime': 'DATETIME',
            'date': 'DATE',
            'time': 'TIME',
            'text': 'TEXT',
            'blob': 'BLOB',
            'object': 'VARCHAR(255)'
        }

        # 转换类型
        mysql_types = []
        types = [sub(r'[^a-zA-Z]','',typ) for typ in types]

        for t in types:
            mysql_type = type_mapping.get(t.lower(), 'VARCHAR(255)')
            mysql_types.append(mysql_type)

        return mysql_types
    
    @with_db_connection
    def create_table(self,table_name:str,columns:list,types:list):
        """
        Creates a table with the given name and columns
        """

        types = self.tranform_types(types)

        create_table_query = f"""
        create table if not exists {table_name}(
        id int auto_increment primary key,
        {','.join([f'{col} {typ}' 
                   for col,typ in zip(columns,types)])}
        );
        """

        self.cursor.execute(create_table_query)
    
    def transform_code(self, coin_code:str):
        if coin_code == 'BTC':
            coin_code = 'BTC1min'
        elif coin_code == 'ETH':
            coin_code = 'ETH1min'
        else:
            raise ValueError('Invalid coin code')
        return coin_code

    @with_db_connection
    def get_data(self,coin_code:str,start_date:str,end_date:str):
        """
        Get data from mysql table
        input parameter:
            coin_code: str
            start_date: str
                format: 'YYYY-MM-DD HH:MM:SS'
            end_date: str
                format: 'YYYY-MM-DD HH:MM:SS'
        output parameter:
            data: pd.DataFrame
        """
        
        # 判断coin_code根据币种匹配table
        coin_code = self.transform_code(coin_code)
        
        if start_date == '':
            start_date = '0000-00-00 00:00:00'
        
        if not datetime_utils.is_valid_datetime_format(start_date):
            raise ValueError('Invalid start date format')
        if not datetime_utils.is_valid_datetime_format(end_date):
            raise ValueError('Invalid end date format')
        
        #sql语句
        query = f"""
        SELECT *
        FROM {coin_code}
        WHERE datetime >= '{start_date}'
        AND datetime <= '{end_date}';
        """

        # 执行查询
        self.cursor.execute(query)
        results = self.cursor.fetchall()
        print('data fetched')

        if results is not None:
            columns = [desc[0] for desc in self.cursor.description]
            bitcoin_data = pd.DataFrame(results,columns=columns).drop(columns=['id'])
            bitcoin_data = bitcoin_data.sort_values(by='DateTime')
            #print(bitcoin_data)
            
        return bitcoin_data
    
    
    def get_data_for_factor(self, coin_code:str, start_date:str, end_date:str, window:int):

        start_date = str(pd.to_datetime(start_date) - pd.Timedelta(minutes=window))

        self.database = 'Bitcoin'
        bitcoin_data = self.get_data(coin_code, start_date, end_date)
        #print(f'get_data_for_factor\n{bitcoin_data}')

        return bitcoin_data
    
    def get_data_for_return(self, coin_code:str, start_date:str, end_date:str, window:int) -> pd.Series:

        end_date = str(pd.to_datetime(end_date) + pd.Timedelta(minutes=window))
        
        self.database = 'Bitcoin'
        bitcoin_data = self.get_data(coin_code, start_date, end_date).iloc[:].reset_index().drop('index', axis=1)
        #print(f'get_data_for_return\n{bitcoin_data}')

        return bitcoin_data
    def get_calender(self, start_date: str, end_date: str):
        "get calender data"
        self.database = 'Calender'
        self.connect_to_mysql()

        query = f"""select * from Calender
            where DateTime >= '{start_date}' and DateTime <= '{end_date}''
            """
        
        self.cursor.execute(query)
        calender_data = self.cursor.fetchall()

        if calender_data is not None:
            columns = [desc[0] for desc in self.cursor.description]
            calender_data = pd.DataFrame(calender_data,columns=columns).drop(columns=['id'])
            calender_data = calender_data.sort_values(by='DateTime')
        
        return calender_data

    @with_db_connection
    def insert_data(self,df:pd.DataFrame,table_name:str):
        """
        Insert data into mysql table
        input parameter:
            df: pd.DataFrame
            table_name: str
        """
        for index,row in df.iterrows():
            placeholders = ','.join(['%s'] * len(row))
            columns = ','.join(row.index)
            insert_query = f"""
            insert into {table_name} ({columns}) values({placeholders})
            """
            self.cursor.execute(insert_query,tuple(row))
        
        self.connection.commit()
        #print(f"Data imported successfully into {table_name}")

    def import_csv_to_mysql(self,csv_file_path:str,table_name:str):
        """
        Import data from csv file to mysql table
        input parameter:
            csv_file_path: str
            table_name: str
        """
        df = pd.read_csv(csv_file_path)
        self.create_table(table_name,df.columns,df.dtypes.astype(str).values)
        self.insert_data(df,table_name)
    
    def create_calender(self, csv_file_path, coin_code: str):

        
        coin_code = self.transform_code(coin_code)
        bitcoin_data = pd.read_csv(csv_file_path)
        calender_data = bitcoin_data.loc[:,'DateTime':'DateTime']
        self.create_table(f'{coin_code}_calender',['DateTime'],['object'])
        self.insert_data(calender_data,f'{coin_code}_calender')
        
if __name__ == '__main__':
    connection = mysql_processor()
    #connection.connect_to_mysql()
    connection.create_calender('/Users/fu/Desktop/量化/Bitcoin/data/1min/BTC.csv','BTC')
    #connection.get_data_for_return('BTC','2023-01-01 00:00:00','2023-01-02 00:00:00',200)
    #connection.close_mysql()



