import mysql.connector
from mysql.connector import Error

class connect_mysql:
    def __init__(self):
        self.host = 'localhost'
        self.user = 'root'
        self.password = 'Fujiarong123!'
        self.database = 'Calender'
    
    def connect_to_mysql(self):

        self.connection = mysql.connector.connect(host=self.host,
                                             user=self.user,
                                             password=self.password,
                                             database=self.database)# Connect to MySQL
        try:
            if self.connection.is_connected():# Check if connected
                self.cursor = self.connection.cursor()# Create cursor
                print('Connected to databse successfully')

        except Error as e:
            print("Error while connecting to MySQL", e)
    
    def close_mysql(self):
        """
        Closes the connection to the MySQL database
        """
        self.connection.close()