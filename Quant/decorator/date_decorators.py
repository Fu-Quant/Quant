from functools import wraps
from Quant.data.mysql_connector import connect_mysql

from numba.np.numpy_support import as_dtype
from numba import types

import numpy as np
def with_db_connection(func):
    
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        #try:
        self.connect_to_mysql()
        result = func(self, *args, **kwargs)
        #finally:
        self.close_mysql()
        return result
    return wrapper



