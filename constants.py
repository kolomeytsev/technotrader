from os import path


DATABASE_DIR = path.realpath(__file__).\
    replace('technotrader/constants.pyc','technotrader/database/Data.db').\
    replace("technotrader\\constants.pyc","technotrader/database\\Data.db").\
    replace('technotrader/constants.py','technotrader/database/Data.db').\
    replace("technotrader\\constants.py","technotrader/database\\Data.db")
CONFIG_FILE_DIR = "net_config.json"
LAMBDA = 1e-4  # lambda in loss function 5 in training


NOW = 0
FIVE_MINUTES = 60 * 5
FIFTEEN_MINUTES = FIVE_MINUTES * 3
HALF_HOUR = FIFTEEN_MINUTES * 2
HOUR = HALF_HOUR * 2
TWO_HOUR = HOUR * 2
FOUR_HOUR = HOUR * 4
EIGHT_HOUR = HOUR * 8
TEN_HOUR = HOUR * 10
TWELVE_HOUR = HOUR * 12
DAY = HOUR * 24
YEAR = DAY * 365
   # trading table name
TABLE_NAME = 'test'
