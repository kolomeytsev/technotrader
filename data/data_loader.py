import sys
sys.path.append('../')
import numpy as np
from datetime import timedelta, datetime
import pandas as pd


class DataLoader:
    """
    Driver which connects to database and performs data requests
    """
    def __init__(self, config):
        pass

    def last_candles(self, exchanges_instruments, n_candles, candles_resolution=None,
                    indicators=None, fields=None, step=None, out_format='pandas_wide'):
        return candles
