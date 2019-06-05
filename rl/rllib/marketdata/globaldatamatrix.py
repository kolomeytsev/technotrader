from __future__ import division
from __future__ import absolute_import
from __future__ import print_function
import numpy as np
import pandas as pd
import datetime
import logging
import sys
sys.path.insert(0,'../centurion/')
from common.exchanges.exchange_main import ExchangeDB
from common.etl.models import OHLCVCandle, Exchange, Instrument
from common.etl.helpers import ExtractorCandles, ExtractorOrderbooks, ExtractorTrades, Resolution
from common.calendar_moex import CalendarMOEX
from rllib.tools.data import panel_fillna
from rllib.constants import *


class HistoryManager:
    # if offline ,the coin_list could be None
    # NOTE: return of the sqlite results is a list of tuples, each tuple is a row
    def __init__(self, last_candles_params, coins_list, end, volume_average_days=1, online=True):
        self.exchange_name = last_candles_params[0]
        if self.exchange_name == "moex":
            self.calendar = CalendarMOEX()
        else:
            self.calendar = None
        self.features = last_candles_params[1]
        self.candles_res = last_candles_params[2]
        #self.relevant_columns = last_candles_params[3]
        self.__storage_period = FIVE_MINUTES  # keep this as 300
        self._coin_number = len(coins_list)
        self._online = online
        self.__volume_average_days = volume_average_days
        self.__coins = coins_list
        #self.__coins_with_base = [
        #    coin + "/BTC" for coin in self.__coins
        #]
        self.exchange = ExchangeDB()
        print(self.__coins)
        """print("self.exchange_name:", self.exchange_name)
        print("self.features:", self.features)
        print("self.candles_res:", self.candles_res)
        print("self.relevant_columns:", self.relevant_columns)
        print("self._coin_number:", self._coin_number)
        print("self._online:", self._online)
        print("self.__volume_average_days:", self.__volume_average_days)
        print("self.__coins:", self.__coins)
        exit(1)"""

    @property
    def coins(self):
        return self.__coins

    def get_global_data_matrix(self, start, end, period=300, features=('close',)):
        """
        :return a numpy ndarray whose axis is [feature, coin, time]
        """
        return self.get_global_panel(start, end, period, features).values

    def put_in_panel(self, panel, candles, epoch, features):
        index = datetime.datetime.fromtimestamp(epoch)
        for candle in candles:
            instr = candle[2].name
            for feature in features:
                panel.loc[feature, instr, index] = getattr(candle[0], feature, None)

    def get_global_panel(self, start, end, period=300, features=('close',), test_portion=None):
        """
        :param start/end: linux timestamp in seconds
        :param period: time interval of each data access point
        :param features: tuple or list of the feature names
        :return a panel, [feature, coin, time]
        """
        start = int(start - (start%period))
        end = int(end - (end%period))
        n_periods = (end - start) // period + 1
        if test_portion is not None:
            train_sec = int((end - start) * (1 - test_portion))
        else:
            train_sec = int(end - start)
        end_train_date = datetime.datetime.fromtimestamp(train_sec + start).strftime("%Y-%m-%d")
        start_train_date =  datetime.datetime.fromtimestamp(start).strftime("%Y-%m-%d")
        end_test_date = datetime.datetime.fromtimestamp(end).strftime("%Y-%m-%d")
        if test_portion is not None:
            print("train:\tfrom %s till %s" % (start_train_date, end_train_date))
            print("test:\tfrom %s till %s" % (end_train_date, end_test_date))
        coins = self.__coins
        #self.__checkperiod(period)
        dates = list(range(start, end + 1, period))
        if self.calendar is not None:
            dates = list(filter(lambda x: self.calendar[x], dates))
        time_index = pd.to_datetime(dates, unit='s')
        panel = pd.Panel(items=features, major_axis=coins, minor_axis=time_index, dtype=np.float32)
        for index in panel.minor_axis:
            epoch = int(index.timestamp())
            ret = self.exchange.session.query(OHLCVCandle, Exchange, Instrument).join(Exchange).join(Instrument).filter(
                    OHLCVCandle.mts == epoch * 1000,
                    Instrument.name.in_(coins),
                    OHLCVCandle.resolution == period,
                    Exchange.name == self.exchange_name
                ).all()    
            self.put_in_panel(panel, ret, epoch, features)
        dates = list(range(start, end + 1, period))
        panel.minor_axis = dates[:panel.shape[2]]
        for feature in self.features:
            print("Nans number for %s: %d" % (feature, panel[feature].isna().sum().sum()))
            panel[feature].fillna(axis=1, method="ffill", inplace=True)
            panel[feature].fillna(axis=1, method="bfill", inplace=True)
        print("panel:", panel)
        print("shape:", panel.shape)
        return panel

    def transform_to_timestamp(datetime_time):
        return int(datetime.datetime.strptime(datetime_time, "%Y-%m-%d %H:%M:%S").timestamp())

    def __checkperiod(self, period):
        if period == FIVE_MINUTES:
            return
        elif period == TEN_MINUTES:
            return
        elif period == FIFTEEN_MINUTES:
            return
        elif period == HALF_HOUR:
            return
        elif period == HOUR:
            return
        elif period == TWO_HOUR:
            return
        elif period == FOUR_HOUR:
            return
        elif period == DAY:
            return
        else:
            raise ValueError('peroid has to be 5min, 15min, 30min, 2hr, 4hr, or a day')
