import sys
sys.path.append('../')
import numpy as np
import pandas as pd
import logging
import time
import calendar
from datetime import timedelta, datetime
import technotrader.data_loader.database_manager as database_manager


def parse_time(time_string):
    return time.mktime(datetime.strptime(time_string, "%Y/%m/%d").timetuple())


class DataLoader:
    """
    Driver which connects to database and performs data requests
    """
    def __init__(self, config, data=None):
        logging.info("initializing data")
        if config.get("type") is not None:
            self.type = config["type"]
        else:
            self.type = "exchange"
        self.start = config["begin"]
        self.end = config["end"]
        self.exchange = config["exchange"]
        self.instruments_list = config["instruments_list"]
        self.candles_res = config["candles_res"]
        self.period = config["candles_res_sec"]
        self.features = ["open", "high", "low", "close", "volume"]
        if self.exchange == "poloniex":
            self.database_manager = database_manager.DatabaseManager(self.instruments_list)
            self.global_data = self.database_manager.get_global_panel(
                self.start,
                self.end,
                period=self.period,
                features=self.features
            )
            self.data = self.transform_panel_to_dict(self.global_data, self.exchange, self.candles_res)
        elif self.type == "csv":
            self.init_data = data
            self.data = self.transform_df_to_dict(self.init_data, self.exchange, self.candles_res)
        else:
            raise ValueError("Exchange of data type {} is not available".format(self.exchange))

    def transform_df_to_dict(self, df, exchange, resolution):
        instruments = df.columns
        indices = df.index
        feature = "close"
        result = {}
        for index in indices:
            date_data = {}
            for asset in instruments:
                key_base = '>'.join([exchange, asset, resolution])
                key = f'{key_base}>{feature}'
                date_data[key] = df[asset][index]
            result[index] = date_data
        return result

    def transform_panel_to_dict(self, panel, exchange, resolution):
        instruments = panel.major_axis
        dates = panel.minor_axis
        features = panel.items
        result = {}
        for date in dates:
            ts = calendar.timegm(date.utctimetuple())
            date_data = {}
            for asset in instruments:
                key_base = '>'.join([exchange, asset, resolution])
                for k in features:
                    key = f'{key_base}>{k}'
                    date_data[key] = panel[k][date][asset]
            result[ts] = date_data
        return result

    def get_data(self, epochs):
        if any([x < 0 for x in epochs]):
            raise ValueError("epoch must be positive")
        return {epoch: self.data[epoch] for epoch in epochs}
