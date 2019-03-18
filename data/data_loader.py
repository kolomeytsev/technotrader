import sys
sys.path.append('../')
import numpy as np
import pandas as pd
import logging
import time
import calendar
from datetime import timedelta, datetime
import technotrader.data.database_manager as database_manager


def parse_time(time_string):
    return time.mktime(datetime.strptime(time_string, "%Y/%m/%d").timetuple())


class DataLoader:
    """
    Driver which connects to database and performs data requests
    """
    def __init__(self, config):
        #data_matrices = DataMatrices.create_from_config(config)
        logging.info("initializing data")
        self.start = config["begin"]
        self.end = config["end"]
        self.exchange = config["exchange"]
        self.instruments_list = config["instruments_list"]
        self.candles_res = config["candles_res"]
        self.period = config["candles_res_sec"]
        self.features = ["open", "high", "low", "close", "volume"]
        self.database_manager = database_manager.DatabaseManager(self.instruments_list)
        if self.exchange == "poloniex":
            self.global_data = self.database_manager.get_global_panel(
                self.start,
                self.end,
                period=self.period,
                features=self.features
            )
        else:
            raise ValueError("Exchange {} is not available".format(self.exchange))
        self.data = self.transform_panel_to_dict(self.global_data, self.exchange, self.candles_res)

    def transform_panel_to_dict(self, panel, exchange, resolution):
        instruments = panel.major_axis
        dates = panel.minor_axis
        features = panel.items
        result = {}
        for date in dates:
            ts = calendar.timegm(date.utctimetuple())
            #ts = int(datetime.timestamp(date))
            date_data = {}
            for asset in instruments:
                key_base = '>'.join([exchange, asset, resolution])
                for k in features:
                    key = f'{key_base}>{k}'
                    date_data[key] = panel[k][date][asset]
            result[ts] = date_data
        return result

    def get_data(self, epochs):
        return {epoch: self.data[epoch] for epoch in epochs}
