import numpy as np
import pandas as pd
from technotrader.trading.constants import *
import sqlite3
from datetime import datetime
import logging


def panel_fillna(panel, type="bfill"):
    """
    fill nan along the 3rd axis
    :param panel: the panel to be filled
    :param type: bfill or ffill
    """
    frames = {}
    for item in panel.items:
        if type == "both":
            frames[item] = panel.loc[item].fillna(axis=1, method="bfill").\
                fillna(axis=1, method="ffill")
        else:
            frames[item] = panel.loc[item].fillna(axis=1, method=type)
    return pd.Panel(frames)


class DatabaseManager:
    def __init__(self, instruments_list):
        self.initialize_db()
        self.storage_period = FIVE_MINUTES  # keep this as 300
        self.instruments_list = instruments_list
        self.instruments_list_no_quote = [asset.split("/")[0] for asset in instruments_list]
        self.quote = instruments_list[0].split("/")[1]
        self._coin_list = self.instruments_list_no_quote
        self._coin_number = len(self._coin_list)

    def initialize_db(self):
        with sqlite3.connect(DATABASE_DIR) as connection:
            cursor = connection.cursor()
            cursor.execute('CREATE TABLE IF NOT EXISTS History (date INTEGER,'
                           ' coin varchar(20), high FLOAT, low FLOAT,'
                           ' open FLOAT, close FLOAT, volume FLOAT, '
                           ' quoteVolume FLOAT, weightedAverage FLOAT,'
                           'PRIMARY KEY (date, coin));')
            connection.commit()

    def get_global_data_matrix(self, start, end, period=300, features=('close',)):
        """
        :return a numpy ndarray whose axis is [feature, coin, time]
        """
        return self.get_global_panel(start, end, period, features).values

    def get_global_panel(self, start, end, period=300, features=('close',)):
        """
        :param start/end: linux timestamp in seconds
        :param period: time interval of each data access point
        :param features: tuple or list of the feature names
        :return a panel, [feature, coin, time]
        """
        start = int(start)
        end = int(end)
        coins = self.instruments_list_no_quote
        self.coins = coins
        for coin in coins:
            self.update_data(start, end, coin)

        logging.info("feature type list is %s" % str(features))
        self.checkperiod(period)
        time_index = pd.to_datetime(list(range(start, end + 1, period)), unit='s')
        panel = pd.Panel(items=features, major_axis=self.instruments_list, minor_axis=time_index, dtype=np.float32)
        connection = sqlite3.connect(DATABASE_DIR)
        try:
            for row_number, coin in enumerate(coins):
                for feature in features:
                    # NOTE: transform the start date to end date
                    if feature == "close":
                        sql = ("SELECT date+300 AS date_norm, close FROM History WHERE"
                               " date_norm>={start} and date_norm<={end}" 
                               " and date_norm%{period}=0 and coin=\"{coin}\"".format(
                               start=start, end=end, period=period, coin=coin))
                    elif feature == "open":
                        sql = ("SELECT date+{period} AS date_norm, open FROM History WHERE"
                               " date_norm>={start} and date_norm<={end}" 
                               " and date_norm%{period}=0 and coin=\"{coin}\"".format(
                               start=start, end=end, period=period, coin=coin))
                    elif feature == "volume":
                        sql = ("SELECT date_norm, SUM(volume)"+
                               " FROM (SELECT date+{period}-(date%{period}) "
                               "AS date_norm, volume, coin FROM History)"
                               " WHERE date_norm>={start} and date_norm<={end} and coin=\"{coin}\""
                               " GROUP BY date_norm".format(
                                    period=period,start=start,end=end,coin=coin))
                    elif feature == "high":
                        sql = ("SELECT date_norm, MAX(high)" +
                               " FROM (SELECT date+{period}-(date%{period})"
                               " AS date_norm, high, coin FROM History)"
                               " WHERE date_norm>={start} and date_norm<={end} and coin=\"{coin}\""
                               " GROUP BY date_norm".format(
                                    period=period,start=start,end=end,coin=coin))
                    elif feature == "low":
                        sql = ("SELECT date_norm, MIN(low)" +
                                " FROM (SELECT date+{period}-(date%{period})"
                                " AS date_norm, low, coin FROM History)"
                                " WHERE date_norm>={start} and date_norm<={end} and coin=\"{coin}\""
                                " GROUP BY date_norm".format(
                                    period=period,start=start,end=end,coin=coin))
                    else:
                        msg = ("The feature %s is not supported" % feature)
                        logging.error(msg)
                        raise ValueError(msg)
                    serial_data = pd.read_sql_query(sql, con=connection,
                                                    parse_dates=["date_norm"],
                                                    index_col="date_norm")
                    panel.loc[feature, coin + "/" + self.quote, serial_data.index] = serial_data.squeeze()
                    panel = panel_fillna(panel, "both")
        finally:
            connection.commit()
            connection.close()
        return panel

    def checkperiod(self, period):
        if period == FIVE_MINUTES:
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
        elif period == EIGHT_HOUR:
            return
        elif period == TEN_HOUR:
            return
        elif period == TWELVE_HOUR:
            return
        elif period == DAY:
            return
        else:
            raise ValueError('peroid has to be 5min, 15min, 30min, 2hr, 4hr, 8hr, 10hr, 12hr or a day')

    # add new history data into the database
    def update_data(self, start, end, coin):
        connection = sqlite3.connect(DATABASE_DIR)
        try:
            cursor = connection.cursor()
            min_date = cursor.execute('SELECT MIN(date) FROM History WHERE coin=?;', (coin,)).fetchall()[0][0]
            max_date = cursor.execute('SELECT MAX(date) FROM History WHERE coin=?;', (coin,)).fetchall()[0][0]

            if min_date is None or max_date is None:
                self.fill_data(start, end, coin, cursor)
            else:
                if max_date + 10 * self.storage_period < end:
                    if not self._online:
                        raise Exception("Have to be online")
                    self.fill_data(max_date + self.storage_period, end, coin, cursor)
                if min_date > start and self._online:
                    self.fill_data(start, min_date - self.storage_period-1, coin, cursor)

            # if there is no data
        finally:
            connection.commit()
            connection.close()

    def fill_data(self, start, end, coin, cursor):
        chart = self._coin_list.get_chart_until_success(
            pair=self._coin_list.allActiveCoins.at[coin, 'pair'],
            start=start,
            end=end,
            period=self.storage_period)
        logging.info("fill %s data from %s to %s"%(
            coin, 
            datetime.utcfromtimestamp(start).strftime('%Y-%m-%d %H:%M'),
            datetime.utcfromtimestamp(end).strftime('%Y-%m-%d %H:%M'))
        )
        for c in chart:
            if c["date"] > 0:
                if c['weightedAverage'] == 0:
                    weightedAverage = c['close']
                else:
                    weightedAverage = c['weightedAverage']

                #NOTE here the USDT is in reversed order
                if 'reversed_' in coin:
                    cursor.execute('INSERT INTO History VALUES (?,?,?,?,?,?,?,?,?)',
                        (c['date'], coin, 1.0 / c['low'], 1.0 / c['high'], 1.0 / c['open'],
                        1.0 / c['close'], c['quoteVolume'], c['volume'],
                        1.0 / weightedAverage))
                else:
                    cursor.execute('INSERT INTO History VALUES (?,?,?,?,?,?,?,?,?)',
                                   (c['date'], coin,c['high'], c['low'], c['open'],
                                    c['close'], c['volume'], c['quoteVolume'],
                                    weightedAverage))
