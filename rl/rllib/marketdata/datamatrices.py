from __future__ import print_function
from __future__ import absolute_import
from __future__ import division
import numpy as np
import pandas as pd
import logging
import datetime

from rllib.tools.configprocess import parse_time
import rllib.marketdata.replaybuffer as rb
import rllib.marketdata.globaldatamatrix as gdm

MIN_NUM_PERIOD = 3


class DataMatrices:
    def __init__(self, last_candles_params, start, end, period, batch_size=50, volume_average_days=30, buffer_bias_ratio=0,
                 market="binance", coins_list=[], window_size=50, feature_number=3, test_portion=0.15,
                 portion_reversed=False, online=False, is_permed=False, data_path=None):
        """
        :param start: Unix time
        :param end: Unix time
        :param access_period: the data access period of the input matrix.
        :param trade_period: the trading period of the agent.
        :param global_period: the data access period of the global price matrix.
                              if it is not equal to the access period, there will be inserted observations
        :param coins_list: coins that would be selected
        :param window_size: periods of input data
        :param train_portion: portion of training set
        :param is_permed: if False, the sample inside a mini-batch is in order
        :param validation_portion: portion of cross-validation set
        :param test_portion: portion of test set
        :param portion_reversed: if False, the order to sets are [train, validation, test]
        else the order is [test, validation, train]
        """
        self.__start = int(start)
        self.__end = int(end)

        self.__coin_no = len(coins_list)
        #self.__features = ["close", "high", "low"]
        self.__features = last_candles_params[1]
        self.feature_number = feature_number
        self.coins_list = coins_list
        self.__history_manager = gdm.HistoryManager(last_candles_params=last_candles_params, 
                                                    coins_list=coins_list, end=self.__end,
                                                    volume_average_days=volume_average_days, online=online)
        self.get_train_test_dates(self.__start, self.__end, period=period, test_portion=test_portion)
        if data_path is None:
            self.__global_data = self.__history_manager.get_global_panel(self.__start, self.__end,
                                                                    period=period, features=self.__features,
                                                                    test_portion=test_portion)
        else:
            self.__global_data = self.read_data(data_path, period=period, features=self.__features)
        self.dates = self.__global_data.minor_axis
        print("global_data shape: ", self.__global_data.shape)
        self.__period_length = period
        # portfolio vector memory, [time, assets]
        self.__PVM = pd.DataFrame(index=self.__global_data.minor_axis,
                                  columns=self.__global_data.major_axis)
        self.__PVM = self.__PVM.fillna(1.0 / self.__coin_no)

        self._window_size = window_size
        
        self._num_periods = len(self.__global_data.minor_axis)
        self.__divide_data(test_portion, portion_reversed)

        self._portion_reversed = portion_reversed
        self.__is_permed = is_permed

        self.__batch_size = batch_size
        self.__delta = 0  # the count of global increased
        end_index = self._train_ind[-1]
        self.__replay_buffer = rb.ReplayBuffer(start_index=self._train_ind[0],
                                                       end_index=end_index,
                                                       sample_bias=buffer_bias_ratio,
                                                       batch_size=self.__batch_size,
                                                       coin_number=self.__coin_no,
                                                       is_permed=self.__is_permed)
        
        logging.info("the number of training examples is %s"
                    ", of test examples is %s" % (self._num_train_samples, self._num_test_samples))
        logging.debug("the training set is from %s to %s" % (min(self._train_ind), max(self._train_ind)))
        logging.debug("the test set is from %s to %s" % (min(self._test_ind), max(self._test_ind)))

    def get_train_test_dates(self, start, end, period=300, test_portion=None):
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
        self.end_train_date = datetime.datetime.fromtimestamp(train_sec + start).strftime("%Y-%m-%d")
        self.start_train_date =  datetime.datetime.fromtimestamp(start).strftime("%Y-%m-%d")
        self.end_test_date = datetime.datetime.fromtimestamp(end).strftime("%Y-%m-%d")

    def check_nans_after_date(self, cur_df, date):
        assert cur_df[cur_df.index > date].isna().sum().sum() == 0, "error, test data has Nans"

    def convert_to_panel(self, df, companies, features, start_ts=None, end_ts=None):
        all_dates = sorted(set(df["minor"]))
        all_dates = np.array(all_dates)
        if start_ts is not None:
            start = datetime.datetime.fromtimestamp(start_ts).strftime("%Y-%m-%d %H-%M-%S")
            all_dates = all_dates[all_dates > start]
        if end_ts is not None:
            end = datetime.datetime.fromtimestamp(end_ts).strftime("%Y-%m-%d %H-%M-%S")
            all_dates = all_dates[all_dates < end]
        time_index = pd.to_datetime(all_dates)
        panel = pd.Panel(items=features, major_axis=companies, minor_axis=time_index, dtype=np.float32)
        for company in companies:
            df_company = df[df["major"] == company]
            df_company = df_company.set_index(pd.to_datetime(df_company["minor"]))
            for feat in features:
                df_to_fill = df_company.loc[time_index][feat]
                self.check_nans_after_date(df_to_fill, self.end_train_date)
                #print("Nans number for %s, %s: %d" % (company, feat, df_to_fill.isna().sum().sum()))
                if feat == "volume":
                    df_to_fill[df_to_fill == 0] = 1e-6
                panel.loc[feat, company, time_index] = df_to_fill.fillna(method="ffill").fillna(method="bfill")
        return panel

    def check_period(self, panel, period):
        panel_delta = (panel.minor_axis[1] - panel.minor_axis[0]).seconds
        print("periods: ", panel_delta, period)
        assert panel_delta == period, "data period and config period does not match"

    def convert_to_panel_combined(self, df, companies, start_ts=None, end_ts=None):
        df.index = pd.to_datetime(df["Unnamed: 0"])
        all_dates = sorted(set(df["Unnamed: 0"]))
        all_dates = np.array(all_dates)
        if start_ts is not None:
            start = datetime.datetime.fromtimestamp(start_ts).strftime("%Y-%m-%d %H-%M-%S")
            all_dates = all_dates[all_dates > start]
        if end_ts is not None:
            end = datetime.datetime.fromtimestamp(end_ts).strftime("%Y-%m-%d %H-%M-%S")
            all_dates = all_dates[all_dates < end]
        time_index = pd.to_datetime(all_dates)
        
        panel = pd.Panel(items=["close"], major_axis=companies, minor_axis=time_index, dtype=np.float32)
        for company in companies:
            df_to_fill = df.loc[time_index, company]
            self.check_nans_after_date(df_to_fill, self.end_train_date)
            panel.loc["close", company, time_index] = df_to_fill.fillna(method="ffill").fillna(method="bfill")
        return panel

    def read_data(self, data_path, period, features):
        print("Reading data:", self.__start, self.__end, period, features)
        df = pd.read_csv(data_path)
        if "major" not in df.columns:
            panel = self.convert_to_panel_combined(df, self.coins_list, self.__start, self.__end)
        else:
            panel = self.convert_to_panel(df, self.coins_list, features, self.__start, self.__end)
        self.check_period(panel, period)
        return panel

    @property
    def global_weights(self):
        return self.__PVM

    @staticmethod
    def create_from_config(config, last_candles_params):
        """
        main method to create the DataMatrices in this project
        @:param config: config dictionary
        @:return: a DataMatrices object
        """
        config = config.copy()
        input_config = config["input"]
        train_config = config["training"]
        start = parse_time(input_config["start_date"])
        end = parse_time(input_config["end_date"])
        return DataMatrices(last_candles_params=last_candles_params,
                            start=start,
                            end=end,
                            market=input_config["market"],
                            feature_number=input_config["feature_number"],
                            window_size=input_config["window_size"],
                            online=input_config["online"],
                            period=input_config["global_period"],
                            coins_list=input_config["coins_list"],
                            is_permed=input_config["is_permed"],
                            buffer_bias_ratio=train_config["buffer_biased"],
                            batch_size=train_config["batch_size"],
                            volume_average_days=input_config["volume_average_days"],
                            test_portion=input_config["test_portion"],
                            portion_reversed=input_config["portion_reversed"],
                            data_path=input_config.get("data_path")
                            )

    @property
    def global_matrix(self):
        return self.__global_data

    @property
    def coin_list(self):
        return self.__history_manager.coins

    @property
    def num_train_samples(self):
        return self._num_train_samples

    @property
    def test_indices(self):
        return self._test_ind[:-(self._window_size+1):]

    @property
    def num_test_samples(self):
        return self._num_test_samples

    def append_experience(self, online_w=None):
        """
        :param online_w: (number of assets + 1, ) numpy array
        Let it be None if in the backtest case.
        """
        self.__delta += 1
        self._train_ind.append(self._train_ind[-1]+1)
        appended_index = self._train_ind[-1]
        self.__replay_buffer.append_experience(appended_index)

    def get_test_set(self):
        return self.__pack_samples(self.test_indices)

    def get_test_dates(self):
        return self.dates[-len(self.test_indices):]

    def get_training_set(self):
        return self.__pack_samples(self._train_ind[:-self._window_size])

    def next_batch(self):
        """
        @:return: the next batch of training sample. The sample is a dictionary
        with key "X"(input data); "y"(future relative price); "last_w" a numpy array
        with shape [batch_size, assets]; "w" a list of numpy arrays list length is
        batch_size
        """
        batch = self.__pack_samples([exp.state_index for exp in self.__replay_buffer.next_experience_batch()])
        return batch

    def __pack_samples(self, indexs):
        indexs = np.array(indexs)
        last_w = self.__PVM.values[indexs-1, :]

        def setw(w):
            self.__PVM.iloc[indexs, :] = w
        M = [self.get_submatrix(index) for index in indexs]
        M = np.array(M)
        X = M[:, :, :, :-1]
        y = M[:, :, :, -1] / M[:, 0, None, :, -2]
        return {"X": X, "y": y, "last_w": last_w, "setw": setw}

    # volume in y is the volume in next access period
    def get_submatrix(self, ind):
        return self.__global_data.values[:, :, ind:ind+self._window_size+1]

    def __divide_data(self, test_portion, portion_reversed):
        train_portion = 1 - test_portion
        s = float(train_portion + test_portion)
        if portion_reversed:
            portions = np.array([test_portion]) / s
            portion_split = (portions * self._num_periods).astype(int)
            indices = np.arange(self._num_periods)
            self._test_ind, self._train_ind = np.split(indices, portion_split)
        else:
            portions = np.array([train_portion]) / s
            portion_split = (portions * self._num_periods).astype(int)
            indices = np.arange(self._num_periods)
            self._train_ind, self._test_ind = np.split(indices, portion_split)

        self._train_ind = self._train_ind[:-(self._window_size + 1)]
        # change the logic here in order to fit both
        # reversed and normal version
        self._train_ind = list(self._train_ind)
        self._num_train_samples = len(self._train_ind)
        self._num_test_samples = len(self.test_indices)
