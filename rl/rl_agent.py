import pandas as pd
import numpy as np
import os
import json

import rllib.tools.configprocess as configprocess
from rllib.learn.rollingtrainer import RollingTrainer


class RLAgent:
    """
    Agent based on Reinforcement Learning
    """
    def __init__(self, last_portfolio):
        self.net_config = configprocess.load_config("rl_configs/test_net_config.json")
        self.instruments_list = self.net_config["input"]["coins_list"]
        #instruments_list = []
        #for bases in self.bases_list:
        #    instruments_list.append(bases + "/BTC")
        #self.instruments_list = instruments_list
        self.instruments_number = len(self.instruments_list)

        self.net_dir = "train_package/1/netfile"

        self.candles_res = "hour,1"
        #self.candles_res_formatted = self.candles_res.replace(',', '')
        self.features = self.net_config["input"]["features"]

        self.exchange_name = "moex"
        self.relevant_columns = []
        for price_label in self.features:
            str_addition = '>' + self.candles_res + '>' + price_label
            relevant_columns_for_price_label = [self.exchange_name + '>' + x + str_addition \
                                                   for x in self.instruments_list]
            self.relevant_columns.append(relevant_columns_for_price_label)
        print("self.relevant_columns:")
        print(self.relevant_columns)
        print("net_config:\n", self.net_config)
        self.last_candles_params = (
            self.exchange_name,
            self.features,
            self.candles_res,
            self.relevant_columns
        )
        self.initialize_rl_agent(last_portfolio)
        self.weights_projection = agent_utils.WeightsProjection(config)
        timetable = agent_utils.ExchangeTimetable(config["exchange"])
        self.data_extractor = agent_utils.DataExtractor(data_loader, 
                                            timetable, config, self.net_config["input"]["window"], True)

    def initialize_rl_agent(self, last_portfolio):
        self.n_steps = 0
        self.hist_omegas = []
        self.hist_last_omegas = []
        self.hist_future_prices = []
        self._rolling_trainer = RollingTrainer(self.net_config, self.last_candles_params, 
                                                self.net_dir, agent=None)
        self._coin_name_list = self._rolling_trainer.coin_list
        self._norm_method = self.net_config["input"]["norm_method"]
        self._agent = self._rolling_trainer.agent

        self._window_size = self.net_config["input"]["window_size"]
        self._coin_number = self.net_config["input"]["coin_number"]
        self._commission_rate = self.net_config["trading"]["trading_consumption"]
        self._fake_ratio = self.net_config["input"]["fake_ratio"]
        self._asset_vector = np.zeros(self._coin_number + 1)
        self._last_omega = np.array(last_portfolio)
        self._omega = np.array(last_portfolio)

    def predict_rl(self, data):
        weight = self._rolling_trainer.decide_by_history(data, self._last_omega)
        return weight / weight.sum()

    def compute_portfolio(self, epoch):
        self.n_steps += 1
        data_prices = self.data_extractor(epoch)
        future_prices = data_prices[0,:,-1] / data_prices[0,:,-2]
        future_prices = np.insert(future_prices, 0, 1)
        if self.n_steps >= 2:
            self._last_omega = self._omega * future_prices / \
                                np.dot(self._omega, future_prices)
        day_weight = self.predict_rl(data_prices)
        print("weights:", day_weight)
        self._omega = day_weight
        preds_dict = {}
        for i, instrument in enumerate(self.instruments_list):
            preds_dict[instrument] = float(day_weight[i + 1])
        result = {
            "preds_dict": preds_dict,
            "last_portfolio": self._omega.tolist()
        }
        return result
