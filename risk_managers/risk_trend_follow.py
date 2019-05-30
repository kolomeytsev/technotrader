import numpy as np
import pandas as pd
from technotrader.trading.agent import Agent
import technotrader.utils.agent_utils as agent_utils


class TrendFollowRiskManager(Agent):
    """
    Risk Manager base on Trend Tracking.
    Returns weights: the greater the weights is
    the less risky is the asset.
    """
    def __init__(self, config, data_loader, trade_log=None):
        super().__init__(config, data_loader)
        self.instruments_list = config["instruments_list"]
        self.window_long = config['window_long']
        self.window_short = config['window_short']
        assert self.window_long > self.window_short, \
            "window_long must be greater than window_short"
        self.down_trend_threshold = config["down_trend_threshold"]
        self.n_inst = len(self.instruments_list)
        timetable = agent_utils.ExchangeTimetable(config["exchange"])
        self.data_extractor = agent_utils.DataExtractor(data_loader,
                                    timetable, config, self.window_long + 1, False)
        self.previous_trends = None
        self.down_trend_counter = np.zeros(len(self.instruments_list))

    def check_trend(self, data):
        long_mean = np.mean(data[-self.window_long:], axis=0)
        short_mean = np.mean(data[-self.window_short:], axis=0)
        trends = short_mean >= long_mean
        # check if previous trends still continue
        prev_trends = self.previous_trends

        self.down_trend_counter[trends] = 0
        self.down_trend_counter[~trends] += 1

        weights = np.ones(self.n_inst)
        weights[self.down_trend_counter >= self.down_trend_threshold] = 0.

        self.previous_trends = trends
        return weights

    def compute_risks(self, epoch):
        data_prices = self.data_extractor(epoch)
        weights = self.check_trend(data_prices)
        print("risk weights:", weights)
        preds_dict = {}
        for i, instrument in enumerate(self.instruments_list):
            preds_dict[instrument] = weights[i]
        return preds_dict
