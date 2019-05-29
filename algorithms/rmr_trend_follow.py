import numpy as np
import pandas as pd
from technotrader.trading.agent import Agent
import technotrader.utils.agent_utils as agent_utils


class RmrTrendFollowAgent(Agent):
    """
    Robust Median Reversion strategy (Huang et al. [2013])
    with Trend Follow.
    """
    def __init__(self, config, data_loader, trade_log=None):
        super().__init__(config, data_loader)
        self.instruments_list = config["instruments_list"]
        self.epsilon = config["epsilon"]
        use_risk_free = config["use_risk_free"]

        self.window_long = config['window_long']
        self.window_short = config['window_short']
        self.short_flag = config["short_flag"]
        self.window = config["window"]
        self.down_trend_threshold = config["down_trend_threshold"]

        self.n_inst = len(self.instruments_list)
        #if use_risk_free:
        #    self.n_inst += 1
        self.last_portfolio = np.ones(self.n_inst) / self.n_inst
        self.compute_L1_median = agent_utils.compute_L1_median
        self.weights_projection = agent_utils.WeightsProjection(config)
        timetable = agent_utils.ExchangeTimetable(config["exchange"])
        max_window = max(config["window"], config["window_long"])
        self.data_extractor = agent_utils.DataExtractor(data_loader,
                                    timetable, config, max_window + 1, False)
        self.previous_trends = None
        self.down_trend_counter = np.zeros(len(self.instruments_list))

    def compute_next_weight(self, data_close):
        prices_median = self.compute_L1_median(data_close)
        x_t1 = prices_median / data_close[-1]
        denom = (np.linalg.norm(x_t1 - np.mean(x_t1))) ** 2
        if denom == 0:
            alpha = 0
        else:
            alpha = min(0, (np.dot(x_t1, self.last_portfolio) - self.epsilon) / denom)
        alpha = min(100000, alpha)
        weights = self.last_portfolio - alpha * (x_t1 - np.mean(x_t1))
        return self.weights_projection(weights)

    def check_trend(self, data, weights):
        long_mean = np.mean(data[-self.window_long:], axis=0)
        short_mean = np.mean(data[-self.window_short:], axis=0)
        trends = short_mean >= long_mean
        # check if previous trends still continue
        prev_trends = self.previous_trends

        self.down_trend_counter[trends] = 0
        self.down_trend_counter[~trends] += 1

        new_weights = weights.copy()
        new_weights[self.down_trend_counter >= self.down_trend_threshold] = 0.

        self.previous_trends = trends
        return new_weights

    def compute_portfolio(self, epoch):
        data_prices = self.data_extractor(epoch)
        weights = self.compute_next_weight(data_prices[-self.window:])
        print("rmr weights:", weights)
        weights = self.check_trend(data_prices, weights)
        print("rmr trend follow weights:", weights)
        print()
        self.last_portfolio = weights.copy()
        preds_dict = {}
        for i, instrument in enumerate(self.instruments_list):
            preds_dict[instrument] = weights[i]
        return preds_dict
