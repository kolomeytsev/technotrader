import numpy as np
import pandas as pd
from technotrader.trading.agent import Agent
import technotrader.utils.agent_utils as agent_utils


class TrendFollowAgent(Agent):
    """
    Trend-Following strategy based on Moving Average.

    Short description:
    Calculating long ('window_long') and short ('window_short') moving averages (MA).
    If short MA is greater than long MA and the asset is not invested then
    we take long position. The portion of invested money is controlled by
    parameter 'portion'. The default portion is 1 / m, m - number of instruments.
    If there is not enough money to invest then invest all that we have.
    If the long MA is greater than short MA then close the long position.
    """
    def __init__(self, config, data_loader, trade_log=None):
        super().__init__(config, data_loader)
        self.n_steps = 0
        self.instruments_list = config['instruments_list']
        self.n_inst = len(self.instruments_list)
        self.last_portfolio = np.ones(self.n_inst) / self.n_inst
        self.window_long = config['window_long']
        self.window_short = config['window_short']
        self.short_flag = config["short_flag"]
        if self.window_long <= self.window_short:
            print("Error: window_long must be greater than window_short.")
            exit(1)
        self.weights_projection = agent_utils.WeightsProjection(config)
        timetable = agent_utils.ExchangeTimetable(config["exchange"])
        self.data_extractor = agent_utils.DataExtractor(data_loader,
                                    timetable, config, self.window_long + 1, False)
        if config.get('portion') is not None:
            self.portion = config['portion']
        else:
            self.portion = 1. / self.n_inst
        if config.get('variant') is not None:
            self.variant = config['variant']
        else:
            self.variant = 0
        if self.variant not in {0, 1}:
            print("Wrong variant. Available variants: 0, 1.")
            exit(1)
        if config.get('sort_method') is not None:
            self.sort_method = config['sort_method']
        else:
            self.sort_method = 0
        if self.sort_method not in {0, 1, 2, 3}:
            print("Wrong sort_method. Available variants: 0, 1, 2, 3")
            exit(1)
        if config.get('sort_threshold') is not None:
            self.sort_threshold = config['sort_threshold']
        else:
            self.sort_threshold = 0
        if config.get('min_hold_periods') is not None:
            self.min_hold_periods = config['min_hold_periods']
        else:
            self.min_hold_periods = 0
        if config.get('sort_desc_flag') is not None:
            self.sort_desc_flag = config['sort_desc_flag']
        else:
            self.sort_desc_flag = False
        self.long_hold_counter = 0
        self.short_hold_counter = 0
        self.long_positions = np.zeros(self.n_inst)
        self.long_positions_flags = np.zeros(self.n_inst).astype(bool)
        self.long_available_money = 1.
        self.short_positions = np.zeros(self.n_inst)
        self.short_positions_flags = np.zeros(self.n_inst).astype(bool)
        self.short_available_money = 1.

    def trend_update_first_long(self, index, trend_flag):
        if trend_flag:
            # up trend
            if not self.long_positions_flags[index]:
                if self.long_available_money > 0:
                    if self.long_available_money >= self.portion:
                        portion = self.portion
                    else:
                        portion = self.long_available_money
                    self.long_available_money -= portion
                    #print("Buying: %s, amount: %.3f" % (self.instruments_list[index], portion))
                    self.long_positions[index] = portion
                    self.long_positions_flags[index] = True
                else:
                    #print("No money to buy:", self.instruments_list[index])
                    pass
            else:
                #print("Holding:", self.instruments_list[index])
                #self.positions[index] = np.nan
                pass
        else:
            # down trend
            if self.long_positions_flags[index]:
                #print("Selling:", self.instruments_list[index])
                self.long_available_money += self.long_positions[index]
                self.long_positions[index] = 0.
                self.long_positions_flags[index] = False
            else:
                #print("Doing nothing:", self.instruments_list[index])
                self.long_positions[index] = 0

    def trend_update_first_short(self, index, trend_flag):
        if not trend_flag:
            # down trend
            if not self.short_positions_flags[index]:
                if self.short_available_money > 0:
                    if self.short_available_money >= self.portion:
                        portion = self.portion
                    else:
                        portion = self.short_available_money
                    self.short_available_money -= portion
                    #print("Buying: %s, amount: %.3f" % (self.instruments_list[index], portion))
                    self.short_positions[index] = portion
                    self.short_positions_flags[index] = True
                else:
                    #print("No money to buy:", self.instruments_list[index])
                    pass
            else:
                #print("Holding:", self.instruments_list[index])
                #self.positions[index] = np.nan
                pass
        else:
            # up trend
            if self.short_positions_flags[index]:
                #print("Selling:", self.instruments_list[index])
                self.short_available_money += self.short_positions[index]
                self.short_positions[index] = 0.
                self.short_positions_flags[index] = False
            else:
                #print("Doing nothing:", self.instruments_list[index])
                self.short_positions[index] = 0

    def trend_update_second(self, index, trend_flag):
        if trend_flag:
            # up trend
            if not self.positions_flags[index]:
                #print("Buying:", self.instruments_list[index])
                self.long_positions[index] = 1.
                self.positions_flags[index] = True
            else:
                #print("Holding:", self.instruments_list[index])
                self.long_positions[index] = 1.
        else:
            # down trend
            if self.positions_flags[index]:
                #print("Selling:", self.instruments_list[index])
                self.long_positions[index] = 0.
                self.long_positions[index] = False
            else:
                #print("Doing nothing:", self.instruments_list[index])
                self.long_positions[index] = 0.

    def trend_follow_next_weight_2_long(self, data):
        long_mean = np.mean(data[-self.window_long:], axis=0)
        short_mean = np.mean(data[-self.window_short:], axis=0)
        trends = short_mean > long_mean
        #print("long trends:", trends)
        # check if previous trends still continue
        prev_trends = np.all(trends[np.argwhere(self.long_positions_flags).flatten()])
        if self.long_hold_counter > 0 and prev_trends:
            # continue with previous trends
            print("Holding previous weights")
            self.long_hold_counter -= 1
            return self.long_positions
        else:
            self.long_hold_counter = self.min_hold_periods
            trends_diff = (short_mean - long_mean) / short_mean
            threshold = np.percentile(trends_diff, self.sort_threshold)

            indices_no_trend = np.argwhere(~trends).flatten()
            sorted_indices_trend = np.argsort(trends_diff)[np.sort(trends_diff) > threshold]

            for i, indices in enumerate([indices_no_trend, sorted_indices_trend]):
                for index in indices:
                    trend_flag = trends[index]
                    if i == 1 and self.sort_method == 3:
                        # in this case treat all assets > threshold as up trends
                        trend_flag = True
                    if self.variant == 0:
                        self.trend_update_first_long(index, trend_flag)
                    else:
                        self.trend_update_second(index, trend_flag)
        if np.abs(self.long_positions).sum() > 1:
            self.long_positions /= np.abs(self.long_positions).sum()
        return self.long_positions

    def trend_follow_next_weight_2_short(self, data):
        long_mean = np.mean(data[-self.window_long:], axis=0)
        short_mean = np.mean(data[-self.window_short:], axis=0)
        trends = short_mean < long_mean
        #print("short trends:", trends)
        # check if previous trends still continue
        prev_trends = np.all(trends[np.argwhere(self.short_positions_flags).flatten()])
        if self.short_hold_counter > 0 and prev_trends:
            # continue with previous trends
            print("Holding previous weights")
            self.short_hold_counter -= 1
            return self.short_positions
        else:
            self.short_hold_counter = self.min_hold_periods
            trends_diff = (short_mean - long_mean) / short_mean
            threshold = np.percentile(trends_diff, 100 - self.sort_threshold)

            indices_no_trend = np.argwhere(~trends).flatten()
            sorted_indices_trend = np.argsort(trends_diff)[np.sort(trends_diff) < threshold]
            sorted_indices_trend = sorted_indices_trend[::-1]
            for i, indices in enumerate([indices_no_trend, sorted_indices_trend]):
                for index in indices:
                    trend_flag = trends[index]
                    if i == 1 and self.sort_method == 3:
                        # in this case treat all assets > threshold as up trends
                        trend_flag = True
                    if self.variant == 0:
                        self.trend_update_first_short(index, trend_flag)
                    else:
                        self.trend_update_second(index, trend_flag)
        if np.abs(self.short_positions).sum() > 1:
            self.short_positions /= np.abs(self.short_positions).sum()
        return -self.short_positions

    def trend_follow_next_weight(self, data):
        long_mean = np.mean(data[-self.window_long:], axis=0)
        short_mean = np.mean(data[-self.window_short:], axis=0)
        trends = short_mean > long_mean
        trends_diff = short_mean - long_mean
        if self.sort_method == 0:
            trends_diff /= short_mean
        indices_no_trend = np.argwhere(~trends).flatten()
        sorted_indices = trends_diff.argsort()
        if self.sort_desc_flag:
            sorted_indices = sorted_indices[::-1]
        sorted_indices_trend = [index for index in sorted_indices if trends[index]]
        for indices in [indices_no_trend, sorted_indices_trend]:
            for index in indices:
                if self.variant == 0:
                    self.trend_update_first_long(index, trends[index])
                else:
                    self.trend_update_second(index, trends[index])
        if np.abs(self.long_positions).sum() > 1:
            self.long_positions /= np.abs(self.long_positions).sum()
        return self.long_positions

    def compute_portfolio(self, epoch):
        self.n_steps += 1
        data_prices = self.data_extractor(epoch)
        if self.sort_method in {0, 1}:
            day_weight = self.trend_follow_next_weight(data_prices)
        elif self.sort_method == 2:
            day_weight_long = self.trend_follow_next_weight_2_long(data_prices)
            if self.short_flag:
                day_weight_short = self.trend_follow_next_weight_2_short(data_prices)
            day_weight = np.zeros(day_weight_long.shape[0])
            day_weight[day_weight_long > 0] = day_weight_long[day_weight_long > 0]
            if self.short_flag:
                day_weight[day_weight_short < 0] = day_weight_short[day_weight_short < 0]
            denom = np.abs(day_weight).sum()
            if denom > 1:
                day_weight /= denom
        else:
            print("Wrong sort_method")
            exit(1)
        if self.verbose:
            print("trend follow weights:", day_weight)
        self.last_portfolio = day_weight
        preds_dict = {}
        for i, instrument in enumerate(self.instruments_list):
            if not np.isnan(day_weight[i]):
                preds_dict[instrument] = day_weight[i]
            else:
                print("Nans as predictions are not supported yet")
                #preds_dict[instrument] = None
                exit(1)
        return preds_dict
