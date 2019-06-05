import numpy as np
import pandas as pd
from technotrader.trading.agent import Agent
import technotrader.utils.agent_utils as agent_utils


class BollingerAgent(Agent):
    """
    Trend-Following strategy based on Bollinger Bands.
    """
    def __init__(self, config, data_loader, trade_log=None):
        super().__init__(config, data_loader)
        self.n_steps = 0
        self.instruments_list = config['instruments_list']
        self.n_inst = len(self.instruments_list)
        self.window = config['window']
        self.std_coef_open = config['std_coef_open']
        self.std_coef_close = config['std_coef_close']
        self.allow_short = config['allow_short']
        if config.get('portion') is not None:
            self.portion = config['portion']
        else:
            self.portion = 1. / self.n_inst
        self.long_positions = np.zeros(self.n_inst)
        self.long_positions_flags = np.zeros(self.n_inst).astype(bool)
        self.short_positions = np.zeros(self.n_inst)
        self.short_positions_flags = np.zeros(self.n_inst).astype(bool)
        self.available_money = 1.
        self.weights_projection = agent_utils.WeightsProjection(config)
        timetable = agent_utils.ExchangeTimetable(config["exchange"])
        self.data_extractor = agent_utils.DataExtractor(data_loader,
                                timetable, config, self.window + 1, False)

    def bollinger_update_long(self, index, trend_flag_open, trend_flag_close):
        if trend_flag_open:
            #print("Long open signal")
            if not self.long_positions_flags[index]:
                if self.available_money > 0:
                    if self.available_money >= self.portion:
                        portion = self.portion
                    else:
                        portion = self.available_money
                    self.available_money -= portion
                    #print("Open long position: %s, amount: %.3f" % \
                    #        (self.instruments_list[index], portion))
                    self.long_positions[index] = portion
                    self.long_positions_flags[index] = True
                else:
                    #print("No money to buy:", self.instruments_list[index])
                    pass
            else:
                #print("Holding:", self.instruments_list[index])
                #self.long_positions[index] = np.nan
                pass
        else:
            #print("No long open signal")
            if self.long_positions_flags[index]:
                if trend_flag_close:
                    #print("Closing long position:", self.instruments_list[index])
                    self.available_money += self.long_positions[index]
                    self.long_positions[index] = 0
                    self.long_positions_flags[index] = False
                else:
                    #print("Holding long position:", self.instruments_list[index])
                    #self.long_positions[index] = np.nan
                    pass
            else:
                #print("Doing nothing:", self.instruments_list[index])
                self.long_positions[index] = 0

    def bollinger_update_short(self, trends_down, lower_band, trends_mean_diff):
        print("Not Implemented")
        exit(1)

    def bollinger_next_weight(self, data):
        mean = np.mean(data[-self.window:], axis=0)
        std = np.std(data[-self.window:], axis=0)
        upper_band_open = mean + self.std_coef_open * std
        lower_band_open = mean - self.std_coef_open * std
        upper_band_close = mean + self.std_coef_close * std
        lower_band_close = mean - self.std_coef_close * std
        trends_up_open = data[-1] >= upper_band_open
        trends_down_open = data[-1] <= lower_band_open
        trends_up_close = data[-1] <= upper_band_close
        trends_down_close = data[-1] >= lower_band_close
        trends_mean_diff = (data[-1] - mean) / data[-1]
        for index in trends_mean_diff.argsort()[::-1]:
            self.bollinger_update_long(index, trends_up_open[index], trends_up_close[index])
        weights =  self.long_positions
        if sum(weights) > 0:
            weights /= sum(weights)
        return weights

    def compute_portfolio(self, epoch):
        self.n_steps += 1
        data_prices = self.data_extractor(epoch)
        day_weight = self.bollinger_next_weight(data_prices)
        if self.verbose:
            print("bollinger weights:", day_weight)
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
