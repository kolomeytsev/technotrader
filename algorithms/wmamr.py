import numpy as np
import pandas as pd
from technotrader.trading.agent import Agent
from technotrader.algorithms.pamr import PamrAgent
import technotrader.utils.agent_utils as agent_utils


class WmamrAgent(PamrAgent):
    def __init__(self, config, data_loader, trade_log=None):
        super().__init__(config, data_loader)      
        self.window = config["window"]
        timetable = agent_utils.ExchangeTimetable(config["exchange"])
        self.data_extractor = agent_utils.DataExtractor(data_loader,
                                            timetable, config, self.window, True)

    def compute_portfolio(self, epoch):
        self.n_steps += 1
        data_price_relatives = self.data_extractor(epoch)
        data_price_relatives_mean = data_price_relatives.mean(0)
        self.update_lagrange_multiplier(data_price_relatives_mean)

        if self.n_steps == 1:
            day_weight = np.ones(self.instruments_number) / self.instruments_number
        else:   
            day_weight = self.pamr_expert(data_price_relatives_mean)
        if self.verbose:
            print("wmamr weights:", day_weight)
        self.last_portfolio = day_weight
        preds_dict = {}
        for i, instrument in enumerate(self.instruments_list):
            preds_dict[instrument] = day_weight[i]
        return preds_dict
