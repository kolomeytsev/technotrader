import numpy as np
import pandas as pd
from technotrader.trading.agent import Agent
import technotrader.utils.agent_utils as agent_utils


class AnticorAgent(Agent):
    """
    Anti Correlation strategy (Borodin et al.[2003; 2004]).
    https://arxiv.org/pdf/1107.0036.pdf
    Variables:
    window: maximum window size, the number of experts (window-1)
    """
    def __init__(self, config, data_loader, trade_log=None):
        super().__init__(config, data_loader)
        self.instruments_list = config['instruments_list']
        self.n_inst = len(self.instruments_list)
        use_risk_free = config["use_risk_free"]
        if use_risk_free:
            self.n_inst += 1
        self.last_portfolio = np.ones(self.n_inst) / self.n_inst
        self.window = config['window']
        self.weights_projection = agent_utils.WeightsProjection(config)
        timetable = agent_utils.ExchangeTimetable(config["exchange"])
        self.data_extractor = agent_utils.DataExtractor(data_loader, 
                                    timetable, config, 2 * self.window + 1, True)
        self.anticor_expert = agent_utils.anticor_expert

    def anticor_kernel(self, data):
        weights = self.anticor_expert(data, self.last_portfolio, self.window)
        return weights

    def compute_portfolio(self, epoch):
        data_price_relatives = self.data_extractor(epoch)
        day_weight = self.anticor_kernel(data_price_relatives)
        day_weight = self.weights_projection(day_weight)
        if self.verbose:
            print("anticor weights:", day_weight)
        self.last_portfolio = day_weight
        preds_dict = {}
        for i, instrument in enumerate(self.instruments_list):
            preds_dict[instrument] = day_weight[i]
        return preds_dict
