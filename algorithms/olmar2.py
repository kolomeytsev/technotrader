import numpy as np
import pandas as pd
from technotrader.trading.agent import Agent
import technotrader.utils.agent_utils as agent_utils


class Olmar2Agent(Agent):
    """
    On-Line Moving Average Reversion strategy (Li and Hoi [2012]). Variant 2.
    """
    def __init__(self, config, data_loader, trade_log=None):
        super().__init__(config, data_loader)
        self.instruments_list = config["instruments_list"]
        self.alpha = config["alpha"]
        self.epsilon = config["epsilon"]
        use_risk_free = config["use_risk_free"]
        self.n_inst = len(self.instruments_list)
        if use_risk_free:
            self.n_inst += 1
        self.last_portfolio = np.ones(self.n_inst) / self.n_inst
        self.data_phi = np.zeros(self.n_inst)
        self.weights_projection = agent_utils.WeightsProjection(config)
        timetable = agent_utils.ExchangeTimetable(config["exchange"])
        self.data_extractor = agent_utils.DataExtractor(data_loader, 
                                        timetable, config, 2, True)

    def olmar2_next_weight(self, x):
        self.data_phi = self.alpha + (1 - self.alpha) * self.data_phi / x
        ell = max([0, self.epsilon - self.data_phi @ self.last_portfolio])
        x_bar = np.mean(self.data_phi)
        denom_part = self.data_phi - x_bar
        denominator = np.dot(denom_part, denom_part)
        if denominator != 0:
            lmbd = ell / denominator
        else:
            lmbd = 0
        weights = self.last_portfolio + lmbd * (self.data_phi - x_bar)
        return self.weights_projection(weights)

    def compute_portfolio(self, epoch):
        data_price_relatives = self.data_extractor(epoch)
        day_weight = self.olmar2_next_weight(data_price_relatives.ravel())
        print("olmar2:", day_weight)
        self.last_portfolio = day_weight
        preds_dict = {}
        for i, instrument in enumerate(self.instruments_list):
            preds_dict[instrument] = day_weight[i]
        return preds_dict
