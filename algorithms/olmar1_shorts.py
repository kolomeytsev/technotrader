import numpy as np
import pandas as pd
from technotrader.trading.agent import Agent
import technotrader.utils.agent_utils as agent_utils


class Olmar1ShortsAgent(Agent):
    """
    On-Line Moving Average Reversion strategy (Li and Hoi [2012]). Variant 1.
    """
    def __init__(self, config, data_loader, trade_log=None):
        super().__init__(config, data_loader)
        self.instruments_list = config["instruments_list"]
        self.epsilon = config["epsilon"]
        use_risk_free = config["use_risk_free"]
        self.n_inst = len(self.instruments_list)
        if use_risk_free:
            self.n_inst += 1
        self.last_portfolio = np.ones(self.n_inst) / self.n_inst
        self.weights_projection = agent_utils.WeightsProjection(config)
        timetable = agent_utils.ExchangeTimetable(config["exchange"])
        self.data_extractor = agent_utils.DataExtractor(data_loader,
                        timetable, config, config["window"] + 1, True)

    def olmar1_next_weight(self, data):
        T, N = data.shape
        data_phi = np.zeros(N)
        tmp_x = np.ones(N)
        for i in range(1, T + 1):
            data_phi = data_phi + 1. / tmp_x
            tmp_x = tmp_x * data[T - i]
        data_phi /= T
        ell = max([0, self.epsilon - (data_phi - 1) @ self.last_portfolio + 1])
        x_bar = np.mean(data_phi)
        denom_part = data_phi - x_bar
        denominator = np.dot(denom_part, denom_part)
        if denominator != 0:
            lmbd = ell / denominator
        else:
            lmbd = 0
        weights = self.last_portfolio + lmbd * (data_phi - x_bar)
        return self.weights_projection(weights)

    def compute_portfolio(self, epoch):
        data_price_relatives = self.data_extractor(epoch)
        day_weight = self.olmar1_next_weight(data_price_relatives)
        if self.verbose:
            print("olmar1 shorts:", day_weight)
        self.last_portfolio = day_weight
        preds_dict = {}
        for i, instrument in enumerate(self.instruments_list):
            preds_dict[instrument] = day_weight[i]
        return preds_dict
