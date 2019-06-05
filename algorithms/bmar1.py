import numpy as np
import pandas as pd
from technotrader.trading.agent import Agent
import technotrader.utils.agent_utils as agent_utils


class Bmar1Agent(Agent):
    """
    Boosting Moving Average Reversion Strategy (Lin Xiao et. al.)
    Version 1.
    
    Link:
    https://pdfs.semanticscholar.org/ed2f/3310d988a09dcce64a887c6d76999989c295.pdf
    """
    def __init__(self, config, data_loader, trade_log=None):
        super().__init__(config, data_loader)
        self.n_steps = 0
        self.instruments_list = config['instruments_list']
        self.n_inst = len(config['instruments_list'])
        use_risk_free = config["use_risk_free"]
        if use_risk_free:
            self.n_inst += 1
        self.last_portfolio = np.ones(self.n_inst) / self.n_inst
        self.epsilon = config['epsilon']
        self.window = config['max_window']
        self.windows = np.arange(2, self.window + 1)
        self.eta = config['eta']
        self.n_experts = len(self.windows)
        self.losses = np.zeros(self.n_experts)
        self.data_phi = np.zeros((self.n_experts, self.n_inst))
        self.weights_projection = agent_utils.WeightsProjection(config)
        timetable = agent_utils.ExchangeTimetable(config["exchange"])
        self.data_extractor = agent_utils.DataExtractor(data_loader, 
                                    timetable, config, self.window + 2, True)

    def get_experts_preds(self, data, index, w):
        T, N = data.shape
        if T < w:
            data_phi = data[T - 1]
            self.data_phi[index, :] = data_phi
        else:
            self.losses[index] += np.mean((data[T - 1] - self.data_phi[index, :])**2)
            data_phi = np.zeros(N)
            tmp_x = np.ones(N)
            for i in range(1, w + 1):
                data_phi = data_phi + 1. / tmp_x
                tmp_x = tmp_x * data[T - i]
            data_phi /= w
        return data_phi

    def bmar1_next_weight(self, data):
        T, N = data.shape
        if T < self.window + 1:
            data_phi = data[T - 1]
        else:
            for i, w in enumerate(self.windows):
                expert_phi = self.get_experts_preds(data, i, w)
                self.data_phi[i, :] = expert_phi

            weights = -self.eta * self.losses
            weights = np.exp(weights) / np.sum(np.exp(weights))
            data_phi = self.data_phi.T.dot(weights)
        ell = max([0, self.epsilon - data_phi @ self.last_portfolio])
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
        self.n_steps += 1
        data_price_relatives = self.data_extractor(epoch)
        day_weight = self.bmar1_next_weight(data_price_relatives)
        if self.verbose:
            print("bmar1 weights", day_weight)
        self.last_portfolio = day_weight
        preds_dict = {}
        for i, instrument in enumerate(self.instruments_list):
            preds_dict[instrument] = day_weight[i]
        return preds_dict
