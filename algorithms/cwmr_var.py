import numpy as np
import pandas as pd
from technotrader.trading.agent import Agent
import technotrader.utils.agent_utils as agent_utils


class CwmrVarAgent(Agent):
    """
    Confidence Weighted Mean Reversion strategy (Li et al. 2011)
    http://proceedings.mlr.press/v15/li11b/li11b.pdf

    This strategy is based on using variance (CWMR-Var problem).
    """
    def __init__(self, config, data_loader, trade_log=None):
        super().__init__(config, data_loader)
        self.instruments_list = config['instruments_list']
        self.n_inst = len(self.instruments_list)
        self.use_risk_free = config["use_risk_free"]
        if self.use_risk_free:
            self.n_inst += 1
        self.last_portfolio = np.ones(self.n_inst) / self.n_inst
        self.epsilon = config['epsilon']
        self.phi = config['confidence']
        self.lmbd = 0
        # mu: last portfolio mean
        self.mu = np.ones(self.n_inst) / self.n_inst
        # sigma: last diagonal covariance matrix
        self.sigma = np.eye(self.n_inst) / (self.n_inst**2)
        self.simplex_projection = agent_utils.simplex_projection
        self.weights_projection = agent_utils.WeightsProjection(config)
        timetable = agent_utils.ExchangeTimetable(config["exchange"])
        self.data_extractor = agent_utils.DataExtractor(data_loader,
                                    timetable, config, 2, True)

    def update_mu_sigma(self, x, x_bar, M, V):
        # See equation (5) in the original article
        a = 2 * self.phi * V**2 - 2 * self.phi * V * x_bar * np.sum(x.T @ self.sigma)
        a /= M**2
        b = 2 * self.phi * V * (self.epsilon - np.log(M)) 
        b += V - x_bar * np.sum(x.T @ self.sigma)
        b /= M**2
        c = self.epsilon - np.log(M) - self.phi * V
        t1 = b
        t2 = np.sqrt(b**2 - 4 * a * c)
        t3 = 2 * a
        if a != 0 and t2 > 0:
            lmbd = np.max([
                0,
                (-t1 + t2) / t3,
                (-t1 - t2) / t3
            ])
        elif a == 0 and b != 0:
            lmbd = max([0, -c / b])
        else:
            lmbd = 0
        lmbd = np.minimum(lmbd, 1e6)
        # Update mu and sigma
        self.mu = self.mu - lmbd * (self.sigma @ (x - x_bar)).flatten() / M
        self.sigma = np.linalg.inv(np.linalg.inv(self.sigma) + \
                        2 * self.phi * lmbd * np.diag(x)**2)
        # Normalize mu and sigma
        self.mu = self.simplex_projection(self.mu)
        self.sigma = self.sigma / np.trace(self.sigma) / self.n_inst**2

    def cwmr_next_weight(self, data):
        x = np.reshape(data, (data.size, 1))
        M = np.inner(self.mu, x.flatten())
        sigma_x = self.sigma @ x
        V = x.T @ sigma_x
        V = V[0, 0]
        x_bar = np.sum(sigma_x) / np.trace(self.sigma)
        self.update_mu_sigma(x, x_bar, M, V)
        return self.mu

    def compute_portfolio(self, epoch):
        data_price_relatives = self.data_extractor(epoch)
        day_weight = self.cwmr_next_weight(data_price_relatives)
        print("cwmr-var weights:", day_weight)
        self.last_portfolio = day_weight
        preds_dict = {}
        for i, instrument in enumerate(self.instruments_list):
            preds_dict[instrument] = day_weight[i]
        return preds_dict
