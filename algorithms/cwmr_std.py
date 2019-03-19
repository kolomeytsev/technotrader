import numpy as np
import pandas as pd
from technotrader.trading.agent import Agent
import technotrader.utils.agent_utils as agent_utils


class CwmrStdAgent(Agent):
    """
    Confidence Weighted Mean Reversion strategy (Li et al. 2011)
    http://proceedings.mlr.press/v15/li11b/li11b.pdf

    This strategy is based on using standard deviation (CWMR-Stdev problem).
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
        # See equation (7) in the original article
        part = (V - x_bar * np.sum(x.T @ self.sigma)) / M**2 + V * self.phi**2 / 2
        a = part**2 
        a -= V**2 * self.phi**4 / 4
        b = 2 * (self.epsilon - np.log(M)) * part
        c = (self.epsilon - np.log(M))**2 - self.phi**2 * V
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
        #print("lmbd", lmbd)
        #print("V", V)
        #print("e:", lmbd**2 * self.phi**2 * V**2 + 4 * V)
        U_root = 0.5 * (-lmbd * self.phi * V + np.sqrt(lmbd**2 * self.phi**2 * V**2 + 4 * V))
        #print("U_root", U_root)
        self.mu = self.mu - lmbd * (self.sigma @ (x - x_bar)).flatten() / M
        if np.linalg.matrix_rank(self.sigma) != self.sigma.shape[0]:
            print("sigma is singular")
            self.sigma += 1e-6
        S = np.linalg.inv(self.sigma) + self.phi * lmbd / U_root * np.diag(x)**2
        if np.linalg.matrix_rank(S) != S.shape[0]:
            print("S is singular")
            S += 1e-6
        self.sigma = np.linalg.inv(S)
        # Normalize mu and sigma
        self.mu = self.simplex_projection(self.mu)
        self.sigma = self.sigma / np.trace(self.sigma) / self.n_inst**2

    def cwmr_next_weight(self, data):
        x = data[-1]
        x = np.reshape(x, (x.size, 1))
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
        print("cwmr-std weights:", day_weight)
        self.last_portfolio = day_weight
        preds_dict = {}
        for i, instrument in enumerate(self.instruments_list):
            preds_dict[instrument] = day_weight[i]
        return preds_dict
