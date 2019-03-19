import numpy as np
import pandas as pd
import scipy as sp
from technotrader.trading.agent import Agent
import technotrader.utils.agent_utils as agent_utils
import cvxopt


class OnsAgent(Agent):
    """
    Online Newton Step strategy.
    A. Agarwal et. al.
    Algorithms for Portfolio Management based on the Newton Method, 2006.
    """
    def __init__(self, config, data_loader, trade_log=None):
        super().__init__(config, data_loader)
        self.instruments_list = config["instruments_list"]
        self.eta = config['mixture']
        self.beta = config['tradeoff']
        self.delta = config['heuristic_tuning']
        self.n_inst = len(self.instruments_list)
        self.use_risk_free = config["use_risk_free"]
        if self.use_risk_free:
            self.n_inst += 1
        self.last_portfolio = np.ones(self.n_inst) / self.n_inst
        self.A = np.eye(self.n_inst)
        self.b = np.zeros((self.n_inst, 1))
        self.temperature = config['temperature']
        self.weights_projection = agent_utils.WeightsProjection(config)
        timetable = agent_utils.ExchangeTimetable(config["exchange"])
        self.data_extractor = agent_utils.DataExtractor(data_loader,
                                            timetable, config, 2, True)

    def find_projection_to_simplex(self, x, M):
        n = M.shape[0]
        P = cvxopt.matrix(2 * M)
        q = cvxopt.matrix(-2 * M @ x)
        G = cvxopt.matrix(-np.eye(n))
        h = cvxopt.matrix(np.zeros((n, 1)))
        A = cvxopt.matrix(np.ones((1, n)))
        b = cvxopt.matrix(1.)
        sol = cvxopt.solvers.qp(P, q, G, h, A, b)
        return np.squeeze(sol['x'])

    def ons_next_weight(self, x):
        """
        :param x: last market price ratios vector
        """
        N = len(x)
        grad = x / (x @ self.last_portfolio)
        grad = grad.reshape(-1, 1)
        hessian = -1 * (grad @ grad.T)
        self.A += (-1) * hessian
        self.b += (1 + 1 / self.beta) * grad
        A_inv = np.linalg.inv(self.A)
        B = self.delta * A_inv @ self.b
        weight = self.find_projection_to_simplex(B, self.A)
        weight = (1 - self.eta) * weight + self.eta * np.ones(N) / N
        if self.temperature > 0:
            return sp.special.softmax(weight / self.temperature)
        return weight / np.abs(weight).sum()

    def compute_portfolio(self, epoch):
        data_price_relatives = self.data_extractor(epoch)
        day_weight = self.ons_next_weight(data_price_relatives[-1])
        print("ons weights:", day_weight)
        self.last_portfolio = day_weight
        preds_dict = {}
        for i, instrument in enumerate(self.instruments_list):
            preds_dict[instrument] = day_weight[i]
        return preds_dict
