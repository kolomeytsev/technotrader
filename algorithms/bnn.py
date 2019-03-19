import numpy as np
import pandas as pd
from technotrader.trading.agent import Agent
import technotrader.utils.agent_utils as agent_utils


class BnnAgent(Agent):
    """
    Bnn strategy.

    Based on article:
    Nonparametric nearest neighbor based empirical portfolio selection strategies
    L. Gyorfi et al. ‎2008
    https://pdfs.semanticscholar.org/e233/2e745f14c4e9c3607c604327638629ddef50.pdf
    """
    def __init__(self, config, data_loader, trade_log=None):
        super().__init__(config, data_loader)
        self.n_steps = 0
        self.instruments_list = config['instruments_list']
        self.n_inst = len(self.instruments_list)
        use_risk_free = config["use_risk_free"]
        if use_risk_free:
            self.n_inst += 1
        self.last_portfolio = np.ones(self.n_inst) / self.n_inst
        self.sequence_length = config["sequence_length"]
        self.nearest_neighbors_number = config['nearest_neighbors']
        self.exp_ret = np.ones((self.sequence_length, self.nearest_neighbors_number + 1))
        self.exp_w = np.ones((self.sequence_length * (self.nearest_neighbors_number + 1),
                              self.n_inst)) / self.n_inst
        self.optimize_weights = agent_utils.optimize_weights
        self.weights_projection = agent_utils.WeightsProjection(config)
        timetable = agent_utils.ExchangeTimetable(config["exchange"])
        self.data_extractor = agent_utils.DataExtractor(data_loader,
                                    timetable, config, None, True)

    def bnn_expert(self, data, k, nn_number):
        """
        Generates portfolio for a specified parameter setting.
        
        :param data: market price ratios vectors
        :param k: sequence length
        :param nn_number: the number of nearest neighbors
        """
        T, N = data.shape
        m = 0
        histdata = np.zeros((T, N))
        normid = np.zeros(T)
        if T <= k + 1:
            weight = np.ones(N) / N
            return weight
        if k == 0 and nn_number == 0:
            histdata = data[:T]
            m = T
        else:
            histdata = data[:T]
            normid[:k] = np.inf
            for i in range(k, T):
                data2 = data[i - k: i - 1] - data[T - k + 1: T]
                normid[i] = np.sqrt(np.trace(data2 @ data2.T))
            sortpos = np.argsort(normid)
            m = int(np.floor(nn_number * T))
            histdata = histdata[sortpos[:m]]
        if m == 0:
            weight = np.ones(N) / N
            return weight
        weight = self.optimize_weights(histdata[:m])
        return weight / sum(weight)

    def bnn_kernel(self, data):
        """
        :param data: market price ratios vectors
        
        Returns:
        :weight: final portfolio, used for next rebalance
        """
        l = self.nearest_neighbors_number
        self.exp_w[self.sequence_length * l] = self.bnn_expert(data, 0, 0)
        for k_index in range(self.sequence_length):
            for l_index in range(l):
                nn_number = 0.02 + 0.5 * l_index / (l - 1)
                self.exp_w[k_index * l + l_index] = self.bnn_expert(data, k_index + 1, nn_number)
        # Combine portfolios according to q(k, l) and previous expert return
        q = 1 / (self.sequence_length * l + 1)
        numerator = q * self.exp_ret[0, l] * self.exp_w[self.sequence_length * l]
        denominator = q * self.exp_ret[0, l]
        for k_index in range(self.sequence_length):
            for l_index in range(l):
                numerator += q * self.exp_ret[k_index, l_index] * self.exp_w[k_index * l + l_index]
                denominator += q * self.exp_ret[k_index, l_index]
        weight = numerator / denominator
        return weight / sum(weight)

    def compute_portfolio(self, epoch):
        self.n_steps += 1
        data_price_relatives = self.data_extractor(epoch, self.n_steps + 1)
        # experts return
        for k_index in range(self.sequence_length):
            for l_index in range(self.nearest_neighbors_number):
                weights = self.exp_w[k_index * self.nearest_neighbors_number + l_index]
                self.exp_ret[k_index, l_index] *= np.dot(data_price_relatives[-1], weights)
        day_weight = self.bnn_kernel(data_price_relatives)
        day_weight = self.weights_projection(day_weight)
        print("bnn weights:", day_weight)
        preds_dict = {}
        for i, instrument in enumerate(self.instruments_list):
            preds_dict[instrument] = day_weight[i]
        return preds_dict
