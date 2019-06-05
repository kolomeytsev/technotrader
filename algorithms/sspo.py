import numpy as np
import pandas as pd
from technotrader.trading.agent import Agent
import technotrader.utils.agent_utils as agent_utils


class SspoAgent(Agent):
    """
    Short-term Sparse Portfolio Optimization (SSPO) Based on
    Alternating Direction Method of Multipliers (ADMM)
    Zhao-Rong Lai et al.
    Journal of Machine Learning Research 19 (2018)

    Link:
    http://jmlr.org/papers/v19/17-558.html
    """
    def __init__(self, config, data_loader, trade_log=None):
        super().__init__(config, data_loader)
        self.n_step = 0
        self.instruments_list = config["instruments_list"]
        self.window = config["window"]
        self.max_iter = int(config["max_iter"])
        self.abstol = config["abstol"]
        self.zeta = config["zeta"]
        self.lmbd = config["lmbd"]
        self.gamma = config["gamma"]
        self.eta = config["eta"]
        self.tao = self.lmbd / self.gamma
        use_risk_free = config["use_risk_free"]
        self.n_inst = len(self.instruments_list)
        if use_risk_free:
            self.n_inst += 1
        self.last_portfolio = np.ones(self.n_inst) / self.n_inst
        self.weights_projection = agent_utils.WeightsProjection(config)
        timetable = agent_utils.ExchangeTimetable(config["exchange"])
        self.data_extractor = agent_utils.DataExtractor(data_loader,
                                timetable, config, config["window"] + 1, False)

    def sspo_next_weight(self, data_close):
        Rpredict = np.max(data_close, axis=0)
        x_tplus1 = Rpredict / data_close[-1]
        x_tplus1 = 1.1 * np.log(x_tplus1) + 1
        x = -x_tplus1
        g = self.last_portfolio
        rho = 0
        I = np.eye(self.n_inst)
        for i in range(self.max_iter):
            A = self.tao * I + self.eta
            B = self.tao * g + self.eta - rho - x
            b, resid, rank, s = np.linalg.lstsq(A, B)
            b_positive = (np.abs(b) - self.gamma)
            b_positive[b_positive < 0] = 0
            g = np.sign(b) * b_positive
            prim_res_tmp = b.sum() - 1
            rho = rho + self.eta * prim_res_tmp
            if np.abs(prim_res_tmp) < self.abstol:
                break
        b_tplus1_hat = self.zeta * b
        return self.weights_projection(b_tplus1_hat)

    def compute_portfolio(self, epoch):
        self.n_step += 1
        data_prices = self.data_extractor(epoch)
        day_weight = self.sspo_next_weight(data_prices)
        if self.verbose:
            print("sspo weights:", day_weight)
        self.last_portfolio = day_weight.copy()
        preds_dict = {}
        for i, instrument in enumerate(self.instruments_list):
            preds_dict[instrument] = day_weight[i]
        return preds_dict
