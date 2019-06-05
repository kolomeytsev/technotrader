import numpy as np
import pandas as pd
from technotrader.trading.agent import Agent
import technotrader.utils.agent_utils as agent_utils


class AnticorAnticorAgent(Agent):
    """
    Anti Correlation strategy (Borodin et al.[2003; 2004]).
    https://arxiv.org/pdf/1107.0036.pdf

    This version is BAH(Anticor(Anticor)) strategy.

    Variables:
    window: maximum window size, the number of experts
    exp_ret: experts' return in the first fold
    exp_w: experts' weights in the first fold
    exp_ret_second: experts' return in the second fold
    exp_w_second: experts' weights in the second fold
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
        self.window = config['window']
        self.exp_ret = np.ones(self.window - 1)
        self.exp_w = np.ones((self.window - 1, self.n_inst)) / self.n_inst
        self.exp_ret_second = np.ones(self.window - 1)
        self.exp_w_second = np.ones((self.window - 1, 
                                    self.window - 1)) / (self.window - 1)
        self.data_anticor = []
        self.weights_projection = agent_utils.WeightsProjection(config)
        timetable = agent_utils.ExchangeTimetable(config["exchange"])
        self.data_extractor = agent_utils.DataExtractor(data_loader, 
                                timetable, config, 2 * self.window + 1, True)
        self.anticor_expert = agent_utils.anticor_expert

    def anticor_anticor_kernel(self, data, data_anticor):
        for k in range(self.window - 1):
            weights = self.anticor_expert(data, self.exp_w[k], k + 2)
            self.exp_w[k] = self.weights_projection(weights)

        for k in range(self.window - 1):
            weights = self.anticor_expert(
                np.array(data_anticor)[-self.window:],
                self.exp_w_second[k],
                k + 2
            )
            self.exp_w_second[k] = self.weights_projection(weights)
        # combine portfolios
        numerator = 0.
        denominator = 0.
        for k in range(self.window - 1):
            numerator += self.exp_ret_second[k] * self.exp_w_second[k]
            denominator += self.exp_ret_second[k]
        weights = numerator / denominator
        weights = self.exp_w.T @ weights
        return weights / np.abs(weights).sum()

    def compute_portfolio(self, epoch):
        self.n_steps += 1
        data_price_relatives = self.data_extractor(epoch)
        if self.n_steps == 1:
            data_anticor = data_price_relatives @ self.exp_w.T
            self.data_anticor = data_anticor.tolist()
        else:
            data_anticor = data_price_relatives[-1].reshape(1, -1) @ self.exp_w.T
            self.data_anticor.append(data_anticor[0])

        for k in range(self.window - 1):
            self.exp_w[k] *= data_price_relatives[-1] / self.data_anticor[-1][k]
            data_anticor_cur = np.array(self.data_anticor[-1])
            self.exp_ret_second[k] *= data_anticor_cur @ self.exp_w_second[k]
            denom = data_anticor_cur @ self.exp_w_second[k]
            self.exp_w_second[k] *= data_anticor_cur / denom

        day_weight = self.anticor_anticor_kernel(data_price_relatives, self.data_anticor)
        day_weight = self.weights_projection(day_weight)
        if self.verbose:
            print("anticor anticor weights:", day_weight)
        self.last_portfolio = day_weight
        preds_dict = {}
        for i, instrument in enumerate(self.instruments_list):
            preds_dict[instrument] = day_weight[i]
        return preds_dict
