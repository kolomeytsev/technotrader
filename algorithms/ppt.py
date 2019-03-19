import numpy as np
import pandas as pd
from technotrader.trading.agent import Agent
import technotrader.utils.agent_utils as agent_utils


class PptAgent(Agent):
    """
    A Peak Price Tracking-Based Learning System
    for Portfolio Selection

    Zhao-Rong Lai et al.
    IEEE TRANSACTIONS ON NEURAL NETWORKS AND LEARNING SYSTEMS 2017
    """
    def __init__(self, config, data_loader, trade_log=None):
        super().__init__(config, data_loader)
        self.n_step = 0
        self.instruments_list = config["instruments_list"]
        self.window = config["window"]
        self.epsilon = config["epsilon"]
        use_risk_free = config["use_risk_free"]
        self.n_inst = len(self.instruments_list)
        if use_risk_free:
            self.n_inst += 1
        self.last_portfolio = np.ones(self.n_inst) / self.n_inst
        self.weights_projection = agent_utils.WeightsProjection(config)
        timetable = agent_utils.ExchangeTimetable(config["exchange"])
        self.data_extractor = agent_utils.DataExtractor(data_loader,
                                timetable, config, config["window"] + 1, False)

    def ppt_next_weight(self, data_close):
        nstk = data_close.shape[1]
        closepredict = np.max(data_close, axis=0)
        x_tplus1 = closepredict / data_close[-1]
        x_tplus1_cent = (np.eye(self.n_inst) - 1 / self.n_inst) @ x_tplus1.T
        daily_port = self.last_portfolio.copy()
        x_tplus1_cent_norm = np.linalg.norm(x_tplus1_cent)
        if x_tplus1_cent_norm != 0:
            daily_port = daily_port + self.epsilon * x_tplus1_cent / x_tplus1_cent_norm
        return self.weights_projection(daily_port)

    def compute_portfolio(self, epoch):
        self.n_step += 1
        data_prices = self.data_extractor(epoch)
        day_weight = self.ppt_next_weight(data_prices)
        print("ppt weights:", day_weight)
        self.last_portfolio = day_weight.copy()
        preds_dict = {}
        for i, instrument in enumerate(self.instruments_list):
            preds_dict[instrument] = day_weight[i]
        return preds_dict
