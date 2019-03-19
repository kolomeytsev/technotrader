import numpy as np
import pandas as pd
from technotrader.trading.agent import Agent
import technotrader.utils.agent_utils as agent_utils


class EgAgent(Agent):
    """
    Exponentiated Gradient strategy (Cover [1991]; Cover and Ordentlich [1996]).
    """
    def __init__(self, config, data_loader, trade_log=None):
        super().__init__(config, data_loader)
        self.instruments_list = config["instruments_list"]
        self.n_inst = len(self.instruments_list)
        self.use_risk_free = config["use_risk_free"]
        if self.use_risk_free:
            self.n_inst += 1
        self.last_portfolio = np.ones(self.n_inst) / self.n_inst
        self.learning_rate = config['learning_rate']
        self.weights_projection = agent_utils.WeightsProjection(config)
        timetable = agent_utils.ExchangeTimetable(config["exchange"])
        self.data_extractor = agent_utils.DataExtractor(data_loader,
                                        timetable, config, 2, True)

    def eg_next_weight(self, last_x):
        weights = self.last_portfolio * np.exp(self.learning_rate * \
                        last_x / (last_x @ self.last_portfolio))
        weights = self.weights_projection(weights)
        return weights / np.abs(weights).sum()

    def compute_portfolio(self, epoch):
        data_price_relatives = self.data_extractor(epoch)
        day_weight = self.eg_next_weight(data_price_relatives[-1])
        print("eg weights:", day_weight)
        self.last_portfolio = day_weight
        preds_dict = {}
        for i, instrument in enumerate(self.instruments_list):
            preds_dict[instrument] = day_weight[i]
        return preds_dict
