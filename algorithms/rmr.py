import numpy as np
import pandas as pd
from technotrader.trading.agent import Agent
import technotrader.utils.agent_utils as agent_utils


class RmrAgent(Agent):
    """
    Robust Median Reversion strategy (Huang et al. [2013]).
    Link:
    https://www.ijcai.org/Proceedings/13/Papers/296.pdf
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
        if config.get("price_relatives_flag") is not None:
            price_relatives_flag = config["price_relatives_flag"]
        else:
            price_relatives_flag = False
        self.window = config["window"]
        self.data_extractor = agent_utils.DataExtractor(data_loader, 
                timetable, config, config["window"] + 1, price_relatives_flag)

    def rmr_next_weights(self, data_close):
        prices_median = agent_utils.compute_L1_median(data_close)
        x_t1 = prices_median / data_close[-1]
        denom = (np.linalg.norm(x_t1 - np.mean(x_t1))) ** 2
        if denom == 0:
            alpha = 0
        else:
            alpha = min(0, (np.dot(x_t1, self.last_portfolio) - self.epsilon) / denom)
        alpha = min(100000, alpha)
        weights = self.last_portfolio - alpha * (x_t1 - np.mean(x_t1))
        return self.weights_projection(weights)

    def compute_portfolio(self, epoch):
        data_prices = self.data_extractor(epoch)
        day_weight = self.rmr_next_weights(data_prices[-self.window:])
        if self.verbose:
            print("rmr weights:", day_weight)
        self.last_portfolio = day_weight.copy()
        preds_dict = {}
        for i, instrument in enumerate(self.instruments_list):
            preds_dict[instrument] = day_weight[i]
        return preds_dict
