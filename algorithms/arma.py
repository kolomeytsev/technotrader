import numpy as np
import pandas as pd
from technotrader.trading.agent import Agent
import technotrader.utils.agent_utils as agent_utils


class ArmaAgent(Agent):
    """
    Online Arma agent

    based on article:
    Chenghao Liu, Steven C. H. Hoi, Peilin Zhao, and Jianling Sun. 2016.
    Online AIRMA algorithms for time series prediction. 
    In Proceedings of the 30th AAAI Conference on Artificial Intelligence

    link:
    https://ink.library.smu.edu.sg/cgi/viewcontent.cgi?article=4620&context=sis_research
    """
    def __init__(self, config, data_loader, trade_log=None):
        super().__init__(config, data_loader)
        self.n_steps = 0
        self.instruments_list = config["instruments_list"]
        self.epsilon = config["epsilon"]
        self.window = config["window"]
        use_risk_free = config["use_risk_free"]
        self.n_inst = len(self.instruments_list)
        if use_risk_free:
            self.n_inst += 1
        self.last_portfolio = np.ones(self.n_inst) / self.n_inst
        self.data_phi = np.zeros(self.n_inst)
        self.weights_projection = agent_utils.WeightsProjection(config)
        timetable = agent_utils.ExchangeTimetable(config["exchange"])
        self.data_extractor = agent_utils.DataExtractor(data_loader,
                                    timetable, config, self.window + 2, False)
        self.arima_class = agent_utils.Arima
        self.init_arma_model()

    def init_arma_model(self):
        if self.config.get("init_weights") is not None:
            init_weights = self.config["init_weights"]
        else:
            init_weights = np.arange(1, self.window + 1) / \
                            self.config["init_weights_denom"]
        if self.config.get("lr_decrease_power") is not None:
            lr_decrease_power = self.config["lr_decrease_power"]
        else:
            lr_decrease_power = None
        if self.config.get("ons_eps") is not None:
            ons_eps = self.config["ons_eps"]
        else:
            ons_eps = None
        self.arma = self.arima_class(
            window=self.config['window'], 
            epsilon=ons_eps,
            learning_rate=self.config["lr"], 
            method=self.config["method"], 
            init_weights=init_weights,
            lr_decrease_power=lr_decrease_power
        )

    def predict_next_weight(self, data_close):
        next_prices = []
        if self.arma.lr_decrease_power is not None:
            denom = self.n_steps**self.arma.lr_decrease_power
        else:
            denom = 1
        for i in range(self.n_inst):
            X = data_close[-(self.arma.window + 1):, i]
            next_price = self.arma.predict_next_price(X, denom)
            next_prices.append(next_price)
        next_prices = np.array(next_prices).flatten()
        next_price_relative = next_prices / data_close[-1]
        numerator = max([0, self.epsilon - next_price_relative @ self.last_portfolio])
        x_bar = np.mean(next_price_relative)
        denom_part = next_price_relative - x_bar
        denominator = np.dot(denom_part, denom_part)
        if denominator != 0:
            lmbd = numerator / denominator
        else:
            lmbd = 0
        weights = self.last_portfolio + lmbd * (next_price_relative - x_bar)
        return self.weights_projection(weights)

    def compute_portfolio(self, epoch):
        self.n_steps += 1
        data_prices = self.data_extractor(epoch)
        data_prices /= data_prices[0]
        day_weight = self.predict_next_weight(data_prices)
        if self.verbose:
            print("predicted weights:", day_weight)
            print("arma model weights:", self.arma.weights)
        self.last_portfolio = day_weight
        preds_dict = {}
        for i, instrument in enumerate(self.instruments_list):
            preds_dict[instrument] = day_weight[i]
        return preds_dict
