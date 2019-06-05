import numpy as np
import pandas as pd
from technotrader.trading.agent import Agent
import technotrader.utils.agent_utils as agent_utils


class CfrOgdAgent(Agent):
    """
    Combination Forecasting Reversion Strategy (Huang et. al. June 2018).
    This version is based on Online Gradient Descent.

    Based on article:
    Combination Forecasting Reversion Strategy (Huang et. al. June 2018).
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
        self.arima_class = agent_utils.Arima
        self.simplex_projection = agent_utils.simplex_projection
        self.compute_L1_median = agent_utils.compute_L1_median
        timetable = agent_utils.ExchangeTimetable(config["exchange"])
        self.weights_projection = agent_utils.WeightsProjection(config)
        self.epsilon = config['epsilon']
        self.window = config['window']
        self.arma = self.init_arima_model(config["config_arma"])
        self.arima = self.init_arima_model(config["config_arima"])
        self.predicted_prices = 0
        self.estim_num = config["estim_num"]
        self.estimators_weights = np.ones(self.estim_num) / self.estim_num
        self.learning_rate = config["learning_rate"]
        self.max_window = max([
            self.window,
            self.arma.window + 1,
            self.arima.window + 2
        ])
        self.data_extractor = agent_utils.DataExtractor(data_loader, 
                                    timetable, config, self.max_window + 1, False)
        if config.get("save_flag") is not None:
            self.save_flag = config["save_flag"]
        else:
            self.save_flag = False
        if self.save_flag:
            self.history_losses = []
            self.history_weighted_losses = []
            self.history_estimators_weights = []
            self.history_next_price_relatives = []
            self.history_predicted_prices = []

    def init_arima_model(self, config):
        if config.get("init_weights") is not None:
            init_weights = config["init_weights"]
        else:
            init_weights = np.arange(1, config["window"] + 1) / config["init_weights_denom"]
        if config.get("lr_decrease_power") is not None:
            lr_decrease_power = config["lr_decrease_power"]
        else:
            lr_decrease_power = None
        if config.get("ons_eps") is not None:
            ons_eps = config["ons_eps"]
        else:
            ons_eps = None
        return self.arima_class(window=config['window'], epsilon=ons_eps,
                    learning_rate=config["lr"], method=config["method"],
                    init_weights=init_weights,
                    lr_decrease_power=lr_decrease_power, verbose=self.verbose)

    def save_results(self, save_path):
        with open(save_path + "/losses", 'w') as f:
            json.dump(self.history_losses, f)
        with open(save_path + "/weighted_losses", 'w') as f:
            json.dump(self.history_weighted_losses, f)
        with open(save_path + "/estimators_weights", 'w') as f:
            json.dump(self.history_estimators_weights, f)
        with open(save_path + "/next_price_relatives", 'w') as f:
            json.dump(self.history_next_price_relatives, f)
        with open(save_path + "/predicted_prices", 'w') as f:
            json.dump(self.history_predicted_prices, f)

    def compute_loss(self, y_true, y_pred):
        return np.mean((y_true - y_pred)**2)

    def compute_loss_grad(self, y_true, y_pred, predicted_prices):
        loss = (y_pred - y_true)[:, np.newaxis] * predicted_prices
        return 2 * np.mean(loss, axis=0)

    def predict_prices(self, prices):
        """
        Predicts next prices using several algorithms:
        L1_median, moving_average, online ARMA and ARIMA
        Resulting matrix sizes:
        rows - number of instruments,
        columns - number of algorithms.
        """
        if self.arma.lr_decrease_power is not None:
            arma_denom = self.n_steps
        else:
            arma_denom = 1
        if self.arima.lr_decrease_power is not None:
            arima_denom = self.n_steps
        else:
            arima_denom = 1
        instruments_number = prices.shape[1]
        predicted_prices = np.zeros((instruments_number, 
                                    self.estim_num))
        predicted_prices[:, 0] = self.compute_L1_median(prices[-self.window:])
        predicted_prices[:, 1] = prices[-self.window:].mean(axis=0)
        arma_next_prices = []
        if self.verbose:
            print("\narma:")
        for i in range(instruments_number):
            X = prices[-(self.arma.window + 1):, i]
            next_price = self.arma.predict_next_price(X, arma_denom)
            arma_next_prices.append(next_price)
        if self.verbose:
            print("arma weights:", self.arma.weights)
        predicted_prices[:, 2] = np.array(arma_next_prices)
        if self.estim_num >= 4:
            arima_next_prices = []
            if self.verbose:
                print("\narima:")
            for i in range(instruments_number):
                X = prices[-(self.arima.window + 2):, i]
                diff = np.diff(X)
                next_price_diff = self.arima.predict_next_price(diff, arima_denom)
                next_price = next_price_diff + X[-1]
                arima_next_prices.append(next_price)
            predicted_prices[:, 3] = np.array(arima_next_prices)
            if self.verbose:
                print("arima weights:", self.arima.weights)
        return predicted_prices

    def cfr_ogd_next_price_relatives(self, prices):
        prices_step = prices[-self.max_window - 1:-1]
        prices_current = prices[-1]
        predicted_prices = self.predict_prices(prices_step)
        weighted_prices = np.sum(predicted_prices * self.estimators_weights, axis=1)
        estimators_losses = []
        for i in range(self.estim_num):
            loss = self.compute_loss(prices_current, predicted_prices[:, i])
            estimators_losses.append(loss)

        weighted_loss = self.compute_loss(prices_current, weighted_prices)
        loss_grad = self.compute_loss_grad(prices_current, 
                                    weighted_prices, predicted_prices)
        self.estimators_weights -= self.learning_rate * loss_grad
        self.estimators_weights = self.simplex_projection(self.estimators_weights)
        self.estimators_weights /= sum(self.estimators_weights)
        if self.verbose:
            print("\ncfr weights:\n", self.estimators_weights, '\n')

        predicted_prices = self.predict_prices(prices[-self.max_window:])        
        next_price_relatives = np.sum(predicted_prices * \
                                    self.estimators_weights, axis=1) / prices[-1]
        if self.save_flag:
            self.history_losses.append(estimators_losses)
            self.history_weighted_losses.append(weighted_loss)
            self.history_estimators_weights.append(self.estimators_weights.tolist())
            self.history_predicted_prices.append(predicted_prices.tolist())
            self.history_next_price_relatives.append(next_price_relatives.tolist())
        return next_price_relatives

    def predict_next_weight(self, prices):
        if prices.shape[0] < self.window + 1:
            next_price_relative = prices[-1] / prices[-2]
        else:
            next_price_relative = self.cfr_ogd_next_price_relatives(prices)
        numerator = min([0, next_price_relative @ self.last_portfolio - self.epsilon])
        x_bar = np.mean(next_price_relative)
        denom_part = next_price_relative - x_bar
        denominator = np.dot(denom_part, denom_part)
        if denominator != 0:
            lmbd = numerator / denominator
        else:
            lmbd = 0
        weights = self.last_portfolio - lmbd * (next_price_relative - x_bar)
        return self.weights_projection(weights)

    def compute_portfolio(self, epoch):
        self.n_steps += 1
        data_prices = self.data_extractor(epoch)
        data_prices /= data_prices[0]
        day_weight = self.predict_next_weight(data_prices)
        if self.verbose:
            print("cfr-ogd weights:", day_weight)
        self.last_portfolio = day_weight
        preds_dict = {}
        for i, instrument in enumerate(self.instruments_list):
            preds_dict[instrument] = day_weight[i]
        return preds_dict
