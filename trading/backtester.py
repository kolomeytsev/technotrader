import numpy as np
import pandas as pd
import logging
from technotrader.trading.trader import Trader


def calculate_pv_after_commission(w1, w0, commission_rate):
    """
    @:param w1: target portfolio vector, first element is btc
    @:param w0: rebalanced last period portfolio vector, first element is btc
    @:param commission_rate: rate of commission fee, proportional to the transaction cost
    """
    mu0 = 1
    mu1 = 1 - 2*commission_rate + commission_rate ** 2
    while abs(mu1-mu0) > 1e-10:
        mu0 = mu1
        mu1 = (1 - commission_rate * w0[0] -
            (2 * commission_rate - commission_rate ** 2) *
            np.sum(np.maximum(w0[1:] - mu1*w1[1:], 0))) / \
            (1 - commission_rate * w1[0])
    return mu1


class BackTester(Trader):
    def __init__(self, config, data_loader, agent, trade_log=None):
        super().__init__(config, data_loader, agent, trade_log)
        self.test_pc_vector = []
        self.test_pc_vector_no_fee = []
        self.log_frequency = config["log_frequency"]
        if config.get("dump_path") is not None:
            self.dump_path = config["dump_path"]
        else:
            self.dump_path = None
        if config.get("dump_freq") is not None:
            self.dump_freq = config["dump_freq"]
        else:
            self.dump_freq = 1

    def save_results(self):
        df = pd.DataFrame.from_dict({
            "returns_no_fee": self.test_pc_vector_no_fee,
            "returns_with_fee": self.test_pc_vector
        })
        df.to_csv(self.dump_path, index=False)

    def finish_trading(self):
        if self.dump_path is not None:
            self.save_results()

    def get_next_valid_price(self, epoch):
        return self.data_loader.get_data([epoch])[epoch]

    def get_future_price_relatives(self, epoch):
        current_prices = self.get_next_valid_price(epoch)
        next_prices = self.get_next_valid_price(epoch + self.step)
        current_prices = np.array([current_prices[x] for x in self.relevant_columns])
        next_prices = np.array([next_prices[x] for x in self.relevant_columns])
        future_price_relatives = next_prices / current_prices
        return future_price_relatives

    def execute_trades(self, weights):
        if self.steps % self.log_frequency == 0:
            logging.info("the step is {}".format(self.steps))
        logging.debug("the raw weights is {}".format(weights))
        future_price_relatives = self.get_future_price_relatives(self.current_epoch)
        future_price_relatives = np.concatenate((future_price_relatives, np.ones(1)))
        pv_after_commission = calculate_pv_after_commission(
            weights,
            self.last_weights,
            self.commission_rate
        )
        portfolio_change_no_fee = np.dot(weights, future_price_relatives)
        portfolio_change = portfolio_change_no_fee * pv_after_commission
        self.total_capital *= portfolio_change
        self.last_weights = pv_after_commission * weights * \
                           future_price_relatives / \
                           portfolio_change
        logging.debug(
            "the portfolio change this period is : {}".format(portfolio_change)
        )
        self.test_pc_vector.append(portfolio_change)
        self.test_pc_vector_no_fee.append(portfolio_change_no_fee)
        if self.dump_path is not None and self.step % self.dump_freq == 0:
            self.save_results()
