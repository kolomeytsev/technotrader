import numpy as np
import logging
from technotrader.trading.trader import Trader


class BackTester(Trader):
    def __init__(self, config, data_loader, agent, trade_log=None):
        super().__init__(config, data_loader, agent, trade_log)

    def trade_by_strategy(self, weights):
        logging.info("the step is {}".format(self._steps))
        logging.debug("the raw weights is {}".format(weights))
        future_price = np.concatenate((np.ones(1), self.__get_matrix_y()))
        pv_after_commission = calculate_pv_after_commission(weights, self._last_weights, self._commission_rate)
        portfolio_change = pv_after_commission * np.dot(weights, future_price)
        self._total_capital *= portfolio_change
        self._last_weights = pv_after_commission * weights * \
                           future_price /\
                           portfolio_change
        logging.debug("the portfolio change this period is : {}".format(portfolio_change))
        self.__test_pc_vector.append(portfolio_change)
