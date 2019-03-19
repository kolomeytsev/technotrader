import numpy as np
import pandas as pd
import datetime
from technotrader.trading.agent import Agent
import technotrader.utils.agent_utils as agent_utils


class CrpAgent(Agent):
    """
    Constant Rebalanced Portfolios strategy (Kelly [1956]; Cover [1991]).
    """
    def __init__(self, config, data_loader, trade_log=None):
        super().__init__(config, data_loader)
        self.n_steps = 0
        self.instruments_list = config['instruments_list']
        self.n_inst = len(self.instruments_list)
        self.exchange = config['exchange']
        if config["weights"] is not None:
            self.weights = config["weights"]
        else:
            print("using uniform weights")
            self.weights = np.ones(self.n_inst) / self.n_inst
            if config["short_flag"]:
                self.weights *= -1
        self.preds_dict = {}
        for i, instrument in enumerate(self.instruments_list):
            self.preds_dict[instrument] = self.weights[i]

    def compute_portfolio(self, epoch):
        print("crp epoch:", datetime.datetime.fromtimestamp(epoch))
        return self.preds_dict
