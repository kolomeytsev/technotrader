import numpy as np
import pandas as pd
from technotrader.trading.agent import Agent
import technotrader.utils.agent_utils as agent_utils


class UpAgent(Agent):
    def __init__(self, config, data_loader, trade_log=None):
        super().__init__(config, data_loader)
        self.n_steps = 0
        self.instruments_list = config["instruments_list"]
        self.n_inst = len(self.instruments_list)
        self.use_risk_free = config["use_risk_free"]
        if self.use_risk_free:
            self.n_inst += 1
        self.flag = config['flag']
        self.temperature = config['temperature']
        self.weights_projection = agent_utils.WeightsProjection(config)
        timetable = agent_utils.ExchangeTimetable(config["exchange"])
        self.data_extractor = agent_utils.DataExtractor(data_loader,
                                    timetable, config, None, True)

    def up_next_weight(self, data):
        del0 = 4e-3 # minimum coordinate
        delta = 5e-3 # spacing of grid
        M = 10 # number of samples
        S = 5 # number of steps in the random walk
        N = data.shape[1]
        r = np.ones(N) / N
        b = np.ones(r.shape[0])
        allM = np.zeros((N, M))
        for m in range(M):
            b = r.copy()
            for i in range(S):
                bnew = b.copy()
                j = np.random.randint(N - 1)
                a = np.random.choice([-1, 1])
                bnew[j] = b[j] + (a * delta)
                bnew[N - 1] = b[N - 1] - (a * delta)
                if bnew[j] >= del0 and bnew[N - 1] >= del0:
                    muliplier_x = min(1, np.exp((b[N - 1] - (2 * del0)) / (N * delta)))
                    x = np.prod(data @ b) * muliplier_x
                    muliplier_y = min(1, np.exp((bnew[N - 1] - (2 * del0)) / (N * delta)))
                    y = np.prod(data @ bnew) * muliplier_y
                    pr = min(y / x, 1) # or pr = min(x / y, 1)
                    if np.random.rand() < pr:
                        b = bnew.copy()
            allM[:, m] = b
        weight = np.mean(allM, 1)
        if self.temperature is not None and self.temperature > 0:
            return sp.special.softmax(weight / self.temperature)
        return weight / np.abs(weight).sum()

    def compute_portfolio(self, epoch):
        self.n_steps += 1
        data_price_relatives = self.data_extractor(epoch, self.n_steps)
        day_weight = self.up_next_weight(data_price_relatives)
        day_weight = self.weights_projection(day_weight)
        print("up weights:", day_weight)
        preds_dict = {}
        for i, instrument in enumerate(self.instruments_list):
            preds_dict[instrument] = day_weight[i]
        return preds_dict
