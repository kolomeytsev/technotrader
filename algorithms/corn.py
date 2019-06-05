import numpy as np
import pandas as pd
from technotrader.trading.agent import Agent
import technotrader.utils.agent_utils as agent_utils


class CornAgent(Agent):
    """
    Correlation-driven Nonparametric Learning strategy (Li et al. [2011]).
    https://ink.library.smu.edu.sg/cgi/viewcontent.cgi?article=3265&context=sis_research
    """
    def __init__(self, config, data_loader, trade_log=None):
        super().__init__(config, data_loader)
        self.n_steps = 0
        self.instruments_list = config['instruments_list']
        self.n_inst = len(self.instruments_list)
        self.window = config['window']
        self.correlation_threshold = config['correlation_threshold']
        self.history_start_date = config.get("history_start_date")
        self.init_history_length = config.get("init_history_length")
        self.use_risk_free = config["use_risk_free"]
        if self.use_risk_free:
            self.n_inst += 1
        self.last_portfolio = np.ones(self.n_inst) / self.n_inst
        self.optimize_weights = agent_utils.optimize_weights
        self.weights_projection = agent_utils.WeightsProjection(config)
        timetable = agent_utils.ExchangeTimetable(config["exchange"])
        self.data_extractor = agent_utils.DataExtractor(data_loader,
                                    timetable, config, None, True)

    def get_init_history(self, epoch):
        if self.init_history_length is None:
            self.init_history_length = 0
        if self.init_history_length == 0 and self.history_start_date is not None:
            start_date = datetime.datetime.strptime(self.history_start_date, "%Y-%m-%d %H:%M:%S")
            end_date = datetime.datetime.fromtimestamp(epoch)
            if end_date < start_date:
                print("Error: history_start_date is greater than current_time")
                exit(1)
            period_lengh = step
            diff_seconds = (end_date - start_date) / datetime.timedelta(seconds=1)
            self.init_history_length = diff_seconds // period_lengh

    def corn_expert(self, data):
        T, N = data.shape
        if T <= self.window:
            weight = np.ones(N) / N
            return weight
        if self.window == 0:
            histdata = data.copy()
        else:
            indices = []
            d2 = data[-self.window:].flatten()
            for i in range(self.window, T):
                d1 = data[i - self.window:i].flatten()
                if np.corrcoef(d1, d2)[0, 1] >= self.correlation_threshold:
                    indices.append(i)
            histdata = data[indices]
        if len(histdata) == 0:
            weight = np.ones(N) / N
        else:
            weight = self.optimize_weights(histdata)
        return weight / np.abs(weight).sum()

    def compute_portfolio(self, epoch):
        if self.n_steps == 0:
            self.get_init_history(epoch)
        self.n_steps += 1
        window = self.init_history_length + self.window + self.n_steps
        data_price_relatives = self.data_extractor(epoch, window)
        day_weight = self.corn_expert(data_price_relatives)
        day_weight = self.weights_projection(day_weight)
        if self.verbose:
            print("corn weights:", day_weight)
        self.last_portfolio = day_weight
        preds_dict = {}
        for i, instrument in enumerate(self.instruments_list):
            preds_dict[instrument] = day_weight[i]
        return preds_dict
