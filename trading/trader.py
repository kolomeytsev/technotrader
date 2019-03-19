import numpy as np
import logging
import time


class Trader:
    def __init__(self, config, data_loader, agent, trade_log=None,
                initial_capital=1.0, waiting_period=0):   
        self.config = config
        self.data_loader = data_loader
        self.agent = agent
        self.trade_log = trade_log
        self.instruments_list = self.agent.instruments_list
        self.price_label = config["price_label"]
        self.relevant_columns = [
            ">".join([
                config["exchange"],
                label,
                config["candles_res"],
                self.price_label
            ])
            for label in self.instruments_list
        ]
        self.steps = 0
        self.total_capital = initial_capital
        self.waiting_period = waiting_period
        self.coin_number = len(self.instruments_list)
        self.commission_rate = config["fee"]
        self.asset_vector = np.zeros(self.coin_number + 1)
        self.last_weights = np.zeros((self.coin_number + 1,))
        self.last_weights[-1] = 1.0

        if self.__class__.__name__=="BackTester":
            # self._initialize_logging_data_frame(initial_BTC)
            self.logging_data_frame = None
            # self._disk_engine =  sqlite3.connect('./database/back_time_trading_log.db')
            # self._initialize_data_base()
        self.current_error_state = 'S000'
        self.current_error_info = ''

        self.begin_epoch = config["begin"]
        self.current_epoch = self.begin_epoch
        self.end_epoch = config["end"]
        self.step = config["step"]
        #columns_list = ["return", "turnover"]
        #self.full_result = {x : [] for x in columns_list} 

    def execute_trades(self, weights):
        pass

    def transform_weights_dict_to_array(self, weights):
        weights_array = np.array([weights[inst] for inst in self.instruments_list])
        risk_free_ampunt = 1 - np.abs(weights_array).sum()
        weights_array = np.append(weights_array, risk_free_ampunt)
        return weights_array

    def trade_body(self):
        self.current_error_state = 'S000'
        starttime = time.time()
        weights = self.agent.compute_portfolio(self.current_epoch)
        weights = self.transform_weights_dict_to_array(weights)
        self.execute_trades(weights)
        if not self.__class__.__name__=="BackTester":
            self.last_weights = weights.copy()
        logging.info('total assets are %3f' % self.total_capital)
        logging.debug("=" * 30)
        trading_time = time.time() - starttime
        if trading_time < self.waiting_period:
            logging.info("sleep for %s seconds" % (self.waiting_period - trading_time))
        self.steps += 1
        return self.waiting_period - trading_time

    def check_epoch_in_timetable(self, epoch):
        return True

    def run(self):
        try:
            if not self.__class__.__name__ == "BackTester":

                raise NotImplementedError()

                current = int(time.time())
                wait = self.waiting_period - (current % self.waiting_period)
                logging.info("sleep for %s seconds" % wait)
                time.sleep(wait + 2)

                while self.current_epoch < self.end_epoch:
                    sleeptime = self.trade_body()
                    time.sleep(sleeptime)
            else:
                while self.current_epoch < self.end_epoch - self.step:
                    if self.check_epoch_in_timetable(self.current_epoch):
                        self.trade_body()
                    self.current_epoch += self.step
        finally:
            self.finish_trading()
