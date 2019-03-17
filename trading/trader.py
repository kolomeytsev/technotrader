import numpy as np
import logging


class Trader:
    def __init__(self, waiting_period, config, total_steps, net_dir, 
                agent, initial_capital=1.0, agent_type="nn"):
        self._steps = 0
        self._total_steps = total_steps
        self._period = waiting_period
        
        self._agent = agent

        # the total assets is calculated with BTC
        self._total_capital = initial_capital
        self._coin_number = config["input"]["coin_number"]
        self._commission_rate = config["trading"]["trading_consumption"]
        self._fake_ratio = config["input"]["fake_ratio"]
        self._asset_vector = np.zeros(self._coin_number+1)

        self._last_weights = np.zeros((self._coin_number+1,))
        self._last_weights[0] = 1.0

        if self.__class__.__name__=="BackTest":
            # self._initialize_logging_data_frame(initial_BTC)
            self._logging_data_frame = None
            # self._disk_engine =  sqlite3.connect('./database/back_time_trading_log.db')
            # self._initialize_data_base()
        self._current_error_state = 'S000'
        self._current_error_info = ''

    def execute_trades(self, weights):
        pass

    def rolling_train(self):
        pass

    def __trade_body(self):
        self._current_error_state = 'S000'
        starttime = time.time()
        weights = self._agent.decide_by_history(self.generate_history_matrix(),
                                              self._last_weights.copy())
        self.trade_by_strategy(weights)
        if self._agent_type == "nn":
            self.rolling_train()
        if not self.__class__.__name__=="BackTest":
            self._last_weights = weights.copy()
        logging.info('total assets are %3f BTC' % self._total_capital)
        logging.debug("="*30)
        trading_time = time.time() - starttime
        if trading_time < self._period:
            logging.info("sleep for %s seconds" % (self._period - trading_time))
        self._steps += 1
        return self._period - trading_time

    def start_trading(self):
        try:
            if not self.__class__.__name__=="BackTest":
                current = int(time.time())
                wait = self._period - (current%self._period)
                logging.info("sleep for %s seconds" % wait)
                time.sleep(wait+2)

                while self._steps < self._total_steps:
                    sleeptime = self.__trade_body()
                    time.sleep(sleeptime)
            else:
                while self._steps < self._total_steps:
                    self.__trade_body()
        finally:
            if self._agent_type=="nn":
                self._agent.recycle()
            self.finish_trading()

