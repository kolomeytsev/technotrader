class Agent:
    """
    Parent class for all algorithms.
    Define compute_portfolio method in your agent,
    this method is called on every step of backtesting
    to get new weights predictions.
    """
    def __init__(self, config, data_loader, trade_log=None):
        self.config = config
        self.data_loader = data_loader
        self.trade_log = trade_log
        if config.get("verbose") is not None:
            self.verbose = config["verbose"]
        else:
            self.verbose = False

    def compute_portfolio(self, epoch):
        raise NotImplementedError()
