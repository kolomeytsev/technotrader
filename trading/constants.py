from os import path

DATABASE_DIR = "database/Data.db"

NOW = 0
FIVE_MINUTES = 60 * 5
FIFTEEN_MINUTES = FIVE_MINUTES * 3
HALF_HOUR = FIFTEEN_MINUTES * 2
HOUR = HALF_HOUR * 2
TWO_HOUR = HOUR * 2
FOUR_HOUR = HOUR * 4
EIGHT_HOUR = HOUR * 8
TEN_HOUR = HOUR * 10
TWELVE_HOUR = HOUR * 12
DAY = HOUR * 24
YEAR = DAY * 365


RESOLUTIONS = {
    "minute,1": 60,
    "minute,5": 300,
    "minute,15": 900,
    "minute,30": 1800,
    "minute,512": 31200,
    "hour,1": 3600,
    "hour,2": 7200,
    "hour,4": 14400,
    "hour,8": 28800,
    "day,1": 86400
}


AGENTS = {
    'eg': ('EgAgent', 'algorithms.eg'),
    'up': ('UpAgent', 'algorithms.up'),
    'crp': ('CrpAgent', 'algorithms.crp'),
    'ons': ('OnsAgent', 'algorithms.ons'),
    'anticor': ('AnticorAgent', 'algorithms.anticor'),
    'bah_anticor': ('BahAnticorAgent', 'algorithms.bah_anticor'),
    'anticor_anticor': ('AnticorAnticorAgent', 'algorithms.anticor_anticor'),
    'bmar1': ('Bmar1Agent', 'algorithms.bmar1'),
    'bmar2': ('Bmar2Agent', 'algorithms.bmar2'),
    'olmar1': ('Olmar1Agent', 'algorithms.olmar1'),
    'olmar2': ('Olmar2Agent', 'algorithms.olmar2'),
    'rmr': ('RmrAgent', 'algorithms.rmr'),
    'rmr_trend_follow': ('RmrTrendFollowAgent', 'algorithms.rmr_trend_follow'),
    'pamr': ('PamrAgent', 'algorithms.pamr'),
    'wmamr': ('WmamrAgent', 'algorithms.wmamr'),
    'cwmr_var': ('CwmrVarAgent', 'algorithms.cwmr_var'),
    'cwmr_std': ('CwmrStdAgent', 'algorithms.cwmr_std'),
    'corn': ('CornAgent', 'algorithms.corn'),
    'bnn': ('BnnAgent', 'algorithms.bnn'),
    'cfr_ogd': ('CfrOgdAgent', 'algorithms.cfr_ogd'),
    'cfr_ons': ('CfrOnsAgent', 'algorithms.cfr_ons'),
    'rl': ('RLAgent', 'rllib.rl_agent'),
    'arma': ('ArmaAgent', 'algorithms.arma'),
    'arima': ('ArimaAgent', 'algorithms.arima'),
    'trend_follow': ('TrendFollowAgent', 'algorithms.trend_follow'),
    'hmm_trend_follow': ('HmmTrendFollowAgent', 'algorithms.hmm_trend_follow'),
    'bollinger': ('BollingerAgent', 'algorithms.bollinger'),
    'meta_eg': ('MetaEgOnsAgent', 'algorithms.meta_eg_ons'),
    'meta_ons': ('MetaEgOnsAgent', 'algorithms.meta_eg_ons'),
    'markowitz': ('MarkowitzAgent', 'algorithms.markowitz'),
    'min_variance': ('MinVarianceAgent', 'algorithms.min_variance'),
    'max_sharpe': ('MaxSharpeAgent', 'algorithms.max_sharpe'),
    'max_return': ('MaxReturnAgent', 'algorithms.max_return'),
    'rrl': ('RrlAgent', 'rrl.rrl_agent'),
    'sspo': ('SspoAgent', 'algorithms.sspo'),
    'ppt': ('PptAgent', 'algorithms.ppt')
}


LIST_PARAMS = ["instruments_list", "price_label"]


LIST_PARAMS_DICT = {
    "bmar2": ["alphas"],
    "rl": ["layers"]
}
