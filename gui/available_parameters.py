EXCHANGES_DATASETS = [
    "poloniex",
    "binance", 
    "NYSE(O)",
    "NYSE(N)",
    "MSCI",
    "TSE",
    "SP500",
    "DJIA"
]


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

PROJECTIONS_METHODS = [
    "simplex",
    "uniform",
    "top_k",
    "identical"
]

PRICE_LABELS = [
    "close",
    "open",
    "high",
    "low"
]
