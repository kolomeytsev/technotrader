import sys
sys.path.insert(0,'/Users/kolomeytsev/olps/')
import json
import importlib
import logging
import argparse
import datetime
import pytz
from technotrader.trading.backtester import BackTester
from technotrader.data.data_loader import DataLoader
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


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
    'meta': ('MetaEgOnsAgent', 'algorithms.meta_eg_ons'),
    'markowitz': ('MarkowitzAgent', 'algorithms.markowitz'),
    'min_variance': ('MinVarianceAgent', 'algorithms.min_variance'),
    'max_sharpe': ('MaxSharpeAgent', 'algorithms.max_sharpe'),
    'max_return': ('MaxReturnAgent', 'algorithms.max_return'),
    'rrl': ('RrlAgent', 'rrl.rrl_agent'),
    'sspo': ('SspoAgent', 'algorithms.sspo'),
    'ppt': ('PptAgent', 'algorithms.ppt')
}


def parse_parameters():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-a', '--agent', type=str, default='',
                        help='name of agent')
    parser.add_argument('-c', '--config', type=str,
                        help='path to config')
    parser.add_argument('-d', '--data-begin', type=str,
                        help='data start datetime')
    parser.add_argument('-b', '--begin', type=str,
                        help='backtest start datetime')
    parser.add_argument('-e', '--end', type=str,
                        help='backtest end datetime')
    parser.add_argument('-r', '--candles-res', default='hour,1', type=str,
                        help='candles resolution (default=hour,1)')
    parser.add_argument('-s', '--step', default=None, type=int,
                        help='step in seconds (default=candles-res)')
    parser.add_argument('--exchange', default='binance', type=str,
                        help='exchange (default=binance)')
    parser.add_argument('-f', '--fee', default=0.001, type=float,
                        help='exchange fee (default=0.001)')
    parser.add_argument('-p', '--path', default=None, type=str,
                        help='dump path (default=None)')
    parser.add_argument('--freq', default=1, type=int,
                        help='dump frequency fee (default=1)')
    parser.add_argument('--short', action='store_const', 
                        const=True, default=False,
                        help='using short (T/F)')
    args = parser.parse_args()
    return args


def get_agent_class(agent_name):
    if AGENTS.get(agent_name) is not None:
        class_name, from_file = AGENTS[agent_name]
        agent_class = getattr(importlib.import_module(from_file), class_name)
    else:
        print("Agent is not available")
        exit(1)
    return agent_class


def fill_agent_config(config, args):
    if args.step is not None:
        step = args.step
    else:
        step = RESOLUTIONS[args.candles_res]
    config["agent_name"] = args.agent
    config["begin"] = convert_time(args.begin)
    config["candles_res"] = args.candles_res
    config["step"] = step
    config["exchange"] = args.exchange
    config["short_flag"] = args.short


def run_agent(config, agent_config):
    exchange_name = config['exchange']['name']
    modules = config['agent']['module_name'].split('.')[1:]
    agent_module = __import__(config['agent']['module_name'])
    for module in modules:
        agent_module = agent_module.__getattribute__(module)
    agent_class = agent_module.__getattribute__(config['agent']['class_name'])
    config['data_loader']['exchange'] = exchange_name


def get_agent(args, agent_config, data_loader, backtest_config=None):    
    Agent = get_agent_class(args.agent)
    agent_name = args.agent
    if backtest_config is not None:
        if "instruments_list" in backtest_config:
            agent_config["instruments_list"] = backtest_config["instruments_list"]
    fill_agent_config(agent_config, args)
    print("agent_config:")
    print(agent_config)
    agent = Agent(agent_config, data_loader)
    return agent


def get_time_as_name_string(start, end):
    name_string = "_" + start.replace('/', '').replace(' ', '').replace(':', '') + \
                  "_" + end.replace('/', '').replace(' ', '').replace(':', '')
    return name_string


def convert_time(time_str):
    dt = datetime.datetime.strptime(time_str, "%Y-%m-%d %H:%M:%S").replace(tzinfo=pytz.utc)
    return int(dt.timestamp())


def generate_data_config(args):
    if args.step is not None:
        step = RESOLUTIONS[args.step]
    else:
        step = RESOLUTIONS[args.candles_res]
    data_name = "data"
    data_name += get_time_as_name_string(args.begin, args.end)
    data_name += '_' + str(step) + '_' + args.exchange
    config = {
        "data_name": data_name,
        "begin":  convert_time(args.data_begin),
        "end":  convert_time(args.end),
        "step": step,
        "candles_res": args.candles_res,
        "candles_res_sec": RESOLUTIONS[args.candles_res],
        "exchange": args.exchange
    }
    return config


def generate_backtest_config(args):
    if args.step is not None:
        step = args.step
    else:
        step = RESOLUTIONS[args.candles_res]
    config = {
        "begin":  convert_time(args.begin),
        "end":  convert_time(args.end),
        "step": step,
        "fee": args.fee,
        "exchange": args.exchange,
        "candles_res": args.candles_res,
        "price_label": "close",
        "log_frequency": 1,
        "dump_path": args.path,
        "dump_freq": args.freq
    }
    return config


def read_configs(args):
    with open(args.config) as f:
        agent_config = json.load(f)
    data_config = generate_data_config(args)
    backtest_config = generate_backtest_config(args)
    data_config["instruments_list"] = agent_config["instruments_list"]
    backtest_config["instruments_list"] = agent_config["instruments_list"]
    return data_config, agent_config, backtest_config


def main():
    args = parse_parameters()
    data_config, agent_config, backtest_config = read_configs(args)
    print("data config:", data_config)
    print("agent config:", agent_config)
    print("backtest config:", backtest_config)
    data_loader = DataLoader(data_config)
    agent = get_agent(args, agent_config, data_loader, backtest_config)
    backtester = BackTester(backtest_config, data_loader, agent, trade_log=None)
    backtester.run()


if __name__ == '__main__':
    main()
