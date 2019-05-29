import sys
sys.path.insert(0,'../')
import json
import copy
import importlib
import logging
import argparse
import datetime
from itertools import product
import pytz
import numpy as np
import pandas as pd
import multiprocessing
from multiprocessing import Pool
from technotrader.trading.backtester import BackTester
from technotrader.data_loader.data_loader import DataLoader
from technotrader.trading.constants import *
import warnings
warnings.simplefilter(action="ignore", category=FutureWarning)


def parse_parameters():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-a', '--agent', type=str, default=None,
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
                        help='using short (True/False)')
    parser.add_argument('--parallel', action='store_const', 
                        const=True, default=False,
                        help='parallel computing (True/False)')
    parser.add_argument('-s', '--processes', default=3, type=int,
                        help='number of processes in parallel (default=4)')
    args = parser.parse_args()
    return args


def get_agent_class(agent_class):
    if AGENTS.get(agent_class) is not None:
        class_name, from_file = AGENTS[agent_class]
        agent_class = getattr(importlib.import_module(from_file), class_name)
    else:
        print("Agent is not available")
        exit(1)
    return agent_class


def fill_agent_config(config, args, agent_class):
    if args.step is not None:
        step = args.step
    else:
        step = RESOLUTIONS[args.candles_res]
    config["candles_res"] = args.candles_res
    config["step"] = step
    config["exchange"] = args.exchange
    config["short_flag"] = args.short


def get_agent(agent_class, agent_config, data_loader, args=None, backtest_config=None):    
    Agent = get_agent_class(agent_class)
    if backtest_config is not None:
        if "instruments_list" in backtest_config:
            agent_config["instruments_list"] = backtest_config["instruments_list"]
    if args is not None:
        fill_agent_config(agent_config, args, agent_class)
    print("agent_config:")
    print(agent_config)
    agent = Agent(agent_config, data_loader)
    return agent


def get_time_as_name_string(start, end):
    name_string = "_" + start.replace('/', '').replace(' ', '_') + \
                  "_" + end.replace('/', '').replace(' ', '')
    return name_string


def convert_time(time_str):
    dt = datetime.datetime.strptime(
            time_str, "%Y-%m-%d %H:%M:%S").replace(tzinfo=pytz.utc)
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
        "log_frequency": 1
    }
    return config


def process_multi_parameters(parameters):
    res = []
    is_range = False
    for param in parameters:
        values = [param]
        if type(param) == str:
            if param.startswith("range"):
                splited = param.split(":")
                if len(splited) == 4:
                    is_range = True
                    start = float(splited[1])
                    end = float(splited[2])
                    step = float(splited[3])
                    values = list(range(start, end, step))
        res += values
    return res


def process_agent_config(agent_class, agent_config):
    agent_configs_multi_params = []
    agent_name = agent_class
    single_params_names = []
    single_params_values = []
    multi_params_names = []
    multi_params_values = []
    for name, value in agent_config.items():
        if name in LIST_PARAMS:
            single_params_names.append(name)
            single_params_values.append(value)
            continue
        if agent_class in LIST_PARAMS_DICT.keys():
            if name in LIST_PARAMS_DICT[agent_class]:
                if isinstance(value, list):
                    if not isinstance(value[0], list):
                        single_params_names.append(name)
                        single_params_values.append(value)
                        continue
        if isinstance(value, list):
            processed_parameters = process_multi_parameters(value)
            multi_params_names.append(name)
            multi_params_values.append(processed_parameters)
        else:
            single_params_names.append(name)
            single_params_values.append(value)
    if len(multi_params_names) == 0:
        agent_config["agent_class"] = agent_class
        agent_config["agent_name"] = agent_class
        return [(agent_class, agent_config)]
    config_first_part = {
        key: value for key, value in zip(single_params_names, single_params_values)
    }
    for current_params in product(*multi_params_values):
        print(current_params)
        new_config = copy.deepcopy(config_first_part)
        agent_name = agent_class
        for param_name, param_value in zip(multi_params_names, current_params):
            new_config[param_name] = param_value
            agent_name += '_' + param_name + '_' + str(param_value)
        agent_name = agent_name.replace(' ', '_')
        agent_name = agent_name.lower()

        new_config["agent_class"] = agent_class
        new_config["agent_name"] = agent_name
        agent_configs_multi_params.append((agent_class, new_config))
    return agent_configs_multi_params


def read_multi_backtest_configs(args, multi_agent_config):
    """
    There can be 2 options:
    1) if instruments_list is in multi_backtest.json config
    then all backtests are tested with these instruments
    2) if not then use instruments from each agent's config
    (so there can be backtests using different instruments)
    """
    agent_configs = []
    if multi_agent_config.get("instruments_list") is not None:
        all_instruments = multi_agent_config["instruments_list"]
    else:
        all_instruments = set()
    for agent_class, configs_path in zip(multi_agent_config["agents"],
                                        multi_agent_config["configs"]):
        with open(configs_path) as f:
            agent_config = json.load(f)
        if multi_agent_config.get("instruments_list") is not None:
            agent_config["instruments_list"] = all_instruments
        else:
            all_instruments.union(agent_config["instruments_list"])

        agent_configs_multi_params = process_agent_config(agent_class, agent_config)
        agent_configs.extend(agent_configs_multi_params)
    return agent_configs, list(all_instruments)


def read_configs(args):
    data_config = generate_data_config(args)
    with open(args.config) as f:
        agent_config = json.load(f)
    backtest_config = generate_backtest_config(args)
    if args.agent is None:
        agent_configs, all_instruments = read_multi_backtest_configs(
                                            args, agent_config)
    else:
        #agent_configs = [(args.agent, agent_config)]
        agent_configs = process_agent_config(args.agent, agent_config)
        all_instruments = agent_config["instruments_list"]
    data_config["instruments_list"] = all_instruments
    return data_config, agent_configs, backtest_config


def save_results_df(backtesters_results, path):
    print("saving results")
    results = {}
    print(backtesters_results)
    for agent_class, agent_name, test_pc_vector_no_fee, test_pc_vector, test_turnover_vector, epochs \
            in backtesters_results:
        results[agent_name + "_returns_no_fee"] = test_pc_vector_no_fee
        results[agent_name + "_returns_with_fee"] = test_pc_vector
        results[agent_name + "_turnover"] = test_turnover_vector
        results[agent_name + "_epochs"] = epochs
    df = pd.DataFrame.from_dict(results)
    df.to_csv(path, index=False)


def save_results(backtesters_results, data_config, agent_configs, 
                backtest_config, path, agent_name_to_config):
    print("saving results")
    results = {
        "data_config": data_config,
        "backtest_config": backtest_config,
        "agents": {}
    }
    agent_configs_dict = dict(agent_configs)
    for agent_class, agent_name, test_pc_vector_no_fee, test_pc_vector, test_turnover_vector, epochs, weights \
            in backtesters_results:
        results["agents"][agent_name] = {}
        results["agents"][agent_name]["returns_no_fee"] = list(test_pc_vector_no_fee)
        results["agents"][agent_name]["returns_with_fee"] = list(test_pc_vector)
        results["agents"][agent_name]["turnover"] = list(test_turnover_vector)
        results["agents"][agent_name]["weights"] = weights
        results["agents"][agent_name]["epochs"] = list(epochs)
        results["agents"][agent_name]["config"] = agent_name_to_config[agent_name]
    with open(path, "w") as f:
        json.dump(results, f)
    return results


def run_backtest(backtester):
    backtester.run()
    results = (
        backtester.agent.config["agent_class"],
        backtester.agent.config["agent_name"],
        backtester.test_pc_vector_no_fee,
        backtester.test_pc_vector,
        backtester.test_turnover_vector,
        backtester.test_epochs,
        backtester.weights
    )
    return results


def run_multi_backtest(data_loader, data_config, agent_configs, backtest_config,
                        args=None, path=None, parallel=False, processes_number=3):
    backtesters_list = []
    agent_name_to_config = {}
    for agent_class, agent_config in agent_configs:
        agent_name_to_config[agent_config["agent_name"]] = agent_config
        agent = get_agent(agent_class, agent_config, data_loader, args, backtest_config)
        backtester = BackTester(backtest_config, data_loader, agent, trade_log=None)
        backtesters_list.append(backtester)
    if parallel:
        if args is not None:
            processes_num = args.processes
        else:
            processes_num = processes_number
        pool = Pool(processes_num)
        results = pool.map(run_backtest, backtesters_list)
    else:
        results = list(map(run_backtest, backtesters_list))
    if path is not None:
        results = save_results(results, data_config, agent_configs, 
                                backtest_config, path, agent_name_to_config)
    return results


def main():
    args = parse_parameters()
    data_config, agent_configs, backtest_config = read_configs(args)
    print("data_config", data_config)
    data_loader = DataLoader(data_config)
    run_multi_backtest(data_loader, data_config, agent_configs, backtest_config,
                        args, args.path, args.parallel)


if __name__ == '__main__':
    main()
