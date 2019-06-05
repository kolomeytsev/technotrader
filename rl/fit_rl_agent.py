import pandas as pd
import numpy as np
import os
import json
import sys
import logging
import rllib.tools.configprocess
from rllib.learn.rollingtrainer import RollingTrainer
from rllib.learn.tradertrainer import TraderTrainer
from rllib.tools.configprocess import load_config


PREDICTION_RANGE_DICT = {
    "minute,10": 600,
    "minute,15": 900,
    "minute,30": 1800,
    "hour,1": 3600,
    "hour,2": 7200,
    "hour,4": 14400,
    "hour,8": 14400 * 2,
    "day,1": 86400
}


def get_last_candles_params(candles_res, exchange_name, features, instruments_list):
    candles_res_formatted = candles_res.replace(',', '')
    relevant_columns = []
    for price_label in features:
        str_addition = '>' + candles_res_formatted + '>' + price_label
        relevant_columns_for_price_label = [exchange_name + '>' + x + str_addition \
                                               for x in instruments_list]
        relevant_columns.append(relevant_columns_for_price_label)
    last_candles_params = (exchange_name, features, candles_res, relevant_columns)
    return last_candles_params


def fit_agent(net_config, last_candles_params, device, train_dir=None, processes=1):
    """
    Fitting agent based on Reinforcement Learning

    based on article:
    "A Deep Reinforcement Learning Framework for the Financial Portfolio Management Problem"
    by Zhengyao Jiang, Dixing Xu, Jinjun Liang (2017)
    """
    if processes == 1:
        console_level = logging.INFO
        logfile_level = logging.DEBUG
    else:
        console_level = logging.WARNING
        logfile_level = logging.INFO
    if train_dir is None:
        print("train_dir is not defined")
        print("Using default dir: train_package")
        train_dir = "train_package"

    if not os.path.exists("./" + train_dir):
        #if the directory does not exist, creates one
        os.makedirs("./" + train_dir)
        os.makedirs("./" + train_dir + '/1')
    all_subdir = os.listdir("./" + train_dir)
    all_subdir.sort()
    print("dirs for train:", all_subdir)
    print("using the first one")
    dir = all_subdir[0]
    if not str.isdigit(dir):
        print("dir is not a digit")
        exit(1)
    if (os.path.isdir("./" + train_dir + "/" + dir + "/tensorboard") or \
                os.path.isdir("./" + train_dir + "/" + dir + "/logfile")):
        print("train is already done")
        exit(1)
    directory = "./" + train_dir + '/1'
    with open(directory + "/" + "net_config.json", 'w') as outfile:
        json.dump(net_config, outfile, indent=4, sort_keys=True)
    save_path = "./" + train_dir + "/" + dir + "/netfile"
    log_file_dir = "./" + train_dir + "/" + dir + "/tensorboard"
    index = dir
    if log_file_dir:
        logging.basicConfig(filename=log_file_dir.replace("tensorboard", "programlog"),
                            level=logfile_level)
        console = logging.StreamHandler()
        console.setLevel(console_level)
        logging.getLogger().addHandler(console)
    print("training at %s started" % index)
    trader_trainer = TraderTrainer(net_config, last_candles_params,
                                    save_path=save_path, device=device)
    train_res = trader_trainer.train_net(log_file_dir=log_file_dir, index=index)
    return train_res


def parse_parameters():
    if len(sys.argv) < 6 or len(sys.argv) > 10:
        print("Usage:\npython fit_rl_agent.py fit_net_config.json prediction_range start end market [train_dir] [data_path]")
        exit(1)
    net_config = rllib.tools.configprocess.load_config(sys.argv[1])
    prediction_range = sys.argv[2].lower()
    net_config["input"]["start_date"] = sys.argv[3]
    net_config["input"]["end_date"] = sys.argv[4]
    net_config["input"]["market"] = sys.argv[5]
    net_config["input"]["global_period"] = PREDICTION_RANGE_DICT[prediction_range]
    train_dir = None
    if len(sys.argv) >= 7:
        train_dir = sys.argv[6]
    if len(sys.argv) >= 8:
        net_config["input"]["data_path"] = sys.argv[7]
    if len(sys.argv) >= 9:
        net_config["input"]["window_size"] = int(sys.argv[8])
    if len(sys.argv) >= 10:
        device = sys.argv[9]
    else:
        device = "cpu"
    return net_config, prediction_range, train_dir, device


def main():
    net_config, prediction_range, train_dir, device = parse_parameters()
    print("Using device: ", device)
    last_candles_params = get_last_candles_params(prediction_range,
                                                net_config["input"]["market"],
                                                net_config["input"]["features"],
                                                net_config["input"]["coins_list"])
    fit_agent(net_config, last_candles_params, device, train_dir)
    print("training finished")


if __name__ == "__main__":
    main()
