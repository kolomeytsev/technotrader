import pandas as pd
import numpy as np
import os
import json
import sys
import subprocess


def parse_parameters():
    if len(sys.argv) != 6:
        print("Usage:\npython3 fit_multiple_rl_agents.py" + \
            " rl_configs/fit_multiple_net_config.json rl_configs/fit_net_config.json prediction_range exchange save_path")
        exit(1)
    with open(sys.argv[1]) as f:
        config = json.load(f)
    fit_config_path = sys.argv[2]
    prediction_range = sys.argv[3].lower()
    exchange = sys.argv[4]
    save_path = sys.argv[5]
    return config, fit_config_path, prediction_range, exchange, save_path


def fit_agents(periods, fit_config_path, prediction_range, exchange, save_path):
    processes = []
    for start, end in periods:
        train_file_name = "_".join(["train_package", start, end, prediction_range, exchange])
        train_file_name = train_file_name.replace("/", "").replace(",", "")
        run_config = {
            "fit_config_path": fit_config_path,
            "prediction_range": prediction_range,
            "start": start,
            "end": end,
            "exchange": exchange,
            "train_dir": save_path + "/" + train_file_name
        }
        command = "python3 fit_rl_agent.py {fit_config_path} {prediction_range} {start} {end} {exchange} {train_dir}".format(
            **run_config
        )
        logs_dir = "logs/" + save_path + "/"
        if not os.path.exists(logs_dir):
            os.makedirs(logs_dir)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        print("Running:")
        print(command)
        print("Output:", logs_dir + train_file_name)
        f = open(logs_dir + train_file_name, "w")
        p = subprocess.Popen(command, shell=True, stdout=f)
        processes.append((p, f))
    return processes


def main():
    print(subprocess.check_output('echo $PATH', shell=True))
    config, fit_config_path, prediction_range, exchange, save_path = parse_parameters()
    processes = fit_agents(config["periods"], fit_config_path, prediction_range, exchange, save_path)
    print("Training started")
    for p, f in processes:
        p.wait()
    print("Training finished")
    for p, f in processes:
        f.close()


if __name__ == "__main__":
    main()
