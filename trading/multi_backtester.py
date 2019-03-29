import sys
sys.path.insert(0,'/Users/kolomeytsev/olps/')
import json
import importlib
import logging
import argparse
import datetime
import pytz
from technotrader.trading.backtester import BackTester
from technotrader.data_loader.data_loader import DataLoader
from technotrader.constants import *
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


class MultiBackTester:
    def __init__(self, args):
        self.args = args

    def run():
        pass
