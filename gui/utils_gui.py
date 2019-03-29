import sys
sys.path.insert(0,'../../')
import numpy as np
import pandas as pd
from PyQt5 import QtCore, QtGui, QtWidgets
import time
import copy
import datetime
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.animation import TimedAnimation
from matplotlib.lines import Line2D
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
import matplotlib.pyplot as plt
from technotrader.gui.backtest_results_window import Ui_FormPlot
from technotrader.utils.metrics import *
from technotrader.trading.constants import *
import threading


def get_time_as_name_string(start, end):
    name_string = "_" + start.replace('/', '').replace(' ', '_') + \
                  "_" + end.replace('/', '').replace(' ', '')
    return name_string


def convert_time(time_str):
    dt = datetime.datetime.strptime(
            time_str, "%Y-%m-%d %H:%M:%S").replace(tzinfo=pytz.utc)
    return int(dt.timestamp())


def convert_time_to_str(timestamp):
    dt = datetime.datetime.utcfromtimestamp(timestamp)
    return dt.strftime("%Y-%m-%d %H:%M:%S")


class Communicate(QtCore.QObject):
    data_signal = QtCore.pyqtSignal(dict)


def dataSendLoop(addData_callbackFunc, weights, time_str, speed):
    # Setup the signal-slot mechanism.
    mySrc = Communicate()
    mySrc.data_signal.connect(addData_callbackFunc)
    if speed == "medium":
        sleep_time = 0.3
    elif speed == "low":
        sleep_time = 1
    else:
        sleep_time = 0.1
    for i in range(len(weights)):
        time.sleep(sleep_time)
        data = {
            "weights": weights[i],
            "current_time": time_str[i]
        }
        mySrc.data_signal.emit(data)


class CustomFigCanvas(FigureCanvas, TimedAnimation):
    def __init__(self, speed):
        self.speed = speed
        #if self.speed == "medium":
        #    interval = 100
        #elif self.speed == "low":
        #    interval = 100
        self.addedData = None
        self.xlim = 200
        self.fig = Figure(figsize=(8, 8), dpi=100)
        self.ax1 = self.fig.add_subplot(111)
        self.ax1.set_xlabel('Assets')
        self.ax1.set_ylabel('Weights')
        self.ax1.set_ylim(-0.1, 1.1)
        self.bar = None
        FigureCanvas.__init__(self, self.fig)
        TimedAnimation.__init__(self, self.fig, interval=100, blit=False)

    def load_data(self, weights, all_instruments):
        self.y_pos = np.arange(len(all_instruments))
        self.weights = weights
        self.all_instruments = all_instruments
        self.ax1.set_xlim(-1, len(all_instruments))
        self.ax1.set_xticks(self.y_pos)
        self.ax1.set_xticklabels(self.all_instruments, rotation=45)

    def new_frame_seq(self):
        return iter(range(10))

    def _init_draw(self):
        pass

    def addData(self, value):
        self.addedData = value["weights"]
        self.current_time = value["current_time"]

    def zoomIn(self, value):
        bottom = self.ax1.get_ylim()[0]
        top = self.ax1.get_ylim()[1]
        bottom += value
        top -= value
        self.ax1.set_ylim(bottom, top)
        self.draw()

    def _step(self, *args):
        try:
            TimedAnimation._step(self, *args)
        except Exception as e:
            print(e)
            TimedAnimation._stop(self)

    def _draw_frame(self, framedata):
        if self.addedData is not None:
            print(self.addedData)
            #self.ax1.cla()
            if self.bar is not None:
                self.bar.remove()
            self.bar = self.ax1.bar(self.y_pos, self.addedData, 
                                    align='center', alpha=1, color="C0", label="agent")
            self.ax1.set_title(self.current_time)
            print("len(self.addedData)", len(self.addedData), self.current_time)
