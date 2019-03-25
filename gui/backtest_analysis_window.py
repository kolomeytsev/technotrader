import sys
sys.path.insert(0,'../../')
import numpy as np
import pandas as pd
from PyQt5 import QtCore, QtGui, QtWidgets
import time
import datetime
from mainwindow import Ui_MainWindow
import matplotlib.pyplot as plt


class BacktestAnalysisWindow:
    def __init__(self):
        super(BacktestAnalysisWindow, self).__init__()
        self.pushButtonOpenResults.clicked.connect(self.file_open)
        print("Here")

    def file_open(self):
        name = QtWidgets.QFileDialog.getOpenFileName(self, "Open File")
        print("name:", name)
