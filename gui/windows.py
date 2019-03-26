import sys
sys.path.insert(0,'../../')
import numpy as np
import pandas as pd
from PyQt5 import QtCore, QtGui, QtWidgets
import time
import datetime
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
import matplotlib.pyplot as plt
from technotrader.gui.backtest_results_window import Ui_FormPlot
from technotrader.utils.metrics import *
from technotrader.trading.constants import *


class WindowRunningBacktest(QtWidgets.QDialog):
    def __init__(self):
        QtWidgets.QDialog.__init__(self)
        self.setObjectName("Dialog")
        self.resize(327, 103)
        self.formLayout = QtWidgets.QFormLayout(self)
        self.formLayout.setObjectName("formLayout")
        self.labelBacktesting = QtWidgets.QLabel(self)
        self.labelBacktesting.setMaximumSize(QtCore.QSize(16777215, 20))
        font = QtGui.QFont()
        font.setPointSize(15)
        self.labelBacktesting.setFont(font)
        self.labelBacktesting.setObjectName("labelBacktesting")
        self.formLayout.setWidget(0, QtWidgets.QFormLayout.LabelRole, self.labelBacktesting)
        self.pushButtonCancel = QtWidgets.QPushButton(self)
        self.pushButtonCancel.setMaximumSize(QtCore.QSize(100, 16777215))
        self.pushButtonCancel.setObjectName("pushButtonCancel")
        self.formLayout.setWidget(2, QtWidgets.QFormLayout.LabelRole, self.pushButtonCancel)
        self.progressBarBacktesting = QtWidgets.QProgressBar(self)
        self.progressBarBacktesting.setProperty("value", 24)
        self.progressBarBacktesting.setObjectName("progressBarBacktesting")
        self.formLayout.setWidget(1, QtWidgets.QFormLayout.SpanningRole, self.progressBarBacktesting)
        self.retranslateUi(self)
        QtCore.QMetaObject.connectSlotsByName(self)
        self.show_dialog()

    def retranslateUi(self, Dialog):
        _translate = QtCore.QCoreApplication.translate
        Dialog.setWindowTitle(_translate("Dialog", "Dialog"))
        self.labelBacktesting.setText(_translate("Dialog", "Backtesting in Progress:"))
        self.pushButtonCancel.setText(_translate("Dialog", "Cancel"))

    def show_dialog(self):
        self.setWindowTitle('Backtesting')
        #self.button = QtWidgets.QPushButton('Cancel')
        self.pushButtonCancel.clicked.connect(self.close)
        self.show()
        self.run_progress_bar()

    def run_progress_bar(self):
        self.completed = 0
        while self.completed < 100:
            self.completed += 1
            self.progressBarBacktesting.setValue(self.completed)
            QtWidgets.QApplication.processEvents()
            time.sleep(0.01)


class BacktestResultsWindow(QtWidgets.QWidget, Ui_FormPlot):
    def __init__(self, data_loader, agents_names, results):
        super(BacktestResultsWindow, self).__init__()
        self.data_loader = data_loader
        self.instruments_list = results["data_config"]["instruments_list"]
        self.agents_results = results["agents"]
        self.data_config = results["data_config"]
        self.backtest_config = results["backtest_config"]
        self.close_prices = self.get_close_prices()
        self.setupUi(self)
        self.resize(1300, 850)
        self.agents_names = agents_names
        self.plot_types = ["cumulative product", "cumulative sum", "returns"]
        self.legend_positions = ["right", "bottom", "no"]
        self.initialize_combo_boxes()
        self.initialize_plot()
        self.metrics_columns = [
            "agent", "final_return_sum", "final_return_prod",
            "apy", "sharpe", "mdd", "turnover", "volatility"
        ]
        self.tableWidgetMetrics.setSortingEnabled(True)
        self.metrics_columns_mapping = {name: i for i, name in enumerate(self.metrics_columns)}
        self.display_agents_metrics()
        self.plot()
        self.pushButtonPlot.clicked.connect(self.plot)
        self.pushButtonComputeMetrics.clicked.connect(self.display_agents_metrics)
        self.pushButtonPlotInstruments.clicked.connect(self.plot_instruments)
        self.pushButtonClearPlot.clicked.connect(self.clear_plot)

    def get_close_prices(self):
        self.begin_epoch = self.backtest_config["begin"]
        self.current_epoch = self.begin_epoch
        self.end_epoch = self.backtest_config["end"]
        self.step = self.backtest_config["step"]
        self.epochs = list(range(self.begin_epoch, self.end_epoch, self.step))
        self.price_label = self.backtest_config["price_label"]
        self.relevant_columns = [
            ">".join([
                self.backtest_config["exchange"],
                label,
                self.backtest_config["candles_res"],
                self.price_label
            ])
            for label in self.instruments_list
        ]
        self.relevant_columns_dict = {
            label: ">".join([
                self.backtest_config["exchange"],
                label,
                self.backtest_config["candles_res"],
                self.price_label
            ])
            for label in self.instruments_list
        }
        data = self.data_loader.get_data(self.epochs)
        close_prices = {
            label: [data[epoch][self.relevant_columns_dict[label]] \
                for epoch in self.epochs] \
                for label in self.instruments_list
        }
        return close_prices

    def initialize_plot(self):
        self.figure = plt.figure()
        self.canvas = FigureCanvas(self.figure)
        self.toolbar = NavigationToolbar(self.canvas, self.groupBoxPlot)
        self.toolbar.setMaximumHeight(25)
        self.verticalLayout_2.addWidget(self.canvas)
        self.verticalLayout_2.addWidget(self.toolbar)

    def instruments_toolbar(self):
        plot_instruments = ["all"]
        plot_instruments.extend(self.instruments_list)
        self.toolbuttonInstruments = QtWidgets.QToolButton(self)
        self.toolbuttonInstruments.setMinimumSize(QtCore.QSize(120, 22))
        self.toolbuttonInstruments.setText('Select Instruments')
        font = QtGui.QFont()
        font.setPointSize(13)
        self.toolbuttonInstruments.setFont(font)
        self.toolmenuInstruments = QtWidgets.QMenu(self)
        self.instruments_dict = {}
        for i, instrument in enumerate(plot_instruments):
            checkBox = QtWidgets.QCheckBox(self.toolmenuInstruments)
            checkBox.setChecked(False)
            checkBox.setText(instrument)
            checkableAction = QtWidgets.QWidgetAction(self.toolmenuInstruments)
            checkableAction.setDefaultWidget(checkBox)
            self.toolmenuInstruments.addAction(checkableAction)
            self.instruments_dict[instrument] = checkBox
        print("self.instruments_dict", self.instruments_dict)
        self.toolbuttonInstruments.setMenu(self.toolmenuInstruments)
        self.toolbuttonInstruments.setPopupMode(QtWidgets.QToolButton.InstantPopup)
        self.toolbuttonInstruments.setMenu(self.toolmenuInstruments)
        self.formLayout_2.setWidget(6, QtWidgets.QFormLayout.FieldRole, self.toolbuttonInstruments)

    def initialize_combo_boxes(self):
        self.instruments_toolbar()
        plot_agents = ["all"]
        plot_agents.extend(self.agents_names)
        self.toolbutton = QtWidgets.QToolButton(self)
        self.toolbutton.setMinimumSize(QtCore.QSize(120, 22))
        self.toolbutton.setText('Select Agents')
        font = QtGui.QFont()
        font.setPointSize(13)
        self.toolbutton.setFont(font)
        self.toolmenu = QtWidgets.QMenu(self)
        self.actions_dict = {}
        for i, agent in enumerate(plot_agents):
            checkBox = QtWidgets.QCheckBox(self.toolmenu)
            checkBox.setChecked(True)
            checkBox.setText(agent)
            checkableAction = QtWidgets.QWidgetAction(self.toolmenu)
            checkableAction.setDefaultWidget(checkBox)
            self.toolmenu.addAction(checkableAction)
            self.actions_dict[agent] = checkBox
        self.toolbutton.setMenu(self.toolmenu)
        self.toolbutton.setPopupMode(QtWidgets.QToolButton.InstantPopup)
        self.toolbutton.setMenu(self.toolmenu)
        self.formLayout_2.setWidget(0, QtWidgets.QFormLayout.FieldRole, self.toolbutton)
        self.comboBoxPlotType.addItems(self.plot_types)
        self.comboBoxPlotType.activated[str].connect(self.choose_plot_type)
        self.comboBoxLegend.addItems(self.legend_positions)
        self.comboBoxLegend.activated[str].connect(self.choose_legend_position)

    def get_number_of_trading_days(self):
        date_start = datetime.datetime.utcfromtimestamp(self.backtest_config["begin"])
        date_end = datetime.datetime.utcfromtimestamp(self.backtest_config["end"])
        return (date_end - date_start).days

    def compute_metrics(self, data):
        final_return_sum = np.cumsum(data - 1)[-1] + 1
        final_return_prod = np.cumprod(data)[-1]
        volatility = np.std(data - 1)
        sharpe = np.mean(data - 1) / volatility
        mdd = compute_max_drawdown(data)
        number_of_days = self.get_number_of_trading_days()
        if TRADING_DAYS_NUMBER.get(self.data_config["exchange"]) is not None:
            trading_days = TRADING_DAYS_NUMBER[self.data_config["exchange"]]
        else:
            trading_days = 253
        power = trading_days / number_of_days
        metrics =  {
            "final_return_sum": final_return_sum,
            "final_return_prod": final_return_prod,
            "apy": final_return_prod**power - 1,
            "sharpe": sharpe,
            "mdd": mdd,
            "volatility": volatility
        }
        return metrics

    def display_computed_metrics(self, metrics, value_format="%.3f"):
        numRows = self.tableWidgetMetrics.rowCount()
        self.tableWidgetMetrics.insertRow(numRows)
        for name, value in metrics.items():
            widget = QtWidgets.QLabel()
            if isinstance(value, str):
                str_format = "%s"
            else:
                str_format = value_format
            widget.setText(str_format % value)
            index = self.metrics_columns_mapping[name]
            self.tableWidgetMetrics.setCellWidget(numRows, index, widget)

    def display_agents_metrics(self):
        #self.metrics = {}
        self.tableWidgetMetrics.setRowCount(0)
        agents_to_plot = self.get_agents_to_plot()
        fee = self.doubleSpinBoxPlotFee.value()
        for agent in agents_to_plot:
            turnover = np.array(self.agents_results[agent]["turnover"])
            data = np.array(self.agents_results[agent]["returns_no_fee"]) - turnover * fee
            metrics = self.compute_metrics(data)
            #self.metrics[agent] = metrics
            metrics["agent"] = agent
            metrics["turnover"] = sum(turnover)
            self.display_computed_metrics(metrics)
        self.tableWidgetMetrics.sortItems(1)

    def choose_agents(self):
        agents_to_plot = self.comboBoxPlotAgents.currentText()
        print("agets to plot chosen: %s" % agents_to_plot)

    def choose_plot_type(self):
        plot_type = self.comboBoxPlotType.currentText()
        print("plot type chosen: %s" % plot_type)

    def choose_legend_position(self):
        plot_type = self.comboBoxLegend.currentText()
        print("legend position chosen: %s" % plot_type)

    def plot_legend(self, ax):
        if self.comboBoxLegend.currentText() == "bottom":
            box = ax.get_position()
            ax.set_position([box.x0, box.y0 + box.height * 0.1,
                             box.width, box.height * 0.9])
            ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
                    ncol=5, prop={"size": 5})
        elif self.comboBoxLegend.currentText() == "right":
            box = ax.get_position()
            ax.set_position([box.x0, box.y0, box.width * 0.9, box.height])
            ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), prop={"size": 5})
        elif self.comboBoxLegend.currentText() == "no":
            pass
        else:
            raise ValueError("Wrong comboBoxLegend value")

    def get_instruments_to_plot(self):
        if self.instruments_dict["all"].isChecked():
            return self.instruments_list
        else:
            return [
                instrument for instrument in self.instruments_list \
                    if self.instruments_dict[instrument].isChecked()
            ]

    def get_agents_to_plot(self):
        if self.actions_dict["all"].isChecked():
            return self.agents_names
        else:
            return [
                agent for agent in self.agents_names \
                    if self.actions_dict[agent].isChecked()
            ]

    def get_cumulative_returns(self, returns):
        if self.comboBoxPlotType.currentText() == "cumulative sum":
            data = np.cumsum(returns - 1) + 1
        elif self.comboBoxPlotType.currentText() == "cumulative product":
            data = np.cumprod(returns)
        elif self.comboBoxPlotType.currentText() == "returns":
            data = returns_with_fee
        else:
            raise ValueError("Wrong comboBoxPlotType value")
        return data

    def clear_plot(self):
        self.figure.clear()
        self.canvas.draw()
        self.clear_flag = True

    def finish_plot(self):
        self.figure.tight_layout()
        self.plot_legend(self.ax)
        self.ax.grid(color='lightgray', linestyle='dashed')
        for tick in self.ax.xaxis.get_major_ticks():
            tick.label.set_fontsize(4)
        for tick in self.ax.yaxis.get_major_ticks():
            tick.label.set_fontsize(4)
        self.canvas.draw()
        self.clear_flag = False

    def plot(self):
        self.figure.clear()
        self.ax = self.figure.add_subplot(111)
        plt.rcParams.update({'font.size': 10})
        agents_to_plot = self.get_agents_to_plot()
        fee = self.doubleSpinBoxPlotFee.value()
        for agent in agents_to_plot:
            turnover = np.array(self.agents_results[agent]["turnover"])
            returns_with_fee = np.array(
                self.agents_results[agent]["returns_no_fee"]) - turnover * fee
            data = self.get_cumulative_returns(returns_with_fee)
            line = self.ax.plot(data, label=agent, linewidth=0.9)
            if self.checkBoxPlotWithNoFee.isChecked():
                data = self.get_cumulative_returns(
                    np.array(self.agents_results[agent]["returns_no_fee"])
                )
                self.ax.plot(data, label=agent + ", no fee", 
                        linewidth=0.9, linestyle='--', c=line[0]._color)
        self.ax.set_ylabel('Returns', size=6)
        self.finish_plot()

    def plot_instruments(self):
        if not self.clear_flag:
            self.ax = self.figure.add_subplot(111)
            plt.rcParams.update({'font.size': 10})
        instruments_to_plot = self.get_instruments_to_plot()
        for instrument in instruments_to_plot:
            data = np.array(self.close_prices[instrument])
            if self.checkBoxNormalized.isChecked():
                data /= data[0]
            line = self.ax.plot(data, label=instrument, linewidth=0.9)
        self.finish_plot()
