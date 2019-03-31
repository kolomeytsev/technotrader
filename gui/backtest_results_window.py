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
from technotrader.gui.ui_backtest_results_window import Ui_FormPlot
from technotrader.gui.utils_gui import *
from technotrader.utils.metrics import *
from technotrader.trading.constants import *


class BacktestResultsWindow(QtWidgets.QWidget, Ui_FormPlot):
    def __init__(self, data_loader, results):
        super(BacktestResultsWindow, self).__init__()
        self.setupUi(self)
        self.add_table_columns()
        self.results = results
        self.process_results(results)
        self.data_loader = data_loader
        if self.data_loader is not None:
            self.close_prices = self.get_close_prices()
        else:
            self.close_prices = None
        self.resize(1300, 850)
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
        self.pushButtonPlotUbah.clicked.connect(self.plot_ubah)
        self.pushButtonPlotUcrp.clicked.connect(self.plot_ucrp)

    def add_table_columns(self):
        self.backtest_info_columns = [
            "run_id", "run_time", "name", "exchange", 
            "resolution", "begin", "end", 
            "return", "sharpe", "turnover", "instruments"
        ]
        self.backtest_info_columns_mapping = {name: i for i, name in enumerate(self.backtest_info_columns)}
        self.tableWidgetGeneralInfo.setColumnWidth(self.backtest_info_columns_mapping["instruments"], 500)

        self.agent_info_columns = [
            "run_id", "agent_name", "parameters"
        ]
        self.agent_info_columns_mapping = {name: i for i, name in enumerate(self.agent_info_columns)}
        self.tableWidgetAgentsInfo.setColumnWidth(self.agent_info_columns_mapping["parameters"], 1000)

    def get_all_instrument(self, results):
        all_instruments_list = set()
        for res in results:
            all_instruments_list.union(res["data_config"]["instruments_list"])
        return list(all_instruments_list)

    def gather_agent_results(self, results):
        agents_results = {}
        for res in results:
            for agent_name, val in res["agents"].items():
                str_addition = '(' + str(res["backtest_config"]["id"]) + ')'
                new_agent_name = agent_name + str_addition
                agents_results[new_agent_name] = val
        print(agents_results.keys())
        return agents_results

    def process_results(self, results):
        if not isinstance(results, list):
            if results["data_config"].get("type") is not None:
                self.data_type = results["data_config"]["type"]
            else:
                self.data_type = "exchange"
            self.exchange = results["backtest_config"]["exchange"]
            self.instruments_list = results["data_config"]["instruments_list"]
            self.agents_results = results["agents"]
            self.data_config = results["data_config"]
            self.backtest_config = results["backtest_config"]
            self.add_all_backtest_info([results])
        else:
            if results[0]["data_config"].get("type") is not None:
                self.data_type = results[0]["data_config"]["type"]
            else:
                self.data_type = "exchange"
            self.instruments_list = self.get_all_instrument(results)
            self.begin_epoch = min([r["backtest_config"]["begin"] for r in results])
            self.end_epoch = max([r["backtest_config"]["end"] for r in results])
            self.agents_results = self.gather_agent_results(results)
            self.exchange = results[0]["data_config"]["exchange"]
            self.add_all_backtest_info(results)
        self.agents_names = list(self.agents_results.keys())

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
        self.clear_plot()

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
        self.formLayout_5.setWidget(6, QtWidgets.QFormLayout.FieldRole, self.toolbuttonInstruments)

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
        self.formLayout_5.setWidget(0, QtWidgets.QFormLayout.FieldRole, self.toolbutton)
        self.comboBoxPlotType.addItems(self.plot_types)
        self.comboBoxPlotType.activated[str].connect(self.choose_plot_type)
        self.comboBoxLegend.addItems(self.legend_positions)
        self.comboBoxLegend.activated[str].connect(self.choose_legend_position)

    def get_number_of_trading_days(self):
        if self.data_type == "exchange":
            date_start = datetime.datetime.utcfromtimestamp(self.begin_epoch)
            date_end = datetime.datetime.utcfromtimestamp(self.end_epoch)
            return (date_end - date_start).days
        else:
            return self.end_epoch - self.begin_epoch

    def compute_metrics(self, data):
        final_return_sum = np.cumsum(data - 1)[-1] + 1
        final_return_prod = np.cumprod(data)[-1]
        volatility = np.std(data - 1)
        sharpe = np.mean(data - 1) / volatility
        mdd = compute_max_drawdown(data)
        number_of_days = self.get_number_of_trading_days()
        if TRADING_DAYS_NUMBER.get(self.exchange) is not None:
            trading_days = TRADING_DAYS_NUMBER[self.exchange]
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
        self.ax = self.figure.add_subplot(111)
        plt.rcParams.update({'font.size': 10})
        self.finish_plot()
        #self.clear_flag = True

    def finish_plot(self):
        self.figure.tight_layout()
        self.plot_legend(self.ax)
        self.ax.grid(color='lightgray', linestyle='dashed')
        for tick in self.ax.xaxis.get_major_ticks():
            tick.label.set_fontsize(4)
        for tick in self.ax.yaxis.get_major_ticks():
            tick.label.set_fontsize(4)
        self.canvas.draw()
        #self.clear_flag = False

    def plot(self):
        #self.figure.clear()
        #self.ax = self.figure.add_subplot(111)
        #plt.rcParams.update({'font.size': 10})
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
        if self.close_prices is None:
            return
        instruments_to_plot = self.get_instruments_to_plot()
        for instrument in instruments_to_plot:
            data = np.array(self.close_prices[instrument])
            if self.checkBoxNormalized.isChecked():
                data /= data[0]
            line = self.ax.plot(data, label=instrument, linewidth=0.9)
        self.finish_plot()

    def plot_ubah(self):
        if self.close_prices is None:
            return
        instruments_to_plot = self.get_instruments_to_plot()
        if len(instruments_to_plot) == 0:
            return
        data = []
        for instrument in instruments_to_plot:
            data.append(self.close_prices[instrument]) 
        data = np.array(data).T
        data_normed = data / data[0]
        ubah = data_normed.mean(1)
        line = self.ax.plot(ubah, label="UBAH", linewidth=0.9)
        self.finish_plot()
    
    def plot_ucrp(self):
        if self.close_prices is None:
            return
        instruments_to_plot = self.get_instruments_to_plot()
        if len(instruments_to_plot) == 0:
            return
        data = []
        for instrument in instruments_to_plot:
            data.append(self.close_prices[instrument])
        data = np.array(data).T
        data_normed = data[1:] / data[:-1]
        ucrp = np.cumprod(data_normed.mean(1))
        line = self.ax.plot(ucrp, label="UCRP", linewidth=0.9)
        self.finish_plot()

    def add_all_backtest_info(self, results):
        for res in results:
            self.add_backtest_info(res)
            self.add_agents(res)

    def add_agent(self, run_id, agent, agents_result):
        numRows = self.tableWidgetAgentsInfo.rowCount()
        self.tableWidgetAgentsInfo.insertRow(numRows)
        widget = QtWidgets.QLabel()
        widget.setText(str(run_id))
        self.tableWidgetAgentsInfo.setCellWidget(numRows, 0, widget)
        widget = QtWidgets.QLabel()
        widget.setText(str(agent))
        self.tableWidgetAgentsInfo.setCellWidget(numRows, 1, widget)
        widget = QtWidgets.QLabel()
        widget.setText(str(agents_result["config"]))
        scroll_area = QtWidgets.QScrollArea()
        scroll_area.setWidget(widget)
        self.tableWidgetAgentsInfo.setCellWidget(numRows, 2, scroll_area)

    def add_agents(self, res):
        for agent, agents_result in res["agents"].items():
            self.add_agent(res["backtest_config"]["id"], agent, agents_result)

    def add_backtest_info(self, results):
        if len(results["agents"].keys()) == 1:
            res_agent = list(results["agents"].items())[0]
            name = res_agent[0]
            agent_dict = res_agent[1]
            returns = np.array(agent_dict["returns_no_fee"])
            final_return = np.cumprod(returns)[-1]
            sharpe = np.mean(returns - 1) / np.std(returns - 1)
            turnover = sum(agent_dict["turnover"])
        else:
            name = "multi"
            final_return = ''
            sharpe = ''
            turnover = ''
        values = {
            "run_id": results["backtest_config"]["id"],
            "run_time": results["backtest_config"]["time"],
            "name": name,
            "exchange": results["backtest_config"]["exchange"],
            "resolution": results["backtest_config"]["candles_res"],
            "begin": convert_time_to_str(results["backtest_config"]["begin"]),
            "end": convert_time_to_str(results["backtest_config"]["end"]),
            "return": final_return,
            "sharpe": sharpe,
            "turnover": turnover,
            "instruments": str(results["data_config"]["instruments_list"])
        }
        numRows = self.tableWidgetGeneralInfo.rowCount()
        self.tableWidgetGeneralInfo.insertRow(numRows)
        for name, value in values.items():
            widget = QtWidgets.QLabel()
            if isinstance(value, str):
                str_format = "%s"
            elif isinstance(value, float):
                str_format = "%.3f"
            elif isinstance(value, int):
                str_format = "%d"
            else:
                ValueError("Wrong format")
            widget.setText(str_format % value)
            index = self.backtest_info_columns_mapping[name]
            self.tableWidgetGeneralInfo.setCellWidget(numRows, index, widget)
