import sys
import numpy as np
import pandas as pd
from PyQt5 import QtCore, QtGui, QtWidgets
import time
from backtest_results_window import Ui_FormPlot
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
import matplotlib.pyplot as plt


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
    def __init__(self, df, agents_names):
        super(BacktestResultsWindow, self).__init__()
        self.setupUi(self)
        self.resize(1300, 850)
        self.df = df
        self.agents_names = agents_names
        self.plot_types = ["cumulative sum", "cumulative product", "returns"]
        self.legend_positions = ["bottom", "right", "no"]
        self.initialize_combo_boxes()
        self.initialize_plot()
        self.compute_metrics()
        self.plot()
        self.pushButtonPlot.clicked.connect(self.plot)

    def initialize_plot(self):
        self.figure = plt.figure()
        self.canvas = FigureCanvas(self.figure)
        self.toolbar = NavigationToolbar(self.canvas, self.groupBoxPlot)
        self.toolbar.setMaximumHeight(25)
        self.verticalLayout_2.addWidget(self.canvas)
        self.verticalLayout_2.addWidget(self.toolbar)

    def initialize_combo_boxes(self):
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

    def compute_metrics(self):
        self.metrics = {}
        for agent in self.agents_names:
            data = self.df[agent + "_returns_no_fee"].values
            final_return = np.cumsum(data - 1)[-1] + 1
            volatility = np.std(data)
            sharpe = np.mean(data) / volatility
            number_of_days = 30
            trading_days = 365
            apy = final_return**(trading_days / number_of_days) - 1
            self.metrics[agent] = {
                "final_return": final_return,
                "sharpe": sharpe,
                "volatility": volatility,
                "apy": apy
            }
            numRows = self.tableWidgetMetrics.rowCount()
            self.tableWidgetMetrics.insertRow(numRows)
            agent_widget = QtWidgets.QLabel()
            agent_widget.setText(agent)
            final_return_widget = QtWidgets.QLabel()
            final_return_widget.setText(str(final_return))
            sharpe_widget = QtWidgets.QLabel()
            sharpe_widget.setText(str(sharpe))
            volatility_widget = QtWidgets.QLabel()
            volatility_widget.setText(str(volatility))
            apy_widget = QtWidgets.QLabel()
            apy_widget.setText(str(apy))
            self.tableWidgetMetrics.setCellWidget(numRows, 0, agent_widget)
            self.tableWidgetMetrics.setCellWidget(numRows, 1, final_return_widget)
            self.tableWidgetMetrics.setCellWidget(numRows, 2, sharpe_widget)
            self.tableWidgetMetrics.setCellWidget(numRows, 3, volatility_widget)
            self.tableWidgetMetrics.setCellWidget(numRows, 4, apy_widget)

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

    def get_agents_to_plot(self):
        if self.actions_dict["all"].isChecked():
            return self.agents_names
        else:
            return [
                agent for agent in self.agents_names \
                    if self.actions_dict[agent].isChecked()
            ]

    def plot(self):
        for agent, action in self.actions_dict.items():
            print(agent, action.isChecked())           
        #layout = QtWidgets.QVBoxLayout()
        #layout.addWidget(self.canvas)
        #layout.addWidget(self.toolbar)
        #self.groupBoxPlot.setLayout(layout)
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        plt.rcParams.update({'font.size': 10})
        agents_to_plot = self.get_agents_to_plot()
        for agent in agents_to_plot:
            if self.comboBoxPlotType.currentText() == "cumulative sum":
                data = np.cumsum(self.df[agent + "_returns_no_fee"].values - 1) + 1
            elif self.comboBoxPlotType.currentText() == "cumulative product":
                data = np.cumprod(self.df[agent + "_returns_no_fee"].values)
            elif self.comboBoxPlotType.currentText() == "returns":
                data = self.df[agent + "_returns_no_fee"].values
            else:
                raise ValueError("Wrong comboBoxPlotType value")
            ax.plot(data, label=agent, linewidth=0.9)
        self.figure.tight_layout()
        self.plot_legend(ax)
        ax.grid(color='lightgray', linestyle='dashed')
        ax.set_ylabel('Returns', size=6)
        for tick in ax.xaxis.get_major_ticks():
            tick.label.set_fontsize(4)
        for tick in ax.yaxis.get_major_ticks():
            tick.label.set_fontsize(4)
        self.canvas.draw()
