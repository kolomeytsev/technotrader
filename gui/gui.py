import sys
sys.path.insert(0,'../../')
from PyQt5 import QtCore, QtGui, QtWidgets
from mainwindow import Ui_MainWindow
from backtest_results_window import Ui_FormPlot
from windows import WindowRunningBacktest, BacktestResultsWindow
from available_parameters import *
import datetime
import time
import pytz
import json
import qdarkstyle
from technotrader.run_backtest import *


def get_time_as_name_string(start, end):
    name_string = "_" + start.replace('/', '').replace(' ', '_') + \
                  "_" + end.replace('/', '').replace(' ', '')
    return name_string


def convert_time(time_str):
    dt = datetime.datetime.strptime(
            time_str, "%Y-%m-%d %H:%M:%S").replace(tzinfo=pytz.utc)
    return int(dt.timestamp())


class TechnoTraderMainWindow(Ui_MainWindow):
    def __init__(self, MainWindow, app):
        self.app = app
        self.setupUi(MainWindow)
        self.initialize_combo_boxes()
        self.initialize_buttons()
        self.process_menu_actions()

    def initialize_combo_boxes(self):
        self.initialize_agents()
        self.comboBoxPriceLabel.addItems(PRICE_LABELS)
        self.comboBoxPriceLabel.activated[str].connect(self.choose_price_label)
        self.comboBoxProjectMethod.addItems(PROJECTIONS_METHODS)
        self.comboBoxProjectMethod.activated[str].connect(self.choose_project_method)
        self.comboBoxDataset.addItems(EXCHANGES_DATASETS)
        self.comboBoxDataset.activated[str].connect(self.choose_dataset)
        candles_resolutions = ['']
        candles_resolutions.extend(list(RESOLUTIONS.keys()))
        self.comboBoxCandlesResolution.addItems(candles_resolutions)
        self.comboBoxCandlesResolution.activated[str].connect(self.choose_candles_resolution)

    def initialize_agents(self):
        agents_configs_path = "default_agents_configs.json"
        with open(agents_configs_path) as f:
            self.agents_configs = json.load(f)
        all_items = ['']
        all_items.extend(list(self.agents_configs.keys()))
        self.comboBoxChooseAgent.addItems(all_items)
        self.comboBoxChooseAgent.activated[str].connect(self.choose_agent)

    def initialize_buttons(self):
        self.pushButtonRunBacktest.clicked.connect(self.run_backtest)
        self.pushButtonAddAgent.clicked.connect(self.add_agent)
        self.pushButtonAddedEdit.clicked.connect(self.edit_agents)
        self.pushButtonAddedDelete.clicked.connect(self.delete_agents)

    def choose_agent(self):
        self.tableWidgetMainParameters.setRowCount(0)
        agent_class = self.comboBoxChooseAgent.currentText()
        if agent_class == '':
            return
        print("agent chosen: %s" % agent_class)
        config = self.agents_configs[agent_class]
        print("config:", config)
        self.lineEditAgentsName.setText(agent_class)
        for param_name, param_value in config.items():
            numRows = self.tableWidgetMainParameters.rowCount()
            print("param_value", param_value)
            self.tableWidgetMainParameters.insertRow(numRows)
            name_widget = QtWidgets.QLabel()
            name_widget.setText(param_name)
            value_widget = QtWidgets.QLineEdit()
            value_widget.setText(str(param_value))
            self.tableWidgetMainParameters.setCellWidget(numRows, 0, name_widget)
            self.tableWidgetMainParameters.setCellWidget(numRows, 1, value_widget)

    def choose_price_label(self):
        price_label = self.comboBoxPriceLabel.currentText()
        print("price_label chosen: %s" % price_label)

    def choose_project_method(self):
        project_method = self.comboBoxProjectMethod.currentText()
        print("project_method chosen: %s" % project_method)

    def choose_dataset(self):
        exchange_dataset = self.comboBoxDataset.currentText()
        print("exchange_dataset chosen: %s" % exchange_dataset)

    def choose_candles_resolution(self):
        self.spinBoxStep.clear()
        candles_resolution = self.comboBoxCandlesResolution.currentText()
        print("candles_resolution chosen: %s" % candles_resolution)
        self.spinBoxStep.setValue(RESOLUTIONS[candles_resolution])

    def process_menu_actions(self):
        self.actionQuit.setShortcut("Ctrl+Q")
        self.actionQuit.triggered.connect(self.close_application)
        self.actionDefault.triggered.connect(self.style_choice_default)
        self.actionFusion.triggered.connect(self.style_choice_fusion)
        self.actionWindows.triggered.connect(self.style_choice_windows)
        self.actionDark.triggered.connect(self.style_choice_dark)

    def style_choice_default(self):
        self.app.setStyleSheet('')
        QtWidgets.QApplication.setStyle(QtWidgets.QStyleFactory.create("macintosh"))

    def style_choice_fusion(self):
        self.app.setStyleSheet('')
        QtWidgets.QApplication.setStyle(QtWidgets.QStyleFactory.create("Fusion"))

    def style_choice_windows(self):
        self.app.setStyleSheet('')
        QtWidgets.QApplication.setStyle(QtWidgets.QStyleFactory.create("Windows"))

    def style_choice_dark(self):
        self.app.setStyleSheet(qdarkstyle.load_stylesheet_pyqt5())

    def close_application(self):
        choice = QtWidgets.QMessageBox.question(
            None,
            "Quit!", "Are you sure you want to quit?",
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.Cancel
        )
        if choice == QtWidgets.QMessageBox.Yes:
            sys.exit()

    def backtest_completed(self):
        msg = QtWidgets.QMessageBox()
        msg.setIcon(QtWidgets.QMessageBox.Information)          
        msg.setWindowTitle("Completed")
        msg.setText("Backtesting Completed")
        okButton = msg.addButton('OK', QtWidgets.QMessageBox.AcceptRole)
        msg.exec()

    def get_agents_config(self, agent_class):
        config_other = {
            "agent_class": agent_class,
            "agent_name": self.lineEditAgentsName.text(),
            "price_label": self.comboBoxPriceLabel.currentText(),
            "use_risk_free": self.checkBoxUseRiskFree.isChecked(),
            "short_flag": self.checkBoxUseShorts.isChecked(),
            "neutralize_flag": self.checkUseNeutralization.isChecked(),
            "projection_method": self.comboBoxProjectMethod.currentText(),
            "top_amount": self.spinBoxTopAmount.value()
        }
        config_main = {}
        for row in range(self.tableWidgetMainParameters.rowCount()):
            param_name = self.tableWidgetMainParameters.cellWidget(row, 0).text()
            param_value = self.tableWidgetMainParameters.cellWidget(row, 1).text()
            print(param_value)
            param_value = param_value.replace("'", '"').replace("None", "null")
            param_value = param_value.replace("True", "true").replace("False", "false")
            config_main[param_name] = json.loads(param_value)
        return config_main, config_other

    def edit_row(self, row_index):
        pass

    def edit_agents(self):
        rows = self.tableWidgetAddedAgents.selectionModel().selectedRows()
        indices_to_edit = sorted([r.row() for r in rows])
        print("Editing agents with indices:", indices_to_edit)
        for index in indices_to_edit:
            self.edit_row(row_index)

    def delete_agents(self):
        rows = self.tableWidgetAddedAgents.selectionModel().selectedRows()
        num_deleted = 0
        indices_to_delete = sorted([r.row() for r in rows])
        print("Deleting agents with indices:", indices_to_delete)
        for index in indices_to_delete:
            self.tableWidgetAddedAgents.removeRow(index - num_deleted)
            num_deleted += 1

    def insert_agent(self, agent_class, config_main_str, config_other_str):
        numRows = self.tableWidgetAddedAgents.rowCount()
        self.tableWidgetAddedAgents.insertRow(numRows)
        agent_class_label = QtWidgets.QLabel()
        agent_class_label.setText(agent_class)
        config_label_main = QtWidgets.QLabel()
        config_label_main.setText(config_main_str)
        scroll_area_main = QtWidgets.QScrollArea()
        scroll_area_main.setWidget(config_label_main)
        config_label_other = QtWidgets.QLabel()
        config_label_other.setText(config_other_str)
        scroll_area_other = QtWidgets.QScrollArea()
        scroll_area_other.setWidget(config_label_other)
        self.tableWidgetAddedAgents.setCellWidget(numRows, 0, agent_class_label)
        self.tableWidgetAddedAgents.setCellWidget(numRows, 1, scroll_area_main)
        self.tableWidgetAddedAgents.setCellWidget(numRows, 2, scroll_area_other)
        header = self.tableWidgetAddedAgents.horizontalHeader()
        header.setSectionResizeMode(0, QtWidgets.QHeaderView.ResizeToContents)
        header.setSectionResizeMode(1, QtWidgets.QHeaderView.ResizeToContents)
        header.setSectionResizeMode(2, QtWidgets.QHeaderView.ResizeToContents)

    def add_agent(self):
        agent_class = self.comboBoxChooseAgent.currentText()
        if agent_class == '':
            self.show_error("Pick an agent!")
            return
        config_main, config_other = self.get_agents_config(agent_class)
        config_main_str = json.dumps(config_main)
        config_other_str = json.dumps(config_other)
        print("adding agent with config:", config_main_str, config_other_str)
        self.insert_agent(config_other["agent_class"], config_main_str, config_other_str)

    def get_instruments_list(self, instruments):
        instruments = instruments.replace("'", '').replace('"', '')
        instruments_list = [x.strip() for x in instruments.split(',')]
        return [x for x in instruments_list if len(x)]

    def parse_configs(self):
        data_begin = self.dateTimeEditDataStart.dateTime()
        data_begin = data_begin.toString(self.dateTimeEditDataStart.displayFormat())
        begin = self.dateTimeEditBacktestStart.dateTime()
        begin = begin.toString(self.dateTimeEditBacktestStart.displayFormat())
        end = self.dateTimeEditBacktestEnd.dateTime()
        end = end.toString(self.dateTimeEditBacktestEnd.displayFormat())
        exchange = self.comboBoxDataset.currentText()
        print(data_begin)
        if begin >= end:
            self.show_error("Backtest start must be earlier than end!")
            return
        if data_begin >= end:
            self.show_error("Data start must be earlier than backtest start!")
            return
        candles_res = self.comboBoxCandlesResolution.currentText()
        step = self.spinBoxStep.value()
        if candles_res == '':
            self.show_error("Pick Candles Resolution!")
            return
        if step == 0:
            self.show_error("Pick Step!")
            return
        instruments = self.textEditInstruments.toPlainText()
        if instruments == '':
            self.show_error("Pick Instruments!")
            return
        instruments_list = self.get_instruments_list(instruments)
        data_name = "data"
        data_name += get_time_as_name_string(begin, end)
        data_name += '_' + str(step) + '_' + exchange
        data_config = {
            "data_name": data_name,
            "begin":  convert_time(data_begin),
            "end":  convert_time(end),
            "step": step,
            "candles_res": candles_res,
            "candles_res_sec": RESOLUTIONS[candles_res],
            "exchange": exchange,
            "instruments_list": instruments_list
        }
        backtest_config = {
            "begin":  convert_time(begin),
            "end":  convert_time(end),
            "step": step,
            "fee": self.doubleSpinBoxFee.value(),
            "exchange": exchange,
            "candles_res": candles_res,
            "price_label": "close",
            "log_frequency": 1
        }
        return data_config, backtest_config

    def parse_agents(self, data_config):
        agents_configs = []
        for row in range(self.tableWidgetAddedAgents.rowCount()):
            agent_config = {
                "instruments_list": data_config["instruments_list"],
                "exchange": data_config["exchange"],
                "candles_res": data_config["candles_res"],
                "step": data_config["step"]
            }
            agent_class = self.tableWidgetAddedAgents.cellWidget(row, 0).text()
            agent_main_params_str = self.tableWidgetAddedAgents.cellWidget(row, 1).widget().text()
            agent_other_params_str = self.tableWidgetAddedAgents.cellWidget(row, 2).widget().text()
            agent_main_params = json.loads(agent_main_params_str)
            agent_other_params = json.loads(agent_other_params_str)
            agent_config.update(agent_main_params.items())
            agent_config.update(agent_other_params.items())
            agents_configs.append((agent_class, agent_config))
        return agents_configs

    def convert_results_to_dataframe(self, backtesters_results):
        results = {}
        for agent_name, test_pc_vector_no_fee, test_pc_vector in backtesters_results:
            results[agent_name + "_returns_no_fee"] = test_pc_vector_no_fee
            results[agent_name + "_returns_with_fee"] = test_pc_vector
        df = pd.DataFrame.from_dict(results)
        return df

    def run_backtest(self):
        """parse_results = self.parse_configs()
        if parse_results is not None:
             data_config, backtest_config = parse_results
        else:
            return
        agents_configs = self.parse_agents(data_config)
        print("data_config\n", data_config)
        print("backtest_config\n", backtest_config)
        print("agents_configs:\n", agents_configs)
        path = self.lineEditDumpPath.text()
        if len(path) == 0:
            path = None
        results = run_multi_backtest(data_config, agents_configs, backtest_config,
                                    path=path, parallel=self.checkBoxParallel.isChecked())
        results_df = self.convert_results_to_dataframe(results)"""
        #self.window_run_backtest = WindowRunningBacktest()
        #self.window_run_backtest.close()
        self.backtest_completed()
        results_df = pd.read_csv("test_poloniex_september.csv")
        agents_names = list(set(x.split('_')[0] for x in results_df.columns))
        agents_names.remove("cfr")
        agents_names.extend(["cfr_ogd", "cfr_ons"])
        #agents_names = [x[0] for x in agents_configs]
        self.plot_backtest_results(results_df, agents_names)

    def show_error(self, text):
        print("error: %s" % text)
        msg = QtWidgets.QMessageBox()
        msg.setIcon(QtWidgets.QMessageBox.Critical)          
        msg.setWindowTitle("Error")
        msg.setText(text)
        okButton = msg.addButton('OK', QtWidgets.QMessageBox.AcceptRole)
        msg.exec()

    def plot_backtest_results(self, df, agents_names):
        self.window = BacktestResultsWindow(df, agents_names)
        self.window.show()


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    techno_trader_ui = TechnoTraderMainWindow(MainWindow, app)
    MainWindow.show()
    sys.exit(app.exec_())
