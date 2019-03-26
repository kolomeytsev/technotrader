import os
import sys
sys.path.insert(0,'../../')
from PyQt5 import QtCore, QtGui, QtWidgets
import datetime
import time
import pytz
import json
import qdarkstyle
from technotrader.run_backtest import *
from technotrader.gui.mainwindow import Ui_MainWindow
from technotrader.gui.backtest_results_window import Ui_FormPlot
from technotrader.gui.windows import WindowRunningBacktest, BacktestResultsWindow
from technotrader.gui.available_parameters import *


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


class TechnoTraderMainWindow(Ui_MainWindow):
    def __init__(self, MainWindow, app):
        self.app = app
        self.setupUi(MainWindow)
        MainWindow.resize(1200, 800)
        self.results_dict = {}
        self.initialize_combo_boxes()
        self.initialize_buttons()
        self.process_menu_actions()
        self.make_size_settings()
        self.backtest_analysis_columns = [
            "show", "run_id", "run_time", "name", "exchange",
            "resolution", "begin", "end", "return", "sharpe", "turnover"
        ]
        self.backtest_analysis_columns_mapping = {
            name: i for i, name in enumerate(self.backtest_analysis_columns)
        }

    def make_size_settings(self):
        self.tableWidgetAddedAgents.setColumnWidth(0, 100)
        self.tableWidgetAddedAgents.setColumnWidth(1, 200)
        self.tableWidgetAddedAgents.setColumnWidth(2, 200)

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
        agents_configs_path = "gui/default_agents_configs.json"
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
        self.pushButtonOpenResults.clicked.connect(self.file_open)

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

    def file_open(self):
        name = QtWidgets.QFileDialog.getExistingDirectory(None, "Open Directory")
        self.textEditOpenResults.setText(name)
        for file in os.listdir(name):
            if file.startswith("backtest_"):
                with open(name + '/' + file) as f:
                    results = json.load(f)
                    self.add_backtest_result(results)
                    self.results_dict[file.split('_')[1]] = results

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
        scroll_area_agent_class = QtWidgets.QScrollArea()
        scroll_area_agent_class.setWidget(agent_class_label)
        config_label_main = QtWidgets.QLabel()
        config_label_main.setText(config_main_str)
        scroll_area_main = QtWidgets.QScrollArea()
        scroll_area_main.setWidget(config_label_main)
        config_label_other = QtWidgets.QLabel()
        config_label_other.setText(config_other_str)
        scroll_area_other = QtWidgets.QScrollArea()
        scroll_area_other.setWidget(config_label_other)
        self.tableWidgetAddedAgents.setCellWidget(numRows, 0, scroll_area_agent_class)
        self.tableWidgetAddedAgents.setCellWidget(numRows, 1, scroll_area_main)
        self.tableWidgetAddedAgents.setCellWidget(numRows, 2, scroll_area_other)
        #header = self.tableWidgetAddedAgents.horizontalHeader()
        #header.setSectionResizeMode(0, QtWidgets.QHeaderView.ResizeToContents)
        #header.setSectionResizeMode(1, QtWidgets.QHeaderView.ResizeToContents)
        #header.setSectionResizeMode(2, QtWidgets.QHeaderView.ResizeToContents)

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
            agent_class = self.tableWidgetAddedAgents.cellWidget(row, 0).widget().text()
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
        for agent_name, test_pc_vector_no_fee, test_pc_vector, test_turnover_vector \
                in backtesters_results:
            results[agent_name + "_returns_no_fee"] = test_pc_vector_no_fee
            results[agent_name + "_returns_with_fee"] = test_pc_vector
            results[agent_name + "_turnover"] = test_turnover_vector
        df = pd.DataFrame.from_dict(results)
        return df

    def get_backtest_id(self):
        results_path = self.lineEditDumpPath.text()
        if len(results_path) == 0:
            return 0
        list_dir = set(os.listdir(results_path))
        if "id_generator.txt" in list_dir:
            with open(results_path + "/id_generator.txt") as f:
                backtest_id = int(f.readline()) + 1
            with open(results_path + "/id_generator.txt", "w") as f:
                f.write("%d" % backtest_id)
        else:
            with open(results_path + "/id_generator.txt", "w") as f:
                f.write("0")
                backtest_id = 0
        return backtest_id

    def run_backtest(self):
        parse_results = self.parse_configs()
        if parse_results is not None:
             data_config, backtest_config = parse_results
        else:
            return
        agents_configs = self.parse_agents(data_config)
        print("data_config\n", data_config)
        print("backtest_config\n", backtest_config)
        print("agents_configs:\n", agents_configs)
        path = self.lineEditDumpPath.text()
        backtest_id = self.get_backtest_id()
        backtest_config["id"] = backtest_id
        backtest_config["time"] = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())
        if len(path) == 0:
            path = None
        else:
            path += ("backtest_%d.json" % backtest_id)
        print("path:", path)
        self.data_loader = DataLoader(data_config)
        results = run_multi_backtest(self.data_loader, data_config, 
                                    agents_configs, backtest_config,
                                    path=path, parallel=self.checkBoxParallel.isChecked())
        #results_df = self.convert_results_to_dataframe(results)
        #self.window_run_backtest = WindowRunningBacktest()
        #self.window_run_backtest.close()
        self.backtest_completed()
        #results_df = pd.read_csv("gui/multi_poloniex_hour_1.csv")
        #agents_names = [x[:-len("_returns_no_fee")] for x in results_df.columns \
        #                    if x.endswith("_returns_no_fee")]
        agents_names = [x[0] for x in agents_configs]
        self.add_backtest_result(results)
        self.show_backtest_results(agents_names, results)

    def show_error(self, text):
        print("error: %s" % text)
        msg = QtWidgets.QMessageBox()
        msg.setIcon(QtWidgets.QMessageBox.Critical)          
        msg.setWindowTitle("Error")
        msg.setText(text)
        okButton = msg.addButton('OK', QtWidgets.QMessageBox.AcceptRole)
        msg.exec()

    def show_results(self):
        print("Showing results")

    def add_backtest_result(self, results):
        if len(results["agents"].keys()) == 1:
            res_agent = list(results["agents"].items())[0]
            name = [0]
            returns = np.array(res_agent["returns_no_fee"])
            final_return = np.cumprod()[-1]
            sharpe = np.mean(returns - 1) / np.std(returns - 1)
            turnover = sum(res_agent["turnover"])
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
            "turnover": turnover
        }
        numRows = self.tableWidgetBacktestsResults.rowCount()
        self.tableWidgetBacktestsResults.insertRow(numRows)
        show_widget = QtWidgets.QPushButton()
        show_widget.setText("Show")
        show_widget.clicked.connect(self.show_results)
        self.tableWidgetBacktestsResults.setCellWidget(numRows, 0, show_widget)
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
            index = self.backtest_analysis_columns_mapping[name]
            self.tableWidgetBacktestsResults.setCellWidget(numRows, index, widget)

    def show_backtest_results(self, agents_names, results):
        self.window = BacktestResultsWindow(self.data_loader, agents_names, results)
        self.window.show()
