import sys
from PyQt5 import QtCore, QtGui, QtWidgets
from mainwindow import Ui_MainWindow
from windows import WindowRunningBacktest
from available_parameters import *
import time
import json
import qdarkstyle


class TechnoTraderMainWindow(Ui_MainWindow):
    def __init__(self, MainWindow, app):
        self.app = app
        self.setupUi(MainWindow)
        self.initialize_agents()
        self.initialize_combo_boxes()
        self.process_menu_actions()
        self.pushButtonRunBacktest.clicked.connect(self.run_backtest)
        self.pushButtonAddAgent.clicked.connect(self.add_agent)

    def initialize_combo_boxes(self):
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
        #self.comboBoxChooseAgent.currentTextChanged.connect(lambda text: self.choose_agent(text))
        self.comboBoxChooseAgent.activated[str].connect(self.choose_agent)

    def choose_agent(self):
        #self.tableWidgetMainParameters.clear()
        self.tableWidgetMainParameters.setRowCount(0)
        agent_class = self.comboBoxChooseAgent.currentText()
        if agent_class == '':
            return
        print("agent chosen: %s" % agent_class)
        config = self.agents_configs[agent_class]
        print("config:", config)
        for param_name, param_value in config.items():
            numRows = self.tableWidgetMainParameters.rowCount()
            print("param_value", param_value)
            self.tableWidgetMainParameters.insertRow(numRows)

            name_widget = QtWidgets.QLabel()
            name_widget.setText(param_name)
            value_widget = QtWidgets.QLineEdit()
            value_widget.setText(str(param_value))
            #self.tableWidgetMainParameters.setCellWidget(numRows, 0, QtWidgets.QTableWidgetItem(param_name))
            #self.tableWidgetMainParameters.setCellWidget(numRows, 1, QtWidgets.QTableWidgetItem(param_value))
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
            "Exit!", "Are you sure?",
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No
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

    def add_agent(self):
        config = {
            self.comboBoxPriceLabel.currentText()
        }
        config = {
            "price_label": self.comboBoxPriceLabel.currentText(),
            "use_risk_free": self.checkBoxUseRiskFree.isChecked(),
            "short_flag": self.checkBoxUseShorts.isChecked(),
            "neutralize_flag": self.checkUseNeutralization.isChecked(),
            "projection_method": self.comboBoxProjectMethod.currentText(),
            "top_amount": self.spinBoxTopAmount.value()
        }
        for row in range(self.tableWidgetMainParameters.rowCount()):
            param_name = self.tableWidgetMainParameters.cellWidget(row, 0).text()
            param_value = self.tableWidgetMainParameters.cellWidget(row, 1).text()
            param_value = param_value.replace("'", '"').replace("None", "null")
            param_value = param_value.replace("True", "true").replace("False", "false")
            config[param_name] = json.loads(param_value)
        print("adding agent with config:", config)

    def run_backtest(self):
        self.window_run_backtest = WindowRunningBacktest()
        self.window_run_backtest.close()
        self.backtest_completed()
        self.plot_backtest_results()

    def plot_backtest_results(self):
        pass


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    #app.setStyleSheet(qdarkstyle.load_stylesheet_pyqt5())
    MainWindow = QtWidgets.QMainWindow()
    techno_trader_ui = TechnoTraderMainWindow(MainWindow, app)
    MainWindow.show()
    sys.exit(app.exec_())
