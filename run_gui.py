import sys
sys.path.insert(0,'../')
from PyQt5 import QtCore, QtGui, QtWidgets
from technotrader.gui.gui import TechnoTraderMainWindow


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    techno_trader_ui = TechnoTraderMainWindow(MainWindow, app)
    MainWindow.show()
    sys.exit(app.exec_())
