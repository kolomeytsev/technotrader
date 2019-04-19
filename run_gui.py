import sys
sys.path.insert(0,'../')
from PyQt5 import QtCore, QtGui, QtWidgets
from technotrader.gui.gui import TechnoTraderMainWindow


def show_fatal_error(self, text):
    print("Error: %s" % text)
    print("Restarting program")
    msg = QtWidgets.QMessageBox()
    msg.setIcon(QtWidgets.QMessageBox.Critical)          
    msg.setWindowTitle("Error")
    msg.setText(text)
    okButton = msg.addButton('OK', QtWidgets.QMessageBox.AcceptRole)
    msg.exec()


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    techno_trader_ui = TechnoTraderMainWindow(MainWindow, app)
    MainWindow.show()
    res = app.exec_()
    sys.exit(res)
