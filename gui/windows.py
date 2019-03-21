import sys
from PyQt5 import QtCore, QtGui, QtWidgets
import time


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
