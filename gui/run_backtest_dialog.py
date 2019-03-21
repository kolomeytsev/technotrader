# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'run_backtest_dialog.ui'
#
# Created by: PyQt5 UI code generator 5.9.2
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_Dialog(object):
    def setupUi(self, Dialog):
        Dialog.setObjectName("Dialog")
        Dialog.resize(327, 103)
        self.formLayout = QtWidgets.QFormLayout(Dialog)
        self.formLayout.setObjectName("formLayout")
        self.labelBacktesting = QtWidgets.QLabel(Dialog)
        self.labelBacktesting.setMaximumSize(QtCore.QSize(16777215, 20))
        font = QtGui.QFont()
        font.setPointSize(15)
        self.labelBacktesting.setFont(font)
        self.labelBacktesting.setObjectName("labelBacktesting")
        self.formLayout.setWidget(0, QtWidgets.QFormLayout.LabelRole, self.labelBacktesting)
        self.pushButtonCancel = QtWidgets.QPushButton(Dialog)
        self.pushButtonCancel.setMaximumSize(QtCore.QSize(100, 16777215))
        self.pushButtonCancel.setObjectName("pushButtonCancel")
        self.formLayout.setWidget(2, QtWidgets.QFormLayout.LabelRole, self.pushButtonCancel)
        self.progressBarBacktesting = QtWidgets.QProgressBar(Dialog)
        self.progressBarBacktesting.setProperty("value", 24)
        self.progressBarBacktesting.setObjectName("progressBarBacktesting")
        self.formLayout.setWidget(1, QtWidgets.QFormLayout.SpanningRole, self.progressBarBacktesting)

        self.retranslateUi(Dialog)
        QtCore.QMetaObject.connectSlotsByName(Dialog)

    def retranslateUi(self, Dialog):
        _translate = QtCore.QCoreApplication.translate
        Dialog.setWindowTitle(_translate("Dialog", "Dialog"))
        self.labelBacktesting.setText(_translate("Dialog", "Backtesting in Progress:"))
        self.pushButtonCancel.setText(_translate("Dialog", "Cancel"))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    Dialog = QtWidgets.QDialog()
    ui = Ui_Dialog()
    ui.setupUi(Dialog)
    Dialog.show()
    sys.exit(app.exec_())

