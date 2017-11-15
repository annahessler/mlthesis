# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'basicgui.ui'
#
# Created by: PyQt4 UI code generator 4.11.4
#
# WARNING! All changes made in this file will be lost!

from PyQt4 import QtCore, QtGui

try:
    _fromUtf8 = QtCore.QString.fromUtf8
except AttributeError:
    def _fromUtf8(s):
        return s

try:
    _encoding = QtGui.QApplication.UnicodeUTF8
    def _translate(context, text, disambig):
        return QtGui.QApplication.translate(context, text, disambig, _encoding)
except AttributeError:
    def _translate(context, text, disambig):
        return QtGui.QApplication.translate(context, text, disambig)

class Ui_GUI(object):
    def setupUi(self, GUI):
        GUI.setObjectName(_fromUtf8("GUI"))
        GUI.setEnabled(True)
        GUI.resize(703, 550)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Preferred, QtGui.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(GUI.sizePolicy().hasHeightForWidth())
        GUI.setSizePolicy(sizePolicy)
        self.centralwidget = QtGui.QWidget(GUI)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Preferred, QtGui.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.centralwidget.sizePolicy().hasHeightForWidth())
        self.centralwidget.setSizePolicy(sizePolicy)
        self.centralwidget.setObjectName(_fromUtf8("centralwidget"))
        self.horizontalLayout = QtGui.QHBoxLayout(self.centralwidget)
        self.horizontalLayout.setObjectName(_fromUtf8("horizontalLayout"))
        self.verticalLayout = QtGui.QVBoxLayout()
        self.verticalLayout.setObjectName(_fromUtf8("verticalLayout"))
        self.horizontalLayout_2 = QtGui.QHBoxLayout()
        self.horizontalLayout_2.setObjectName(_fromUtf8("horizontalLayout_2"))
        self.modelLabel = QtGui.QLabel(self.centralwidget)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Fixed, QtGui.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.modelLabel.sizePolicy().hasHeightForWidth())
        self.modelLabel.setSizePolicy(sizePolicy)
        self.modelLabel.setObjectName(_fromUtf8("modelLabel"))
        self.horizontalLayout_2.addWidget(self.modelLabel)
        self.modelLineEdit = QtGui.QLineEdit(self.centralwidget)
        self.modelLineEdit.setObjectName(_fromUtf8("modelLineEdit"))
        self.horizontalLayout_2.addWidget(self.modelLineEdit)
        self.modelBrowseButton = QtGui.QPushButton(self.centralwidget)
        self.modelBrowseButton.setObjectName(_fromUtf8("modelBrowseButton"))
        self.horizontalLayout_2.addWidget(self.modelBrowseButton)
        self.verticalLayout.addLayout(self.horizontalLayout_2)
        self.dividerLine = QtGui.QFrame(self.centralwidget)
        self.dividerLine.setFrameShape(QtGui.QFrame.HLine)
        self.dividerLine.setFrameShadow(QtGui.QFrame.Sunken)
        self.dividerLine.setObjectName(_fromUtf8("dividerLine"))
        self.verticalLayout.addWidget(self.dividerLine)
        self.inputLabel = QtGui.QLabel(self.centralwidget)
        self.inputLabel.setObjectName(_fromUtf8("inputLabel"))
        self.verticalLayout.addWidget(self.inputLabel)
        self.burnList = QtGui.QListView(self.centralwidget)
        self.burnList.setObjectName(_fromUtf8("burnList"))
        self.verticalLayout.addWidget(self.burnList)
        self.predictButton = QtGui.QPushButton(self.centralwidget)
        self.predictButton.setObjectName(_fromUtf8("predictButton"))
        self.verticalLayout.addWidget(self.predictButton)
        self.display = QtGui.QLabel(self.centralwidget)
        self.display.setMinimumSize(QtCore.QSize(0, 300))
        self.display.setObjectName(_fromUtf8("display"))
        self.verticalLayout.addWidget(self.display)
        self.horizontalLayout.addLayout(self.verticalLayout)
        GUI.setCentralWidget(self.centralwidget)
        self.toolBar = QtGui.QToolBar(GUI)
        self.toolBar.setObjectName(_fromUtf8("toolBar"))
        GUI.addToolBar(QtCore.Qt.TopToolBarArea, self.toolBar)
        self.actionN = QtGui.QAction(GUI)
        self.actionN.setCheckable(True)
        self.actionN.setChecked(True)
        self.actionN.setObjectName(_fromUtf8("actionN"))
        self.actionKg = QtGui.QAction(GUI)
        self.actionKg.setCheckable(True)
        self.actionKg.setObjectName(_fromUtf8("actionKg"))
        self.actionLbs = QtGui.QAction(GUI)
        self.actionLbs.setCheckable(True)
        self.actionLbs.setObjectName(_fromUtf8("actionLbs"))
        self.actionOpenCal = QtGui.QAction(GUI)
        self.actionOpenCal.setShortcut(_fromUtf8(""))
        self.actionOpenCal.setObjectName(_fromUtf8("actionOpenCal"))
        self.actionSaveCal = QtGui.QAction(GUI)
        self.actionSaveCal.setObjectName(_fromUtf8("actionSaveCal"))
        self.actionSaveCalAs = QtGui.QAction(GUI)
        self.actionSaveCalAs.setObjectName(_fromUtf8("actionSaveCalAs"))
        self.actionSave_2 = QtGui.QAction(GUI)
        self.actionSave_2.setObjectName(_fromUtf8("actionSave_2"))
        self.actionOpenRec = QtGui.QAction(GUI)
        self.actionOpenRec.setObjectName(_fromUtf8("actionOpenRec"))
        self.actionSaveRec = QtGui.QAction(GUI)
        self.actionSaveRec.setObjectName(_fromUtf8("actionSaveRec"))
        self.actionSaveRecAs = QtGui.QAction(GUI)
        self.actionSaveRecAs.setObjectName(_fromUtf8("actionSaveRecAs"))
        self.actionExportSnippet = QtGui.QAction(GUI)
        self.actionExportSnippet.setObjectName(_fromUtf8("actionExportSnippet"))
        self.modelLabel.setBuddy(self.modelBrowseButton)

        self.retranslateUi(GUI)
        QtCore.QMetaObject.connectSlotsByName(GUI)

    def retranslateUi(self, GUI):
        GUI.setWindowTitle(_translate("GUI", "Hot Topic", None))
        self.modelLabel.setText(_translate("GUI", "Model:", None))
        self.modelLineEdit.setText(_translate("GUI", "None set...", None))
        self.modelBrowseButton.setText(_translate("GUI", "Browse...", None))
        self.inputLabel.setText(_translate("GUI", "Inputs:", None))
        self.predictButton.setText(_translate("GUI", "Predict!", None))
        self.display.setText(_translate("GUI", "Display", None))
        self.toolBar.setWindowTitle(_translate("GUI", "toolBar", None))
        self.actionN.setText(_translate("GUI", "N", None))
        self.actionKg.setText(_translate("GUI", "kg", None))
        self.actionLbs.setText(_translate("GUI", "lbs", None))
        self.actionOpenCal.setText(_translate("GUI", "Open...", None))
        self.actionSaveCal.setText(_translate("GUI", "Save", None))
        self.actionSaveCalAs.setText(_translate("GUI", "Save As...", None))
        self.actionSave_2.setText(_translate("GUI", "Save", None))
        self.actionOpenRec.setText(_translate("GUI", "Open...", None))
        self.actionSaveRec.setText(_translate("GUI", "Save", None))
        self.actionSaveRecAs.setText(_translate("GUI", "Save As...", None))
        self.actionExportSnippet.setText(_translate("GUI", "Export Snippet...", None))

