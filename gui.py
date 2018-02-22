import os
# ignore the gross warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"]="3"
import time
import numpy as np
from functools import wraps
import cv2
import multiprocessing
from PyQt5 import QtCore, QtGui, QtWidgets, uic

# dynamically generate the gui skeleton file from the ui file
with open('basicgui.py', 'w') as pyfile:
    uic.compileUi('basicgui.ui', pyfile)
import basicgui

from lib import rawdata
from lib import dataset
from lib import util
from lib import model
from lib import viz
from lib.async import async

class GUI(basicgui.Ui_GUI, QtCore.QObject):

    sigPredict = QtCore.pyqtSignal(str,str)

    BURN_RENDER_SIZE = (300,200)

    def __init__(self, app):
        QtCore.QObject.__init__(self)
        self.app = app
        self.mainwindow = QtWidgets.QMainWindow()
        self.setupUi(self.mainwindow)

        self.data = rawdata.load()
        self.model = None
        self.dataset = dataset.emptyDataset(self.data)

        # do stuff
        self.initBurnTree()
        self.initDatasetTree()

        self.modelBrowseButton.clicked.connect(self.browseModels)
        self.loadDatasetButton.clicked.connect(self.browseDatasets)
        self.predictButton.clicked.connect(self.predictButtonPressed)

        # img = np.random.random((200,600))*255
        # self.showImage(img,self.display)
        # self.predictions = {}

        self.mainwindow.show()
        # self.useModel("/Users/nickcrews/Documents/CSThesis/mlthesis/models/15Nov09_41")
        # self.useDataset("/Users/nickcrews/Documents/CSThesis/mlthesis/datasets/21Feb11-42.npz")
        # self.predict()

    def initBurnTree(self):
        model = QtGui.QStandardItemModel()
        self.burnTree.setModel(model)
        burns = sorted(self.data.burns.keys())
        for name in burns:
            burnItem = QtGui.QStandardItem(name)
            burnItem.setSelectable(True)
            model.appendRow(burnItem)
            dates = sorted(self.data.burns[name].days.keys())
            for d in dates:
                dateItem = QtGui.QStandardItem(d)
                dateItem.setCheckable(True)
                dateItem.setCheckState(QtCore.Qt.Unchecked)
                dateItem.setSelectable(True)
                burnItem.appendRow(dateItem)
        self.burnTree.setColumnWidth(0, 300)
        self.burnTree.expandAll()
        self.burnTree.selectionModel().selectionChanged.connect(self.burnDataSelected)
        self.burnTree.clicked.connect(self.dayChecked)

    def initDatasetTree(self):
        model = QtGui.QStandardItemModel()
        self.datasetTree.setModel(model)
        self.datasetTree.expandAll()
        self.datasetTree.setColumnWidth(0, 300)
        self.datasetTree.selectionModel().selectionChanged.connect(self.datasetDaySelected)

    def dayChecked(self, modelIndex):
        # idx = modelIndex.indexes()[0]
        dateItem = self.burnTree.model().itemFromIndex(modelIndex)
        p = dateItem.parent()
        burnName, date = p.text(), dateItem.text()
        if dateItem.checkState() == QtCore.Qt.Checked:
            self.dataset.add(burnName, date)
        else:
            self.dataset.remove(burnName, date)
        self.displayDataset()

    def burnDataSelected(self, selected, deselected):
        idx = selected.indexes()[0]
        item = self.burnTree.model().itemFromIndex(idx)
        p = item.parent()
        if p is None:
            # must have selected a burn, not a date
            self.displayBurn(item.text())
        else:
            # selected a date
            self.displayDay(p.text(), item.text())

    def datasetDaySelected(self,selected, deselected):
        idx = selected.indexes()[0]
        item = self.datasetTree.model().itemFromIndex(idx)
        p = item.parent()
        burnName, date = p.text(), item.text()
        self.displayDatasetDay(burnName, date)

    def displayDatasetDay(self, burnName, date):
        mask = self.dataset.points[burnName][date]
        resized = cv2.resize(mask, self.BURN_RENDER_SIZE)*255
        self.showImage(resized, self.datasetDisplay)

    def displayDataset(self):
        model = self.datasetTree.model()
        model.clear()
        burnNames = sorted(self.dataset.points.keys())
        for name in burnNames:
            burnItem = QtGui.QStandardItem(name)
            burnItem.setSelectable(True)
            model.appendRow(burnItem)
            dates = sorted(self.dataset.points[name].keys())
            for d in dates:
                dateItem = QtGui.QStandardItem(d)
                dateItem.setSelectable(True)
                burnItem.appendRow(dateItem)
        self.datasetTree.expandAll()

    def displayBurn(self, burnName):
        burn = self.data.burns[burnName]
        dem = viz.renderBurn(burn)
        resized = cv2.resize(dem, self.BURN_RENDER_SIZE)
        self.showImage(resized, self.burnDisplay)

    def displayDay(self, burnName, date):
        day = self.data.burns[burnName].days[date]
        img = viz.renderDay(day)
        resized = cv2.resize(img, self.BURN_RENDER_SIZE)
        self.showImage(resized, self.burnDisplay)
        # print('displaying day:', day)

    def browseModels(self):
        # print('browsing!')
        fname = QtWidgets.QFileDialog.getExistingDirectory(directory='models/', options=QtWidgets.QFileDialog.ShowDirsOnly)
        self.useModel(fname)

    def useModel(self, fname):
        try:
            self.model = model.load(fname)
        except:
            print('could not open that model')
            return
        self.modelLineEdit.setText(fname)
        self.predictModelLineEdit.setText(fname)
        self.trainModelLineEdit.setText(fname)

    def browseDatasets(self):
        fname, _ = QtWidgets.QFileDialog.getOpenFileName(directory='datasets/', filter="numpy archives (*.npz)")
        self.useDataset(fname)

    def useDataset(self, fname):
        try:
            self.dataset = dataset.load(fname)
        except Exception as e:
            print('Could not open that dataset:', e)
            return
        self.datasetDatasetLineEdit.setText(fname)
        self.trainDatasetLineEdit.setText(fname)
        self.predictDatasetLineEdit.setText(fname)
        self.displayDataset()

    def predictButtonPressed(self, checked=0):
        self.predict()

    def donePredicting(self, result):
        print('got a result:', result)

    @async(callback=donePredicting)
    def predict(self):
        print('starting predictions')
        result = self.model.predict(self.dataset)
        print('done with work')
        return self, 'this is the result'

    @staticmethod
    def showImage(img, label):
        if img.dtype.kind == 'f':
            # convert from float to uint8
            img = (img*255).astype(np.uint8)
        assert img.dtype == np.uint8
        if len(img.shape) > 2:
            # color images
            h, w, bytesPerComponent = img.shape
            bytesPerLine = bytesPerComponent * w;
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            QI=QtGui.QImage(img.data, w, h, bytesPerLine, QtGui.QImage.Format_RGB888)
        else:
            # black and white images
            h,w = img.shape[:2]
            QI=QtGui.QImage(img, w, h, QtGui.QImage.Format_Indexed8)
        # QI.setColorTable(COLORTABLE)
        label.setPixmap(QtGui.QPixmap.fromImage(QI))

if __name__ == '__main__':
    app = QtWidgets.QApplication([])
    gui = GUI(app)

    app.exec_()
