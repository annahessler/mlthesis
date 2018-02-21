import os
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

def async(*args):

    assert len(args) in (0,1)
    if len(args) == 1:
        callback = args[0]
    else:
        callback = False

    class Runner(QtCore.QThread):

        if callback:
            mySignal = QtCore.pyqtSignal(object,name="mySignal")

        def __init__(self, target, *args, **kwargs):
            super().__init__()
            self._target = target
            self._args = args
            self._kwargs = kwargs
            self.callback = callback

        def run(self):
            print('hi!')
            result = self._target(*self._args, **self._kwargs)
            print('done here')
            if self.callback:
                self.mySignal.emit(result)


    def async_func(func):
        runner = Runner(func)
        # Keep the runner somewhere or it will be destroyed
        func.__runner = runner
        if callback:
            runner.mySignal.connect(callback)
        print('starting thread!')
        runner.start()

    return async_func

class GUI(basicgui.Ui_GUI, QtCore.QObject):

    sigPredict = QtCore.pyqtSignal(str,str)

    def __init__(self, app):
        QtCore.QObject.__init__(self)
        self.app = app
        self.mainwindow = QtWidgets.QMainWindow()
        self.setupUi(self.mainwindow)

        self.data = rawdata.load()
        self.model = None
        self.dataset = None

        # do stuff
        self.initBurnTree()

        self.modelBrowseButton.clicked.connect(self.browseModels)
        self.loadDatasetButton.clicked.connect(self.browseDatasets)
        self.predictButton.clicked.connect(self.predict)

        # img = np.random.random((200,600))*255
        # self.showImage(img,self.display)
        # self.predictions = {}

        test()
        self.mainwindow.show()

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
                # dateItem.setCheckable(True)
                # dateItem.setCheckState(QtCore.Qt.Unchecked)
                dateItem.setSelectable(True)
                burnItem.appendRow(dateItem)
        self.burnTree.setColumnWidth(0, 300)
        self.burnTree.expandAll()
        self.burnTree.selectionModel().selectionChanged.connect(self.burnDataSelected)

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

    def displayBurn(self, burnName):
        burn = self.data.burns[burnName]
        dem = viz.renderBurn(burn)
        SIZE = (400,300)
        resized = cv2.resize(dem, SIZE)
        self.showImage(resized, self.burnDisplay)

    def displayDay(self, burnName, date):
        day = self.data.burns[burnName].days[date]
        img = viz.renderDay(day)
        SIZE = (400,300)
        resized = cv2.resize(img, SIZE)
        self.showImage(resized, self.burnDisplay)
        # print('displaying day:', day)

    def browseModels(self):
        # print('browsing!')
        fname = QtWidgets.QFileDialog.getExistingDirectory(directory='models/', options=QtWidgets.QFileDialog.ShowDirsOnly)
        try:
            self.model = model.load(fname)
            self.modelLineEdit.setText(fname)
            self.predictModelLineEdit.setText(fname)
            self.trainModelLineEdit.setText(fname)
        except:
            print('could not open that model')
        # img = viz.renderModel(self.model)
        # self.showImage(self.modelDisplay, img)

    def browseDatasets(self):
        fname, _ = QtWidgets.QFileDialog.getOpenFileName(directory='datasets/', filter="numpy archives (*.npz)")
        try:
            self.dataset = dataset.load(fname)
            self.datasetDatasetLineEdit.setText(fname)
            self.trainDatasetLineEdit.setText(fname)
            self.predictDatasetLineEdit.setText(fname)
        except Exception as e:
            print('Could not open that dataset:', e)

    def donePredicting(self, result):
        print('got a result:', result)
        pass

    @async(donePredicting)
    def predict(self):
        print('starting predictions...')
        time.sleep(3)
        print('done computing')
        return 'this is the result'

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

# def dec(func):
#
#     def newFunc():
#         print('starting')
#         func()
#         print('done')
#
#     return newFunc
#
# @dec
# def work():
#     print('im doing work')
#
# work()

if __name__ == '__main__':
    app = QtWidgets.QApplication([])
    gui = GUI(app)

    app.exec_()
