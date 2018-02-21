import numpy as np
import os
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

class GUI(basicgui.Ui_GUI, QtCore.QObject):

    sigPredict = QtCore.pyqtSignal(str,str)

    def __init__(self, app):
        QtCore.QObject.__init__(self)
        self.app = app
        self.mainwindow = QtWidgets.QMainWindow()
        self.qdir = QtCore.QDir()
        self.setupUi(self.mainwindow)

        # do stuff
        # chooseBurnsModel = CheckableDirModel()
        # self.burnTree.setModel(chooseBurnsModel)
        # self.burnTree.setRootIndex(chooseBurnsModel.index(self.qdir.absoluteFilePath('/home/n_crews/Documents/thesis/mlthesis/data/')))
        # self.burnTree.setRootIndex(chooseBurnsModel.index(QtGui.QDir.currentPath()));
        self.getFires()

        self.modelBrowseButton.clicked.connect(self.browseModels)
        self.predictButton.clicked.connect(self.predict)

        img = np.random.random((200,600))*255
        self.showImage(img,self.display)

        self.predictions = {}
        self.burnSelections = {}

        self.mainwindow.show()

    def getFires(self):
        burnFolder = os.path.abspath('data/')
        burns = util.listdir_nohidden(burnFolder)
        model = QtGui.QStandardItemModel()

        # model = QtWidgets.QFileSystemModel()
        # model = QtWidgets.QFileSystemModel()
        self.burnTree.setModel(model)
        # self.burnTree.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectItems)
        # self.burnTree.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
        # model.setRootPath(burnFolder)
        # self.burnTree.setRootIndex(model.index(burnFolder))
        # parentItem = model.invisibleRootItem()
        for name in burns:
            burnItem = QtGui.QStandardItem(name)
            burnItem.setSelectable(True)
            model.appendRow(burnItem)
            dates = util.availableDates(name)
            for d in dates:
                dateItem = QtGui.QStandardItem(d)
                dateItem.setCheckable(True)
                dateItem.setSelectable(True)
                # dateItem.setFlags(QtCore.Qt.ItemIsUserCheckable | QtCore.Qt.ItemIsEnabled)
                dateItem.setCheckState(QtCore.Qt.Unchecked)
                burnItem.appendRow(dateItem)
            if len(dates):
                burnItem.setCheckable(True)
                # burnItem.setFlags(QtCore.Qt.ItemIsUserCheckable | QtCore.Qt.ItemIsEnabled)
                burnItem.setCheckState(QtCore.Qt.Unchecked)

        self.burnTree.setColumnWidth(0, 300)
        # self.burnTree.selectionChanged.connect(self.datasetClicked)
        # self.burnTree.selectionChanged.connect(datasetClicked)
        self.burnTree.selectionModel().selectionChanged.connect(self.burnDataSelected)
        # print(model.headerData())
        # model.setHorizontalHeaderLabels([])
        # self.burnTree.clicked.connect(self.datasetClicked)

    def burnDataSelected(self, selected, deselected):
        # print('clicked', selected)
        idx = selected.indexes()[0]
        # print(idx)
        item = self.burnTree.model().itemFromIndex(idx)
        # print(item, item.text())
        p = item.parent()
        # print(p)
        if p is None:
            # must have selected a burn, not a date
            self.displayBurn(item.text())
        else:
            # selected a date
            self.displayDay(p.text(), item.text())

    def displayBurn(self, burnName):
        print('displaying burn: ' + burnName)

    def displayDay(self, burnName, date):
        print('displaying day:', burnName, date)

    def browseModels(self):
        # print('browsing!')
        fname = QtWidgets.QFileDialog.getExistingDirectory(directory='models/', options=QtWidgets.QFileDialog.ShowDirsOnly)
        if fname == '':
            return
        self.modelLineEdit.setText(fname)
        self.model = model.load(fname)
        print(fname)
        img = viz.renderModel(self.model)
        self.showImage(self.modelDisplay, img)


    def predict(self):
        selectedBurns = []
        mod = self.burnTree.model()
        for index in range(mod.rowCount()):
            i = mod.item(index)
            # print(i.checkState())
            if i.checkState() == QtCore.Qt.Checked:
                selectedBurns.append(i.text())
        print('opening the data for the burns,', selectedBurns)
        data = rawdata.RawData.load(burnNames=selectedBurns, dates='all')
        ds = dataset.Dataset(data, dataset.Dataset.vulnerablePixels)

        from lib import model
        modelFileName = self.modelLineEdit.text()
        print('loading model', modelFileName)
        mod = model.load(modelFileName)
        print(mod)
        # predictions = mod.predict(ds)
        # self.predictions.update(predictions)
        # print('opening modelFileName!')
        # todo: load keras model
        # self.sigPredict.emit(modelFileName, burnName)

    def donePredicting():
        pass

    @staticmethod
    def showImage(img, label):
        h,w = img.shape[:2]
        QI=QtGui.QImage(img, w, h, QtGui.QImage.Format_Indexed8)
        # QI.setColorTable(COLORTABLE)
        label.setPixmap(QtGui.QPixmap.fromImage(QI))

class CheckableDirModel(QtWidgets.QDirModel):
    def __init__(self, parent=None):
        QtGui.QDirModel.__init__(self, None)
        self.checks = {}

    def data(self, index, role=QtCore.Qt.DisplayRole):
        if role != QtCore.Qt.CheckStateRole:
            return QtGui.QDirModel.data(self, index, role)
        else:
            if index.column() == 0:
                return self.checkState(index)

    def flags(self, index):
        return QtGui.QDirModel.flags(self, index) | QtCore.Qt.ItemIsUserCheckable

    def checkState(self, index):
        if index in self.checks:
            return self.checks[index]
        else:
            return QtCore.Qt.Unchecked

    def setData(self, index, value, role):
        if (role == QtCore.Qt.CheckStateRole and index.column() == 0):
            self.checks[index] = value
            self.emit(QtCore.SIGNAL("dataChanged(QModelIndex,QModelIndex)"), index, index)
            return True

        return QtGui.QDirModel.setData(self, index, value, role)

def async(func, args, callback):
    pass

if __name__ == '__main__':
    app = QtWidgets.QApplication([])
    gui = GUI(app)

    app.exec_()
