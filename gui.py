import numpy as np
import os
from pyqtgraph.Qt import QtCore, QtGui, uic

# dynamically generate the gui skeleton file from the ui file
with open('basicgui.py', 'w') as pyfile:
    uic.compileUi('basicgui.ui', pyfile)
import basicgui

from lib import rawdata, dataset

class GUI(basicgui.Ui_GUI, QtCore.QObject):

    sigPredict = QtCore.pyqtSignal(str,str)

    def __init__(self, app):
        QtCore.QObject.__init__(self)
        self.app = app
        self.mainwindow = QtGui.QMainWindow()
        self.qdir = QtCore.QDir()
        self.setupUi(self.mainwindow)

        self.getFires()

        self.modelBrowseButton.clicked.connect(self.browseModels)
        self.predictButton.clicked.connect(self.predict)

        img = np.random.random((200,600))*255
        self.showImage(img)

        self.predictions = {}

        self.mainwindow.show()

    def getFires(self):
        burnFolder = '/home/n_crews/Documents/thesis/mlthesis/data/'
        burns = os.listdir(burnFolder)
        model = QtGui.QStandardItemModel()
        for name in burns:
            if name.startswith('.'):
                continue
            item = QtGui.QStandardItem(name)
            item.setCheckable(True)
            item.setFlags(QtCore.Qt.ItemIsUserCheckable | QtCore.Qt.ItemIsEnabled)
            item.setCheckable(True)
            model.appendRow(item)

        self.burnList.setModel(model)

    def browseModels(self):
        fname = QtGui.QFileDialog.getExistingDirectory(directory='models/', options=QtGui.QFileDialog.ShowDirsOnly)
        if fname == '':
            return
        self.modelLineEdit.setText(fname)
        print(fname, type(fname))

    def predict(self):
        selectedBurns = []
        mod = self.burnList.model()
        for index in range(mod.rowCount()):
            i = mod.item(index)
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
        predictions = mod.predict(ds)
        self.predictions.update(predictions)

    def showImage(self, img):
        h,w = img.shape[:2]
        QI=QtGui.QImage(img, w, h, QtGui.QImage.Format_Indexed8)
        self.display.setPixmap(QtGui.QPixmap.fromImage(QI))
class CheckableDirModel(QtGui.QDirModel):
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


if __name__ == '__main__':
    app = QtGui.QApplication([])
    gui = GUI(app)

    app.exec_()
