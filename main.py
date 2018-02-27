import numpy as np

from lib import rawdata
from lib import dataset
from lib import metrics
from lib import viz
from lib import preprocess
from lib import util
from lib import model

def openDatasets():
    data = rawdata.load()
    masterDataSet = dataset.Dataset(data, dataset.Dataset.vulnerablePixels)
    ptList = masterDataSet.sample(sampleEvenly=False)
    # masterDataSet.masks = dataset.Dataset.toDict(ptList)
    trainPts, validatePts, testPts =  util.partition(ptList, ratios=[.6,.7])
    train = dataset.Dataset(data, trainPts)
    validate = dataset.Dataset(data, validatePts)
    test = dataset.Dataset(data, testPts)
    return train, validate, test

def openAndTrain():
    from lib import model

    # data = rawdata.RawData.load(burnNames='all', dates='all')
    # masterDataSet = dataset.Dataset(data, dataset.Dataset.vulnerablePixels)
    # masterDataSet.masks = masterDataSet.evenOutPositiveAndNegative()
    train, validate, test = openDatasets()
    train.save('train')
    test.save('test')
    validate.save('validate')
    # print(train, validate, test)
    mod = getModel()
    mod.fit(train, validate)
    mod.saveWeights()
    predictions = mod.predict(test)
    util.savePredictions(predictions)
    return test, predictions

def reloadPredictions():
    predFName = "09Nov10:39.csv"
    predictions = util.openPredictions('output/predictions/'+predFName)
    test = dataset.Dataset.open("output/datasets/test.json")
    return test, predictions

def getModel(weightsFile=None):
    from lib import model
    numWeatherInputs = 8
    usedLayers = ['dem','ndvi', 'aspect', 'band_2', 'band_3', 'band_4', 'band_5'] #, 'slope'
    AOIRadius = 30
    pp = preprocess.PreProcessor(numWeatherInputs, usedLayers, AOIRadius)

    mod = model.FireModel(pp, weightsFile)
    return mod

def example():
    try:
        import sys
        modfname = sys.argv[1]
        datasetfname = sys.argv[2]
    except:
        print('about to import tkinter')
        from tkinter import Tk
        from tkinter.filedialog import askopenfilename, askdirectory
        print('done!')
        root = Tk()
        print('Tked')
        root.withdraw()
        print('withdrawn')
        modfname = askdirectory(initialdir = "models/",title="choose a model")
        datasetfname = askdirectory(initialdir = "output/datasets",title="choose a dataset")
        root.destroy()
    print('here')
    test = dataset.openDataset(datasetfname)
    mod = getModel(modfname)
    # mod.fit(train, val)
    predictions = mod.predict(test)
    # test, predictions = openAndTrain()
    res = viz.visualizePredictions(test, predictions)
    viz.showPredictions(res)

def preProcess():
    m = model.load("/Users/nickcrews/Documents/CSThesis/mcscn/mlthesis/models/15Nov09_41")
    ds = dataset.load("/Users/nickcrews/Documents/CSThesis/mcscn/mlthesis/datasets/even.npz")
    ds.filterPoints(ds.vulnerablePixels)
    ds.evenOutPositiveAndNegative()
    m.preProcessor.processAndSave(ds)

def train(m):
    m.fit_generator("/Users/nickcrews/Documents/CSThesis/mlthesis/processed/23Feb15-09", epochs=10)
    return m

def load():
    return model.load("/Users/nickcrews/Documents/CSThesis/mlthesis/models/15Nov09_41")

def mcscnMakeDS():
    ds = dataset.load()
    print(ds)
    ds.filterPoints(ds.vulnerablePixels)
    ds.evenOutPositiveAndNegative()
    ds.save('training.npz')
    return ds

def mcscnPreprocess():
    print('loading model...')
    m = model.load("models/15Nov09_41")
    print('\rloading model...done\nloading dataset...')
    ds = dataset.load('datasets/training.npz')
    print('\rloading dataset...done\nprocessing...')
    m.preProcessor.processAndSave(ds)

def mcscnTrain(modelFile=None):
    if modelFile is None:
        m = model.OurModel()
    else:
        m = model.load(modelFile)
    directory = 'processed/26Feb14-49'
    m.fit_generator(directory, epochs=15, steps_per_epoch=1)
    m.save()

ds = mcscnTrain()
