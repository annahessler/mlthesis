
import numpy as np

from lib import rawdata
from lib import dataset
from lib import metrics
from lib import viz
from lib import preprocess

def openAndTrain():
    from lib import model

    data = rawdata.RawData.load(burnNames='all', dates='all')
    masterDataSet = dataset.Dataset(data, dataset.Dataset.vulnerablePixels)
    masterDataSet.points = masterDataSet.evenOutPositiveAndNegative()
    train, validate, test = masterDataSet.split(ratios=[.6,.7])
    train.save('train')
    test.save('test')
    # print(train, validate, test)

    numWeatherInputs = 8
    usedLayers = ['dem','ndvi']
    AOIRadius = 30
    pp = preprocess.PreProcessor(numWeatherInputs, usedLayers, AOIRadius)

    mod = model.FireModel(pp)
    mod.fit(train, validate)
    predictions = mod.predict(test)
    viz.savePredictions(predictions)
    return test, predictions

def reload():
    predFName = "08Nov15:37.csv"
    predictions = viz.openPredictions('output/predictions/'+predFName)
    test = dataset.Dataset.open("output/datasets/test.json")
    return test, predictions

test, predictions = reload()
res = viz.visualizePredictions(test, predictions)
# viz.show(res)
# viz.save(res,'predictions')
