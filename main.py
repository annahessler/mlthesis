
import numpy as np

from lib import model
from lib import datamodule
from lib import dataset
from lib import viz
from lib import rawdata
from lib import datasetchange

data = rawdata.RawData.load(burnNames='all', dates='all')
data = data.augment()
masterDataSet = datasetchange.Dataset(data, whichBurns='all', whichDates='all')
masterDataSet.whichPixels = masterDataSet.findVulnerablePixels(radius=500)

# data = datamodule.Data.defaultData('0731')
# dataset = dataset.Dataset(data)

usedLayers = ['perim', 'ndvi', 'slope', 'aspect']
dataset.usedLayers = usedLayers
dataset.usedWeather = []
#dataset.usedWeather = ['maxTemp','avgHum']
trainData, validateData, testData = dataset.split()

mod = model.FireModel(*trainData.getModelParams())
mod.fit(trainData, validateData)
predictions = mod.predict(testData)
np.savetxt("res.csv", predictions, delimiter=',')


res = viz.visualizePredictions(testData, predictions)
viz.show(res)
viz.save(res,'predictions')
