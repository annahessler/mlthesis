
import numpy as np

from lib import rawdata
from lib import dataset
from lib import metrics
from lib import model
from lib import viz
from lib import preprocess

data = rawdata.RawData.load(burnNames='all', dates='all')
masterDataSet = dataset.Dataset(data, dataset.Dataset.vulnerablePixels)
masterDataSet.points = masterDataSet.evenOutPositiveAndNegative()
train, validate, test = masterDataSet.split(ratios=[.6,.7])
# print(train, validate, test)

numWeatherInputs = 8
usedLayers = ['dem','ndvi']
AOIRadius = 30
pp = preprocess.PreProcessor(numWeatherInputs, usedLayers, AOIRadius)

mod = model.FireModel(pp)
mod.fit(train, validate)
predictions = mod.predict(test)
np.savetxt("res.csv", predictions, delimiter=',')


# res = viz.visualizePredictions(test, predictions)
# viz.show(res)
# viz.save(res,'predictions')
