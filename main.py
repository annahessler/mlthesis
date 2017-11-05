import matplotlib as mpl
mpl.use('Agg')
import numpy as np

from lib import rawdata
from lib import dataset
from lib import metrics
from lib import model
from lib import viz

data = rawdata.RawData.load(burnNames='all', dates='all')
data = data.augment()
masterDataSet = dataset.Dataset(data, dataset.Dataset.vulnerablePixels)
masterDataSet.points = masterDataSet.evenOutPositiveAndNegative()
train, validate, test = masterDataSet.split(ratios=[.6,.7])
# print(train, validate, test)

usedLayers = ['dem','ndvi']
AOIRadius = 30
wm = metrics.WeatherMetric()
inputSettings = model.InputSettings(usedLayers, wm, AOIRadius)

mod = model.FireModel(inputSettings)
mod.fit(train, validate, test)
predictions = mod.predict(test)
np.savetxt("res.csv", predictions, delimiter=',')


# res = viz.visualizePredictions(test, predictions)
# viz.show(res)
# viz.save(res,'predictions')
