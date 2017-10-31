
import numpy as np

# from lib import model
from lib import datamodule
from lib import dataset
from lib import viz

data = datamodule.Data.defaultData('0731')
dataset = dataset.Dataset(data)

usedLayers = ['perim', 'ndvi', 'slope', 'aspect']
dataset.usedLayers = usedLayers
# dataset.usedWeather = ['maxTemp','avgHum']
dataset.usedWeather = []
trainData, testData = dataset.split()

weather, aois, out = trainData.getData()
nsamples, height, width, nchannels = aois.shape

# mod = model.FireModel(weather.shape[1], nchannels, (height, width))
# viz.saveModel(mod)
# mod.fit(trainData)
# predictions = mod.predict(testData)

predictions = np.loadtxt("output/predictions/res.csv")
print(predictions)

res = viz.visualizePredictions(testData, predictions)
viz.show(res)
viz.save(res,'predictions')
