# preprocess.py
import numpy as np

class PreProcessor(object):
    '''What is responsible for extracting the used data from the dataset and then
    normalizing or doing any other steps before feeding it into the network.'''

    def __init__(self, numWeatherInputs, numLayers, AOIRadius):
        self.numWeatherInputs = numWeatherInputs
        self.numLayers = numLayers
        self.AOIRadius = AOIRadius

    def process(self, dataset):
        '''Take a dataset and return the extracted inputs and outputs'''
        # get all of the weather for each day, normalize them, and compute metrics
        namesAndDates = dataset.getUsedBurnNamesAndDates()
        wms = []
        for burnName, date in namesAndDates:
            bNd = burnName, date
            wm = dataset.data.getWeather(burnName, date)
            wms.append(wm)

        calculateWeatherMetrics(wms)

        return [weatherInputs, imgInputs], outputs

def calculateWeatherMetrics(weatherMatrices):

    for wm in weatherMatrices:
        temp, dewpt, temp2, wdir, wspeed, precip, hum = wm


def totalPrecip(weatherMatrix):
    temp, dewpt, temp2, wdir, wspeed, precip, hum = weatherMatrix
    return sum(precip)

def mergeSamples(samples):
    weather, spatial, outputs = [], [], []
    for sample in samples:
        w, s = sample.getInputs()
        weather.append(w)
        spatial.append(s)
        outputs.append(sample.getOutput())
    return [np.array(weather), np.array(spatial)], np.array(outputs)

def getInputsAndOutputs(dataset, inputSettings):
    # get the weather data for each day
    samples = dataset.getSamples(inputSettings)
    return mergeSamples(samples)

def normalize(inputs, outputs):
    weather, spatial = inputs

def windMetrics(weatherData):
    col = 4
    n, s, e, w = 0

    for i in np.shape(weatherData)[0]:
        if weatherData[i][col] > 90 and weatherData[i][col] < 270: #going north
            ''' sin(wind direction) * wind speed '''
            n += (np.sin(weatherData[i][col]) * weatherData[i][col + 1])
        if weatherData[i][col] < 90 and weatherData[i][col] > 270: #going south
            s += (np.sin(weatherData[i][col]) * weatherData[i][col + 1])
        if weatherData[i][col] < 360 and weatherData[i][col] > 180: #going east
            e += (np.cos(weatherData[i][col]) * weatherData[i][col + 1])
        if weatherData[i][col] > 0 and weatherData[i][col] < 180: #going west
            w += (np.cos(weatherData[i][col]) * weatherData[i][col + 1])

    weather = [n, s, e, w]
    return weather


class Sample(object):
    '''An actual sample that can be fed into the network.

    Encapsulates both the input and output data, as well as the Point it was taken from'''

    # if we calculate the weatherMetric for a Day, memo it
    memoedWeatherMetrics = {}
    memoedPadded = {}

    def __init__(self, data, point, inputSettings):
        self.data = data
        self.point = point
        self.inputSettings = inputSettings

    def getInputs(self):
        burnName, date, location = self.point
        burn = self.data.burns[burnName]
        day = burn.days[date]

        inpset = self.inputSettings
        if (burnName, date) in Sample.memoedWeatherMetrics:
            weatherMetrics = Sample.memoedWeatherMetrics[(burnName, date)]
        else:
            weatherMetrics = inpset.weatherMetrics.calculate(day.weather)
            Sample.memoedWeatherMetrics[(burnName, date)] = weatherMetrics

        if (burnName, date) not in Sample.memoedPadded:
            padded = self.stackAndPadLayers(burn.layers, day.startingPerim)
            Sample.memoedPadded[(burnName, date)] = padded
        else:
            padded = Sample.memoedPadded[(burnName, date)]
        aoi = self.extract(padded, location)

        return [weatherMetrics, aoi]

    def extract(self, padded, location):
        '''Assume padded is bordered by radius self.inputSettings.AOIRadius'''
        y,x = location
        r = self.inputSettings.AOIRadius
        lox = r+(x-r)
        hix = r+(x+r+1)
        loy = r+(y-r)
        hiy = r+(y+r+1)
        aoi = padded[loy:hiy,lox:hix]
        # print(stacked.shape, padded.shape)s
        return aoi

    def stackAndPadLayers(self, layers, startingPerim):
        usedLayerNames, metric, AOIRadius = self.inputSettings
        if usedLayerNames == 'all':
            names = list(layers.keys())
            names.sort()
            usedLayers = [layers[name] for name in names]
        else:
            usedLayers = [layers[name] for name in usedLayerNames]
        usedLayers.insert(0, startingPerim)
        stacked = np.dstack(usedLayers)

        r = AOIRadius
        # pad with zeros around border of image
        padded = np.lib.pad(stacked, ((r,r),(r,r),(0,0)), 'constant')
        return padded

    def getOutput(self):
        burnName, date, (y, x) = self.point
        burn = self.data.burns[burnName]
        day = burn.days[date]
        return day.endingPerim[y, x]

    def __repr__(self):
        return "Sample({}, {}, {})".format(self.data, self.point, self.inputSettings)

class InputSettings(object):

    def __init__(self, usedLayerNames, AOIRadius=30, weatherMetrics=None, ):
        self.AOIRadius = AOIRadius
        self.weatherMetrics = weatherMetrics if weatherMetrics is not None else InputSettings.dummyMetric
        self.usedLayerNames = usedLayerNames
        assert type(self.usedLayerNames) == list
        assert len(usedLayerNames) > 0

    @staticmethod
    def dummyMetric(weatherMatrix):
        return 42
