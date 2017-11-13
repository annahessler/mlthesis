import numpy as np
import cv2

PIXEL_SIZE = 30

class Data(object):
    '''Contains an layered image of input data and an image of output data, as well as non-spatial data such as weather'''

    def __init__(self):
        self.shape = None

        self.layers = {}
        self.weather = {}
        self.output = None

    def addLayer(self, layerName, data, use=True):
        '''Add the actual layer of data(either spatial 2D or 1D weather/spatial-invariant) to our set'''
        assert len(data.shape) == 2
        if self.shape is not None:
            if self.shape != data.shape:
                raise ValueError("All spatial data must be the same dimension.")
        else:
            self.shape = data.shape
        self.layers[layerName] = data

    def addData(self, variableName, data, use=True):
        self.weather[variableName] = data

    def addOutput(self, data):
        if self.shape is not None:
            if self.shape != data.shape:
                raise ValueError("All spatial data must be the same dimension.")
        else:
            self.shape = data.shape
        self.output = data

    def stackLayers(self, layerNames=None):
        if layerNames is None:
            layers = self.layers.values()
        else:
            layers = [self.layers[name] for name in layerNames]
        stacked = np.dstack(tuple(layers))
        return stacked

    def stackWeather(self, variableNames=None):
        if variableNames is None:
            metrics = list(self.weather.values())
        else:
            metrics = [self.weather[name] for name in variableNames]
        stacked = np.array(metrics)
        return stacked

    def getOutput(self):
        '''Return a 1D array of output values, ready to be used by the model'''
        if self.output is None:
            raise ValueError("Output data must be set.")
        return self.output

    @staticmethod
    def defaultData(dateString):
        d = Data()

        dem = cv2.imread('data/raw/dem.tif', cv2.IMREAD_UNCHANGED)
        slope = cv2.imread('data/raw/slope.tif',cv2.IMREAD_UNCHANGED)
        landsat = cv2.imread('data/raw/landsat.png', cv2.IMREAD_UNCHANGED)
        ndvi = cv2.imread('data/raw/NDVI_1.tif', cv2.IMREAD_UNCHANGED)
        aspect = cv2.imread('data/raw/aspect.tif', cv2.IMREAD_UNCHANGED)
        d.addLayer('dem', dem)
        d.addLayer('slope', slope)
        for name, band in zip(['r','g','b','nir'], cv2.split(landsat)):
            d.addLayer(name,band)
        d.addLayer('ndvi', ndvi)
        d.addLayer('aspect', aspect)

        perim = Data.openStartingPerim(dateString)
        d.addLayer('perim', perim)

        weatherData = Data.createWeatherMetrics(Data.openWeatherData(dateString))
        for name, data in zip(['maxTemp', 'avgWSpeed', 'avgWDir', 'precip', 'avgHum'], weatherData):
            d.addData(name, data)

        output = Data.openEndingPerim(dateString)
        d.addOutput(output)

        return d

    def __repr__(self):
        res = "Data("
        res += "layers:" + repr(self.layers.keys())
        res += ", data:" + repr(self.weather.keys())
        return res

    def openStartingPerim(dateString):
        perimFileName = 'data/raw/perims/' + dateString + '.tif'
        perim = cv2.imread(perimFileName, cv2.IMREAD_UNCHANGED)*255
        return perim

    def openEndingPerim(dateString):
        '''Get the fire perimeter on the next day'''
        month, day = dateString[:2], dateString[2:]
        nextDay = str(int(day)+1).zfill(2)
        guess = month+nextDay
        perimFileName = 'data/raw/perims/' + guess + '.tif'
        # print(perimFileName)
        perim = cv2.imread(perimFileName, cv2.IMREAD_UNCHANGED)
        if perim is None:
            # overflowed the month, that file didnt exist
            nextMonth = str(int(month)+1).zfill(2)
            guess = nextMonth+'01'
            perimFileName = 'data/raw/perims/' + guess + '.tif'
            # print(perimFileName)
            perim = cv2.imread(perimFileName, cv2.IMREAD_UNCHANGED)
            if perim is None:
                raise RuntimeError('Could not find a perimeter for the day after ' + dateString)
        return perim

    def openWeatherData(dateString):
        fname = 'data/raw/weather/' + dateString + '.csv'
        # the first row is the headers, and only cols 4-11 are actual data
        data = np.loadtxt(fname, skiprows=1, usecols=range(5,12), delimiter=',').T
        return data

    def createWeatherMetrics(weatherData):
        temp, dewpt, temp2, wdir, wspeed, precip, hum = weatherData
        avgWSpeed = sum(wspeed)/len(wspeed)
        totalPrecip = sum(precip)
        avgWDir= sum(wdir)/len(wdir)
        avgHum = sum(hum)/len(hum)
        return np.array( [max(temp), avgWSpeed, avgWDir, totalPrecip, avgHum])
