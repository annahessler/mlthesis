import numpy as np
import cv2
# import matplotlib.pyplot as plt

VULNERABLE_RADIUS = 300
PIXEL_SIZE = 30

class Data(object):
    '''Contains an layered image of input data and an image of output data

    Some layers are actually used by the NN, others are just internal layers used to calculate other layers.
    Only some of the pixels within the dataset are actually used by the NN (eg ignore pixels way far away from the fire).
    '''
    def __init__(self):
        self.shape = None

        self.layers = {}
        self.data = {}
        self.output = None

        self.usedVariables = []
        self.datasets = {}

    def setUsedPixels(self, indices):
        '''Select which of the pixels we will use as inputs to the model'''
        assert len(indices) == 2
        assert type(indices[0]) == type(np.ndarray([])) and type(indices[1]) == type(np.ndarray([]))
        assert len(indices[0].shape) == len(indices[1].shape) == 1
        assert indices[0].size == indices[1].size
        self.usedPixels = indices

    def addLayer(self, layerName, data, use=True):
        '''Add the actual layer of data(either spatial 2D or 1D weather/spatial-invariant) to our set'''
        assert len(data.shape) == 2
        if self.shape is not None:
            if self.shape != data.shape:
                raise ValueError("All spatial data must be the same dimension.")
        else:
            self.shape = data.shape
        self.layers[layerName] = data
        if use and layerName not in self.usedVariables:
            self.usedVariables.append(layerName)

    def addData(self, variableName, data, use=True):
        self.data[variableName] = data
        if use and variableName not in self.usedVariables:
            self.usedVariables.append(variableName)

    def addOutput(self, data):
        self.output = data

    def addInput(self, variableName):
        '''Select a layer to be used as an input variable to the model'''
        if variableName not in self.data or variableName not in self.layers:
            raise ValueError("{} doesn't exist in data or layers".format(variableName))
        if variableName not in self.usedVariables:
            self.usedVariables.append(variableName)

    def stackLayers(self, layerNames):
        layers = [self.layers[name] for name in layerNames]
        stacked = np.dstack(tuple(layers))
        return stacked

    def getAOIs(self, layerNames, indices, radius=15):
        stacked = self.stackLayers(layerNames)
        aois = []
        r = radius
        for y,x in zip(*indices):
            aois.append(stacked[y-r:y+r+1,x-r:x+r+1,:])
        arr = np.array(aois)

        # we need the tensor to have dimensions (nsamples, nchannels, AOIwidth,AOIheight)
        # it starts as (nsamples, AOIwidth,AOIheight, nchannels)
        swapped = np.swapaxes(arr,1,3)
        swapped = np.swapaxes(swapped,2,3)
        return swapped

    def findVulnerablePixels(self, radius=VULNERABLE_RADIUS):
        '''Return the indices of the pixels that close to the current fire perimeter'''
        startingPerim = self.layers['perim']
        kernel = np.ones((3,3))
        its = int(round((2*(radius/PIXEL_SIZE)**2)**.5))
        dilated = cv2.dilate(startingPerim, kernel, iterations=its)
        border = dilated - startingPerim
        return np.where(border)

    def chooseDatasets(self, ratio=.75, shuffle=True):
        startingPerim = self.layers['perim']
        xs,ys = findVulnerableIndices(self)
        if shuffle:
            p = np.random.permutation(len(xs))
            xs,ys = xs[p], ys[p]
        # now split them into train and test sets
        splitIndex = si = int(ratio*len(xs))
        train = (xs[:si], ys[:si])
        test  = (xs[si:], ys[si:])
        self.datasets['train'] = train
        self.datasets['test']  = test

    def getOutput(self):
        '''Return a 1D array of output values, ready to be used by the model'''
        if self.output is None:
            raise ValueError("Output data must be set.")
        if self.usedPixels is None:
            return self.output
        else:
            return self.output[self.usedPixels]

    @staticmethod
    def defaultDataset(dateString):
        d = Dataset()

        dem = cv2.imread('data/raw/dem.tif', cv2.IMREAD_UNCHANGED)
        slope = cv2.imread('data/raw/slope.tif',cv2.IMREAD_UNCHANGED)
        landsat = cv2.imread('data/raw/landsat.png', cv2.IMREAD_UNCHANGED)
        ndvi = cv2.imread('data/raw/NDVI_1.tif', cv2.IMREAD_UNCHANGED)
        d.addLayer('dem', dem)
        d.addLayer('slope', slope)
        for name, band in zip(['r','g','b','nir'], cv2.split(landsat)):
            d.addLayer(name,band)
        d.addLayer('ndvi', ndvi)

        perim = openStartingPerim(dateString)
        d.addLayer('perim', perim)

        weatherData = createWeatherMetrics(openWeatherData(dateString))
        for name, data in zip(['maxTemp', 'avgWSpeed', 'avgWDir', 'precip', 'avgHum'], weatherData):
            d.addData(name, data)

        output = openEndingPerim(dateString)
        d.addOutput(output)

        return d

    def __repr__(self):
        res = "Dataset("
        res += "layers:" + repr(self.layers.keys())
        res += ", data:" + repr(self.data.keys())
        return res


def openLandData():
    dem = cv2.imread('data/raw/dem.tif', cv2.IMREAD_UNCHANGED)
    slope = cv2.imread('data/raw/slope.tif',cv2.IMREAD_UNCHANGED)
    landsat = cv2.imread('data/raw/landsat.png', cv2.IMREAD_UNCHANGED)
    ndvi = cv2.imread('data/raw/NDVI_1.tif', cv2.IMREAD_UNCHANGED)
    return np.dstack((dem, slope, landsat, ndvi))

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

def createDerivedData(rawData):
    startingPerim = rawData[:,:,0].astype(np.uint8)
    distance = findDistance(startingPerim)
    return np.dstack((distance,))

