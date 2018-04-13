# preprocess.py
from collections import namedtuple
import numpy as np
try:
    import matplotlib.pyplot as plt
except:
    pass

try:
    from lib import util
except:
    import util
class PreProcessor(object):
    '''What is responsible for extracting the used data from the dataset and then
    normalizing or doing any other steps before feeding it into the network.'''

    def __init__(self, numWeatherInputs, whichLayers, AOIRadius):
        self.numWeatherInputs = numWeatherInputs
        self.whichLayers = whichLayers
        self.AOIRadius = AOIRadius

    def process(self, dataset):
        '''Take a dataset and return the extracted inputs and outputs'''
        # create dictionaries mapping from Point to actual data from that Point
        metrics = calculateWeatherMetrics(dataset)
        oneMetric = list(metrics.values())[0]
        assert len(oneMetric) == self.numWeatherInputs, "Your weather metric function must return the expected number of metrics"
        aois = getSpatialData(dataset, self.whichLayers, self.AOIRadius)
        outs = getOutputs(dataset)

        # convert the dictionaries into lists, then arrays
        w, i, o = [], [], []
        ptList = dataset.toList(dataset.points)
        for pt in ptList:
            burnName, date, location = pt
            w.append(metrics[burnName, date])
            i.append(aois[burnName, date, location])
            o.append(outs[burnName, date, location])
        weatherInputs = np.array(w)
        imgInputs = np.array(i)
        outputs = np.array(o)

        return ([weatherInputs, imgInputs], outputs), ptList

def calculateWeatherMetrics(dataset):
    '''Return a dictionary mapping from (burnName, date) id's to a dictionary of named weather metrics.'''
    metrics = {}
    for burnName, date in dataset.getUsedBurnNamesAndDates():
        wm = dataset.data.getWeather(burnName, date)
        precip = totalPrecipitation(wm)
        temp = maximumTemperature1(wm)
        temp2 = maximumTemperature2(wm)
        hum = averageHumidity(wm)
        winds = windMetrics(wm)
        entry = [precip, temp, temp2, hum] + winds
        metrics[(burnName, date)] = entry
    # now normalize all of them
    # ensure we keep order
    ids = list(metrics.keys())
    arr = np.array( [metrics[i] for i in ids] )
    normed = util.normalize(arr, axis=0)
    metrics = {i:nums for (i,nums) in zip(ids, normed)}
    return metrics

def getSpatialData(dataset, whichLayers, AOIRadius):
    # for each channel in the dataset, get all of the used data
    layers = {layerName:dataset.getAllLayers(layerName) for layerName in whichLayers}
    # now normalize them
    layers = normalizeLayers(layers)
    # now order them in the whichLayers order, stack them, and pad them
    paddedLayers = stackAndPad(layers, whichLayers, dataset, AOIRadius)
    # now extract out the aois around each point
    result = {}
    for pt in dataset.toList(dataset.points):
        burnName, date, location = pt
        padded = paddedLayers[(burnName, date)]
        aoi = extract(padded, location, AOIRadius)
        result[(burnName, date, location)] = aoi
    return result

def normalizeLayers(layers):
    result = {}
    for name, data in layers.items():
        if name == 'dem':
            result[name] = normalizeElevations(data)
        else:
            result[name] = normalizeNonElevations(data)
    return result

def normalizeElevations(dems):
    avgElevation = {}
    validIndicesDict = {}
    ranges = {}
    for burnName, dem in dems.items():
        validIndices = np.where(np.isfinite(dem))
        validIndicesDict[burnName] = validIndices
        validPixels = dem[validIndices]
        avgElevation[burnName] = np.mean(validPixels)
        ranges[burnName] = validPixels.max()-validPixels.min()

    maxRange = max(ranges.values())
    results = {}
    for burnName, dem in dems.items():
        validIndices = validIndicesDict[burnName]
        validPixels = dem[validIndices]
        normed = util.normalize(validPixels)
        blank = np.zeros_like(dem, dtype=np.float32)
        thisRange = ranges[burnName]
        scaleFactor = thisRange/maxRange
        blank[validIndices] = scaleFactor * normed
        results[burnName] = blank
    return results

def normalizeNonElevations(nonDems):
    splitIndices = [0]
    validPixelsList = []
    validIndicesList = []
    names = list(nonDems.keys())
    for name in names:
        layer = nonDems[name]
        validIndices = np.where(np.isfinite(layer))
        validPixels = layer[validIndices]

        validPixelsList += validPixels.tolist()
        splitIndices.append(splitIndices[-1] + len(validPixels))
        validIndicesList.append(validIndices)

    # now layers.shape is (nburns, height, width)
    arr = np.array(validPixelsList)
    normed = util.normalize(arr)
    splitIndices = splitIndices[1:]
    splitBackUp = np.split(normed, splitIndices)
    results = {}
    for name, validIndices, normedPixels in zip(names,validIndicesList,splitBackUp):
        src = nonDems[name]
        img = np.zeros_like(src, dtype=np.float32)
        img[validIndices] = normedPixels
        results[name] = img
    return results

def getOutputs(dataset):
    result = {}
    for pt in dataset.toList(dataset.points):
        burnName, date, location = pt
        out = dataset.data.getOutput(burnName, date, location)
        result[(burnName, date, location)] = out
    return result

def stackAndPad(layerDict, whichLayers, dataset, AOIRadius):
    result = {}
    for burnName, date in dataset.getUsedBurnNamesAndDates():
        # guarantee that the perim mask is just 0s and 1s
        day = dataset.data.burns[burnName].days[date]
        sp = day.startingPerim
        sp[sp!=0]=1

        layers = [layerDict[layerName][burnName] for layerName in whichLayers]
        layers = [sp] + layers
        stacked = np.dstack(layers)
        r = AOIRadius
        # pad with zeros around border of image
        padded = np.lib.pad(stacked, ((r,r),(r,r),(0,0)), 'constant')
        result[(burnName, date)] = padded
    return result

def extract(padded, location, AOIRadius):
    '''Assume padded is bordered by radius self.inputSettings.AOIRadius'''
    y,x = location
    r = AOIRadius
    lox = r+(x-r)
    hix = r+(x+r+1)
    loy = r+(y-r)
    hiy = r+(y+r+1)
    aoi = padded[loy:hiy,lox:hix]
    return aoi

# =================================================================
# utility functions

def totalPrecipitation(weatherMatrix):
    temp, dewpt, temp2, wdir, wspeed, precip, hum = weatherMatrix
    return sum(precip)

def averageHumidity(weatherMatrix):
    temp, dewpt, temp2, wdir, wspeed, precip, hum = weatherMatrix
    return sum(hum)/len(hum)

def maximumTemperature1(weatherMatrix):
    temp, dewpt, temp2, wdir, wspeed, precip, hum = weatherMatrix
    return max(temp)

def maximumTemperature2(weatherMatrix):
    temp, dewpt, temp2, wdir, wspeed, precip, hum = weatherMatrix
    return max(temp2)

def windMetrics(weatherMatrix):
    temp, dewpt, temp2, wdir, wspeed, precip, hum = weatherMatrix
    wDirRad = [(np.pi/180) * wDirDeg for wDirDeg in wdir]
    n, s, e, w = 0, 0, 0, 0
    for hr in range(len(wdir)):
        if wdir[hr] > 90 and wdir[hr] < 270: #from south
            s += abs(np.cos(wDirRad[hr]) * wspeed[hr])
        if wdir[hr] < 90 or wdir[hr] > 270: #from north
            n += abs(np.cos(wDirRad[hr]) * wspeed[hr])
        if wdir[hr] < 360 and wdir[hr] > 180: #from west
            w += abs(np.sin(wDirRad[hr]) * wspeed[hr])
        if wdir[hr] > 0 and wdir[hr] < 180: #from east
            e += abs(np.sin(wDirRad[hr]) * wspeed[hr])
    components = [n, s, e, w]
    return components

# =========================================================

if __name__ == '__main__':
    import rawdata
    import dataset
    data = rawdata.RawData.load()
    ds = dataset.Dataset(data, points=dataset.Dataset.vulnerablePixels)
    pp = PreProcessor(8, ['dem', 'ndvi', 'g'], 30)
    (inp, out), ptList = pp.process(ds)
    weather, img = inp
    print(weather[0])
    print(img[0])
    print(out[0])
