# preprocess.py
from collections import namedtuple
import numpy as np
# import scipy.stats as stats
import scipy.stats
import lib.stats as stats
try:
    import matplotlib.pyplot as plt
except:
    pass

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
        # print('for the Day', burnName, date)
        # print('weather matrix is', wm)
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
    normed = normalize(arr, axis=0)
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
    # normalizeLayers(result)
    return result

def normalizeLayers(layers):
    result = {}
    for name, data in layers.items():
        # if name != 'ndvi':
        #     continue
        if name == 'dem':
            result[name] = normalizeElevations(data)
        else:
            # print('normalizing layer', name)
            result[name] = normalizeNonElevations(data)
    return result

def normalizeElevations(dems):
    avgElevation = {}
    validIndicesDict = {}
    ranges = {}
    for burnName, dem in dems.items():
        validIndices = validPixelIndices(dem)
        validIndicesDict[burnName] = validIndices
        validPixels = dem[validIndices]
        avgElevation[burnName] = np.mean(validPixels)
        ranges[burnName] = validPixels.max()-validPixels.min()

        # vis = dem.copy()
        # vis[validIndices] = 42
        # plt.imshow(dem)
        # plt.figure("valid")
        # plt.imshow(vis)
        # plt.show()
    maxRange = max(ranges.values())
    results = {}
    for burnName, dem in dems.items():
        validIndices = validIndicesDict[burnName]
        # print(validIndices)
        validPixels = dem[validIndices]
        # print('valid pixels size:', validPixels.size)
        normed = normalize(validPixels)
        blank = np.zeros_like(dem, dtype=np.float32)
        thisRange = ranges[burnName]
        scaleFactor = thisRange/maxRange
        blank[validIndices] = scaleFactor * normed
        # print('scaleFactor is ', scaleFactor)
        # plt.imshow(blank)
        # plt.show()
        results[burnName] = blank
    return results

def normalizeNonElevations(nonDems):
    splitIndices = [0]
    validPixelsList = []
    validIndicesList = []
    names = list(nonDems.keys())
    for name in names:
        layer = nonDems[name]
        validIndices = validPixelIndices(layer)
        validPixels = layer[validIndices]

        # print('valid indices:', validIndices)
        # print('len of valid pixels', validPixels.shape, len(validPixels))

        # vis = np.zeros_like(layer)
        # vis[validIndices] = 42
        # plt.figure('before '+name)
        # plt.imshow(layer)
        # plt.figure('valid '+name)
        # plt.imshow(vis)
        # plt.show()

        validPixelsList += validPixels.tolist()
        splitIndices.append(splitIndices[-1] + len(validPixels))
        validIndicesList.append(validIndices)

    # now layers.shape is (nburns, height, width)
    arr = np.array(validPixelsList)
    # print('array is', arr, arr.min(), arr.max())
    normed = normalize(arr)
    # print(normed)
    # print('split indices', splitIndices)
    splitIndices = splitIndices[1:]
    # print('split indices', splitIndices)
    splitBackUp = np.split(normed, splitIndices)
    # print('split back up:', splitBackUp.shape)
    results = {}
    for name, validIndices, normedPixels in zip(names,validIndicesList,splitBackUp):
        # print(name, validIndices, normedPixels)
        src = nonDems[name]
        img = np.zeros_like(src, dtype=np.float32)
        img[validIndices] = normedPixels
        results[name] = img
        # if normedPixels.size > 0:
        #     print('min and max of ', name, normedPixels.min(), normedPixels.max())
        # plt.figure('normed'+name)
        # plt.imshow(img)
    # plt.show()
    return results

def validPixelIndices(layer):
    '''return the locations of all of the pixels which are not NODATA.

    if there are any realllllly big or small pixels, use those. Otherwise, they are big, but not that big.

    Then, the NODATA pixels are some really large or small value and there are lots of them contiguous.
    First rely on the fact that they will be really common. If there is one value that covers >20% of the img,
    use those pixels.

    Sometimes though there aren't many NODATA pixels. Then we rely on the fact that the small number of them have really large
    or small values, far away from the mean'''
    HUGE = 1e10
    reasonable = np.where(np.absolute(layer) < HUGE)
    if len(reasonable[0]) < layer.size:
        return reasonable
    # get the modes and counts for the entire image
    m, count = stats.mode(layer.flatten(), axis=0)
    # get the most common entry
    # if that pixel is really common (>20% of the img)
    CUTOFF_PERCENT = .2
    if count > CUTOFF_PERCENT * layer.size:
        # print('using mode method for valid pixels')
        return np.where(layer!=m)
    # print('using deviance method for valid pixels')
    trimMean = scipy.stats.trim_mean(layer, .1, axis=None)
    dev = np.std(layer)
    # print('means and std:', np.mean(layer), trimMean, dev)
    lowDeviance = abs(layer-trimMean) < 3*dev
    return np.where(lowDeviance)

def normalizeLayers2(layerDict):
    ids = list(layerDict.keys())
    layerSets = np.array( [layerDict[i] for i in ids] )
    print(layerSets.shape)
    nsamples, h, w, nlayers = layerSets.shape
    # flatten everything except for the layers
    channels = layerSets.reshape((nsamples*h*w, nlayers))
    # normalize everything along the channels
    normed = normalize(channels, axis=0)
    # back to original shape
    result = normed.reshape(layerSets.shape)
    print(result[0,:,:,0], result[:,:,:,0].min(), result[:,:,:,0].max())
    backToDict = {i:aoi for i, aoi in zip(ids, result)}
    print(backToDict)
    # channels = np.split(layerSets, nlayers, axis=-1)
    # print(channels[0].shape)
    # channels = [c[:3000] for c in channels]
    # channels = channels[:100]
    # npixels = h*w
    # oneDimChannels = [arr.reshape((nsamples, npixels)) for arr in channels]
    # print(channels, len(channels))
    # print(channels[0].shape)
    # normed = [normalize(np.squeeze(c)) for c in channels]
    # print(normed[0].shape)

    # reuse the same array to avoid allocating an entirely new onexz
    # layerSets[:1000] = np.stack(normed, axis=-1)
    # print(layerSets.shape)
    # return layerSets

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
    # print(stacked.shape, padded.shape)s
    return aoi

# =================================================================
# utility functions

def normalize(arr, axis=None):
    '''Rescale an array so that it varies from 0-1.

    if axis=0, then each column is normalized independently
    if axis=1, then each row is normalized independently'''
    # print('subtracting min')
    res = arr - arr.min(axis=axis)
    # print('dividing where', res)
    res = res.astype(np.float32)
    # where dividing by zero, just use zero
    res = np.divide(res, res.max(axis=axis), out=np.zeros_like(res), where=res!=0)
    # print('done')
    return res

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
        # print(wdir[i], wDirRad[i], wspeed[i])
        if wdir[hr] > 90 and wdir[hr] < 270: #from south
            # print('south!', -np.cos(wDirRad[i]) * wspeed[i])
            s += abs(np.cos(wDirRad[hr]) * wspeed[hr])
        if wdir[hr] < 90 or wdir[hr] > 270: #from north
            # print('north!', np.cos(wDirRad[i]) * wspeed[i])
            n += abs(np.cos(wDirRad[hr]) * wspeed[hr])
        if wdir[hr] < 360 and wdir[hr] > 180: #from west
            # print('west!', -np.sin(wDirRad[i]) * wspeed[i])
            w += abs(np.sin(wDirRad[hr]) * wspeed[hr])
        if wdir[hr] > 0 and wdir[hr] < 180: #from east
            # print('east!',np.sin(wDirRad[i]) * wspeed[i])
            e += abs(np.sin(wDirRad[hr]) * wspeed[hr])
    components = [n, s, e, w]
    # print(weather)
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
