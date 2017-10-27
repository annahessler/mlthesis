import numpy as np
import cv2
# import matplotlib.pyplot as plt

VULNERABLE_RADIUS = 300
PIXEL_SIZE = 30

class Dataset(np.ndarray):

    # def __init__(self, data):
    #     super().__init__(data)

    def getX(self):
        return self[:,:]

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

def findVulnerablePixels(startingPerim, radius=VULNERABLE_RADIUS):
    '''Return the indices of the pixels that close to the current fire perimeter'''
    kernel = np.ones((3,3))
    its = int(round((2*(radius/PIXEL_SIZE)**2)**.5))
    dilated = cv2.dilate(startingPerim, kernel, iterations=its)
    border = dilated - startingPerim
    return np.where(border)

def createWeatherMetrics(weatherData):
    temp, dewpt, temp2, wdir, wspeed, precip, hum = weatherData
    avgWSpeed = sum(wspeed)/len(wspeed)
    totalPrecip = sum(precip)
    avgWDir= sum(wdir)/len(wdir)
    avgHum = sum(hum)/len(hum)
    return np.array( [max(temp), avgWSpeed, avgWDir, totalPrecip, avgHum])

def createRawData(dateString):
    # create inputs
    startingPerim = openStartingPerim(dateString)
    landData = openLandData()
    weatherData = createWeatherMetrics(openWeatherData(dateString))
    h,w = landData.shape[:2]
    tiledWeather = np.tile(weatherData,(h,w,1))

    return np.dstack((startingPerim, landData, tiledWeather))

def createData(dateString):
    # open all of our raw data, read directly from data files
    raw = createRawData(dateString)
    # there are some secondary inputs to model, such as distance to fire, etc
    derived = createDerivedData(raw)
    output = openEndingPerim(dateString)
    return np.dstack((raw, derived, output))

def createDerivedData(rawData):
    startingPerim = rawData[:,:,0].astype(np.uint8)
    distance = findDistance(startingPerim)
    return np.dstack((distance,))

def findDistance(startingPerim):
    inverted = 255-startingPerim
    dist = cv2.distanceTransform(inverted, cv2.DIST_L2, 5)
    return dist

def chooseDatasets(data, ratio=.75, shuffle=True):
    startingPerim = data[:,:,0]
    vulnerableIndices = findVulnerablePixels(startingPerim)
    if shuffle:
        xs,ys = vulnerableIndices
        p = np.random.permutation(len(xs))
        xs,ys = xs[p], ys[p]
    # now split them into train and test sets
    splitIndex = si = int(ratio*len(xs))
    train = (xs[:si], ys[:si])
    test  = (xs[si:], ys[si:])
    return train, test

def saveData(data, dateString):
    np.savetxt('data/forModel/'+ dateString+'.csv', list(data.reshape(-1, data.shape[-1])), delimiter=',')

date = '0731'
data = createData(date)
train, test = chooseDatasets(data)

# print(train)
saveData(data[train], 'train')
saveData(data[test], 'test')

# print(data[train])
