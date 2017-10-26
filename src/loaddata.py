import os
import numpy as np
import cv2
import matplotlib.pyplot as plt

VULNERABLE_RADIUS = 300
os.chdir('../data')

def openLandData():
    dem = cv2.imread('raw/dem.tif', cv2.IMREAD_UNCHANGED)
    slope = cv2.imread('raw/slope.tif',cv2.IMREAD_UNCHANGED)
    landsat = cv2.imread('raw/landsat.png', cv2.IMREAD_UNCHANGED)
    return np.dstack((dem, slope, landsat))

def openStartingPerim(dateString):
    perimFileName = 'raw/perims/' + dateString + '.tif'
    perim = cv2.imread(perimFileName, cv2.IMREAD_UNCHANGED)*255
    return perim

def openEndingPerim(dateString):
    '''Get the fire perimeter on the next day'''
    month, day = dateString[:2], dateString[2:]
    nextDay = str(int(day)+1).zfill(2)
    guess = month+nextDay
    perimFileName = 'raw/perims/' + guess + '.tif'
    # print(perimFileName)
    perim = cv2.imread(perimFileName, cv2.IMREAD_UNCHANGED)
    if perim is None:
        # overflowed the month, that file didnt exist
        nextMonth = str(int(month)+1).zfill(2)
        guess = nextMonth+'01'
        perimFileName = 'raw/perims/' + guess + '.tif'
        # print(perimFileName)
        perim = cv2.imread(perimFileName, cv2.IMREAD_UNCHANGED)
        if perim is None:
            raise RuntimeError('Could not find a perimeter for the day after ' + dateString)
    return perim

def findVulnerablePixels(startingPerim, radius=VULNERABLE_RADIUS):
    '''Return the indices of the pixels that are candidates for our testing'''
    kernel = np.ones((3,3))
    its = int(round((2*(radius/30)**2)**.5))
    dilated = cv2.dilate(startingPerim, kernel, iterations=its)
    border = dilated - startingPerim
    # cv2.imshow('start',startingPerim)
    # cv2.imshow('dil',dilated)
    # cv2.imshow('border', border)
    # cv2.waitKey(0)
    return np.where(border)

def openWeatherData(dateString):
    fname = 'raw/weather/' + dateString + '.csv'
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

def createData(dateString):
    # create inputs
    startingPerim = openStartingPerim(dateString)
    landData = openLandData()
    distance = findDistance(startingPerim)

    weatherData = createWeatherMetrics(openWeatherData(dateString))
    h,w = landData.shape[:2]
    tiledWeather = np.tile(weatherData,(h,w,1))

    # combine inputs, then combine with outputs
    inputs = np.dstack((startingPerim, distance, landData, tiledWeather))
    output = openEndingPerim(dateString)
    return np.dstack((inputs, output))

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

print(data)
saveData(data, '0731')

# print(data[train])