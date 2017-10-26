import numpy as np
import cv2
import matplotlib.pyplot as plt

def openLandData():
    dem = cv2.imread('data/dem.tif', cv2.IMREAD_UNCHANGED)
    slope = cv2.imread('data/slope.tif',cv2.IMREAD_UNCHANGED)
    landsat = cv2.imread('data/landsat.png', cv2.IMREAD_UNCHANGED)
    return np.dstack((dem, slope, landsat))

def openStartingPerim(dateString):
    perimFileName = 'data/perims/' + dateString + '.tif'
    print(perimFileName)
    perim = cv2.imread(perimFileName, cv2.IMREAD_UNCHANGED)*255
    return perim

def openEndingPerim(dateString):
    '''Get the fire perimeter on the next day'''
    month, day = dateString[:2], dateString[2:]
    nextDay = str(int(day)+1).zfill(2)
    guess = month+nextDay
    perimFileName = 'data/perims/' + guess + '.tif'
    # print(perimFileName)
    perim = cv2.imread(perimFileName, cv2.IMREAD_UNCHANGED)
    if perim is None:
        # overflowed the month, that file didnt exist
        nextMonth = str(int(month)+1).zfill(2)
        guess = nextMonth+'01'
        perimFileName = 'data/perims/' + guess + '.tif'
        # print(perimFileName)
        perim = cv2.imread(perimFileName, cv2.IMREAD_UNCHANGED)
        if perim is None:
            raise RuntimeError('Could not find a perimeter for the day after ' + dateString)
    return perim

def choosePixels(startingPerim, radius):
    '''Return the indices of the pixels that are candidates for our testing'''
    print(startingPerim)
    kernel = np.ones((3,3))
    its = int(round((2*(radius/30)**2)**.5))
    print(its)
    dilated = cv2.dilate(startingPerim, kernel, iterations=its)
    border = dilated - startingPerim
    # cv2.imshow('start',startingPerim)
    # cv2.imshow('dil',dilated)
    # cv2.imshow('border', border)
    # cv2.waitKey(0)
    return np.where(border)

def openWeatherData(dateString):
    fname = './data/weather/' + dateString + '.csv'
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
    landData = openLandData()
    endingPerim = openEndingPerim(dateString)
    combined = np.dstack((landData, endingPerim))

    startingPerim = openStartingPerim(dateString)
    # in meters
    radius = 500
    testPixels = combined[choosePixels(startingPerim, radius)]

    # print(testPixels, combined.size, testPixels.size)

    weatherData = createWeatherMetrics(openWeatherData(dateString))
    tiledWeather = np.tile(weatherData,(testPixels.shape[0],1))
    # print(tiledWeather)
    # print(tiledWeather.shape, testPixels.shape)

    # there 18000 pixels of data, each with 11 inputs and one output
    everything = np.hstack((tiledWeather, testPixels))
    # print(everything, everything.shape)

    toPrint = list(everything)
    np.savetxt(dateString + '.csv', toPrint, delimiter = ',')

    return everything



print(createData('0731'))
