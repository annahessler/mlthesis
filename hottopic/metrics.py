#metrics.py
import numpy as np

class WeatherMetric(object):

    numOutputs = 5

    @staticmethod
    def calculate(weatherMatrix):
        temp, dewpt, temp2, wdir, wspeed, precip, hum = weatherMatrix
        avgWSpeed = sum(wspeed)/len(wspeed)
        totalPrecip = sum(precip)
        avgWDir= sum(wdir)/len(wdir)
        avgHum = sum(hum)/len(hum)
        return np.array( [max(temp), avgWSpeed, avgWDir, totalPrecip, avgHum])
