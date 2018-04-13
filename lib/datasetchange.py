import numpy as np
import cv2
from .datamodule import PIXEL_SIZE

class Dataset(object):

    def __init__(self, data, whichBurns=None, whichDays=None, whichPixels=None):
        self.data = data
        self.whichBurns = whichBurns   if whichBurns  is not None else 'all'
        self.whichDays = whichDays     if whichDays   is not None else 'all'
        self.whichPixels = whichPixels if whichPixels is not None else 'all'

    def getData(self):
        return self.data

    def getPixels(self):
        return self.pixels

    def getBurns(self):
        return self.burns

    def getDays(self):
        return self.days

    def getLayers(self):
        return self.layers

    def findVulnerablePixels(self, startingPerim, radius=None):
        if radius is None:
            radius = self.VULNERABLE_RADIUS
        '''Return the indices of the pixels that close to the current fire perimeter'''
        kernel = np.ones((3,3))
        its = int(round((2*(radius/PIXEL_SIZE)**2)**.5))
        dilated = cv2.dilate(startingPerim, kernel, iterations=its)
        border = dilated - startingPerim
        return np.where(border)
