from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import os
import cv2
import numpy as np
from lib import image

def openWeatherData(dateString, fireName):
    fname = 'data/raw/' + fireName + '/weather/' + dateString + '.csv'
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

datagen = ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')

def collectData(fireName, date):
    dem = cv2.imread('data/raw/' + fireName + '/dem.tif', cv2.IMREAD_UNCHANGED)
    print('dem is all good', dem)
    print('directory is ', os.listdir)
    # np.savetxt('data/raw/dembefore.csv', dem, delimiter=',')
    aspect = cv2.imread('data/raw/'+ fireName + '/aspect.tif', cv2.IMREAD_UNCHANGED) 
    perim = cv2.imread('data/raw/' + fireName + '/perims/' + date + '.tif', cv2.IMREAD_UNCHANGED)
    perim_next = cv2.imread('data/raw' + fireName + '/perims/' + '0801')
    weather = createWeatherMetrics(openWeatherData(date, fireName))
    toAugment = np.dstack((dem, aspect, perim))
    print('weather shape', weather.shape)
    print('toaugment shape ', toAugment.shape)
    tiledWeather = np.tile(weather, (toAugment.shape))
    print(tiledWeather.shape)
    toAugment = np.dstack((toAugment, tiledWeather))
    # toAugment = toAugment.reshape((1,) + toAugment.shape )
    print('x shape is ', toAugment.shape)
    return toAugment




x = collectData('riceRidge', '0731')
oidg = image.ourImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest',
        data_format = 'channels_last'
    )
x = np.lib.pad(x, ((1,1), (1,1), (0,0)) 'constant')

augmented = oidg.random_transform(x, 7)
print(augmented.shape)
np.savetxt('data/raw/demafterreturn.csv', augmented[:,:,0], delimiter=',')

before = x[:,:,0]
result = augmented[:,:,0]
print('before shape is ', before.shape)
print('result shape is ', result.shape)

cv2.imwrite('before.png', before.reshape(before.shape[:2]))
cv2.imwrite('after.png', result.reshape(result.shape[:2]))

# the .flow() command below generates batches of randomly transformed images
# and saves the results to the `preview/` directory
# i = 0
# for batch in datagen.flow(augmented, batch_size=10,
#                           save_to_dir='data/raw/riceRidgeAugmented', save_prefix='test', save_format='tif'):
#     i += 1
#     if i > 25:
#         break  # otherwise the generator would loop indefinitely
