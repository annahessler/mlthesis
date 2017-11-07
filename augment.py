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

def collectData(fireName, date, next_day):
    dem = cv2.imread('data/raw/' + fireName + '/dem.tif', cv2.IMREAD_UNCHANGED)
    print('DEM SHAPE: ', dem.shape)
    print('dem is all good', dem)
    # print('directory is ', os.listdir())
    np.savetxt('data/raw/dembefore.csv', dem, delimiter=',')
    aspect = cv2.imread('data/raw/'+ fireName + '/aspect.tif', cv2.IMREAD_UNCHANGED)
    print('ASPECT SHAPE: ', aspect.shape)
    print(fireName, date)
    perim = cv2.imread('data/raw/'+fireName+'/perims/'+date+'.tif', 0)
    print('PERIM SHAPE: ', perim)
    perim_next = cv2.imread('data/raw' + fireName + '/perims/' + next_day + '.tif')
    weather = createWeatherMetrics(openWeatherData(date, fireName))
    print('SHAPES: ', dem.shape, aspect.shape, perim.shape)
    toAugment = np.dstack((dem, aspect, perim))
    print('weather shape', weather.shape)
    print('toaugment shape ', toAugment.shape)
    tiledWeather = np.tile(weather, (toAugment.shape))
    print(tiledWeather.shape)
    toAugment = np.dstack((toAugment, tiledWeather))
    # toAugment = toAugment.reshape((1,) + toAugment.shape )
    print('x shape is ', toAugment.shape)
    return toAugment

def doMore(x, fire, date):
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

    x1 = np.lib.pad(x, ((1,1),(1,1),(0,0)), 'constant')
    augmented = oidg.random_transform(x1, 7)
    print(augmented.shape)
    np.savetxt('data/raw/demafterreturn.csv', augmented[:,:,0], delimiter=',')

    before = x[:,:,0]
    result = augmented[:,:,0]
    print('before shape is ', before.shape)
    print('result shape is ', result.shape)

    cv2.imwrite('before'+ fire+ date+ '.png', before.reshape(before.shape[:2]))
    cv2.imwrite('after'+ fire+ date+ '.png', result.reshape(result.shape[:2]))

fires = ['riceRidge', 'coldSprings', 'beaverCreek']
rrdays = ['0731', '0801', '0802', '0803']
csdays = ['0711', '0712', '0713', '0714', '0715']
bcdays = ['0629', '0630']
bcdays2 = ['0711', '0712', '0713', '0714', '0715', '0716']
bcdays3 = ['0801', '0802']
bcdays4 = ['0804', '0805']
bcdays5 = ['0807', '0808', '0809', '0810']

for i in fires:
    if i == fires[0]:
        for r, value in enumerate(rrdays[:-1], 0):
            x = collectData(i, rrdays[r], rrdays[r+1])
            doMore(x, i, rrdays[r])
    if i == fires[1]:
        for c, value in enumerate(csdays[:-1], 0):
            x = collectData(i, csdays[c], csdays[c+1])
            doMore(x, i, csdays[c])
    if i == fires[2]:
        for b, value in enumerate(bcdays[:-1], 0):
            x = collectData(i, bcdays[b], bcdays[b+1])
            doMore(x, i, bcdays[b])
        for b, value in enumerate(bcdays2[:-1], 0):
            x = collectData(i, bcdays2[b], bcdays2[b+1])
            doMore(x, i, bcdays2[b])
        for b, value in enumerate(bcdays3[:-1], 0):
            x = collectData(i, bcdays3[b], bcdays3[b+1])
            doMore(x, i, bcdays3[b])
        for b, value in enumerate(bcdays4[:-1], 0):
            x = collectData(i, bcdays4[b], bcdays4[b+1])
            doMore(x, i, bcdays4[b])
        for b, value in enumerate(bcdays5[:-1], 0):
            x = collectData(i, bcdays5[b], bcdays5[b+1])
            doMore(x, i, bcdays5[b])



# x = collectData('riceRidge', '0731')



# the .flow() command below generates batches of randomly transformed images
# and saves the results to the `preview/` directory
# i = 0
# for batch in datagen.flow(augmented, batch_size=10,
#                           save_to_dir='data/raw/riceRidgeAugmented', save_prefix='test', save_format='tif'):
#     i += 1
#     if i > 25:
#         break  # otherwise the generator would loop indefinitely
