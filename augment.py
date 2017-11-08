from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import os
import cv2
import numpy as np
from lib import image
import time
from time import localtime, strftime



def openEndingPerim(dateString, fire):
    month, day = dateString[:2], dateString[2:]
    nextDay = str(int(day)+1).zfill(2)
    guess = month+nextDay
    perimFileName = 'data/raw/'+ fire + '/perims/' + guess + '.tif'
    perim = cv2.imread(perimFileName, cv2.IMREAD_UNCHANGED)
    if perim is None:
        # overflowed the month, that file didnt exist
        nextMonth = str(int(month)+1).zfill(2)
        guess = nextMonth+'01'
        perimFileName = 'data/raw/' + fire + '/perims/' + guess + '.tif'
        perim = cv2.imread(perimFileName, cv2.IMREAD_UNCHANGED)
        if perim is None:
            raise RuntimeError('Could not find a perimeter for the day after ' + dateString)
        return perim

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
        rotation_range=300,
        width_shift_range=0.2,
        height_shift_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')

def collectData(fireName, days):
    days_arr = []
    # still need landsat to be read in
    dem = cv2.imread('data/raw/' + fireName + '/dem.tif', cv2.IMREAD_UNCHANGED)
    print('DEM SHAPE: ', dem.shape)
    # print('directory is ', os.listdir())
    # np.savetxt('data/raw/dembefore.csv', dem, delimiter=',')
    aspect = cv2.imread('data/raw/'+ fireName + '/aspect.tif', cv2.IMREAD_UNCHANGED)
    print('ASPECT SHAPE: ', aspect.shape)
    # print(fireName, date)
    for day in days:
        perim = cv2.imread('data/raw/'+fireName+'/perims/'+day+'.tif', 0)
        days_arr.append(perim)
    # print('PERIM SHAPE: ', perim.shape)
    # print('perim_next day ' + next_day)
    # perim_next = cv2.imread('data/raw/' + fireName + '/perims/' + next_day + '.tif')
    # print('perim_next shape ', perim_next.shape)
    slope = cv2.imread('data/raw/'+ fireName + '/slope.tif', cv2.IMREAD_UNCHANGED)
    print('slope shape ', slope.shape)
    # print('SHAPES: ', dem.shape, aspect.shape, perim.shape, perim_next.shape, slope.shape)
    # if(perim.shape != aspect.shape):
    #     xdiff = aspect.shape[0] - perim.shape[0]
    #     print('xdiff is ', xdiff)
    #     ydiff = aspect.shape[1] - perim.shape[1]
    #     print('ydiff is ', ydiff)
    #     perim = np.lib.pad(perim, ((xdiff, 0), (0,0)), 'constant')
    #     print('new perime shape is ', perim.shape)

    perim_tuple = (dem, aspect, slope)

    for p in days_arr:
        perim_tuple = perim_tuple + (p,)
    print('PERIM TUPLE: ' + str(perim_tuple))


    toAugment = np.dstack(perim_tuple)
    # print('weather shape', weather.shape)
    print('toaugment shape ', toAugment.shape)
    print('x shape is ', toAugment.shape)
    return toAugment, perim_tuple

def rotateWindDirection(theta, fire, date, int_index):
    weather = openWeatherData(date, fire)
    temp, dewpt, temp2, wdir, wspeed, precip, hum = weather
    wdir = (wdir + ((180/np.pi)+theta))%360
    weather = temp, dewpt, temp2, wdir, wspeed, precip, hum
    np.savetxt('data/raw/' + fire+ 'Augmented' + int_index + '/weather/' + date + '.csv', weather, delimiter=',')
    return weather

def doMore(x, fire, days, p_tuple):
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
    augmented, theta = oidg.random_transform(x1, 7)
    print(augmented.shape)

    int_index = strftime("%d%b%H:%M:%S", localtime()) + str(np.random.randint(low=1, high=99)) + str(time.time())
    to_save_dir = os.mkdir('data/raw/' + fire + 'Augmented' + int_index + '/')
    to_save_weather = os.mkdir('data/raw/' + fire + 'Augmented' + int_index+ '/weather/' )
    to_save_perims = os.mkdir('data/raw/' + fire + 'Augmented'  + int_index + '/perims/')

    for date in days:
        weather = rotateWindDirection(theta, fire, date, int_index)
    # np.savetxt('data/raw/demafterreturn.csv', augmented[:,:,0], delimiter=',')

    dem = augmented[:,:,0]
    aspect = augmented[:,:,1]
    #perim = augmented[:,:,2]
    # perim_next =
    slope = augmented[:,:,2]
    # cv2.imwrite('before'+ fire+ date+ '.png', before.reshape(before.shape[:2]))


    print('current dir is ', os.listdir())
    os.chdir('data/raw/' + fire + 'Augmented' + int_index + '/')
    # cv2.imwrite('dem.tif', result.reshape(dem.shape[:2]))
    cv2.imwrite('dem.tif', dem)
    cv2.imwrite('aspect.tif', aspect)
    cv2.imwrite('slope.tif', slope)
    os.chdir('perims/')
    for n, perim in enumerate(p_tuple, 0): #figure out how to do this with two perims
        cv2.imwrite(days[n] +'.tif', p_tuple[n+3])
    os.chdir('../../../..')
    print(os.listdir())
    print('done with 1')

fires = ['riceRidge', 'coldSprings'] # , 'beaverCreek'
rrdays = ['0731', '0801', '0802', '0803']
csdays = ['0711', '0712', '0713', '0714']
# bcdays = ['0629', '0630']
# bcdays2 = ['0711', '0712', '0713', '0714', '0715', '0716']
# bcdays3 = ['0801', '0802']
# bcdays4 = ['0804', '0805']
# bcdays5 = ['0807', '0808', '0809', '0810']

for fire in fires:
    if fire == fires[0]:
        # for r, value in enumerate(rrdays[:-1], 0):
        x, y = collectData(fire, rrdays)
        doMore(x, fire, rrdays, y)
    if fire == fires[1]:
        # for c, value in enumerate(csdays[:-1], 0):
        x, y = collectData(fire, csdays)
        doMore(x, fire, csdays, y)
    # if i == fires[2]:
    #     for b, value in enumerate(bcdays[:-1], 0):
    #         x = collectData(i, bcdays[b], bcdays[b+1])
    #         doMore(x, i, bcdays[b])
    #     for b, value in enumerate(bcdays2[:-1], 0):
    #         x = collectData(i, bcdays2[b], bcdays2[b+1])
    #         doMore(x, i, bcdays2[b])
    #     for b, value in enumerate(bcdays3[:-1], 0):
    #         x = collectData(i, bcdays3[b], bcdays3[b+1])
    #         doMore(x, i, bcdays3[b])
    #     for b, value in enumerate(bcdays4[:-1], 0):
    #         x = collectData(i, bcdays4[b], bcdays4[b+1])
    #         doMore(x, i, bcdays4[b])
    #     for b, value in enumerate(bcdays5[:-1], 0):
    #         x = collectData(i, bcdays5[b], bcdays5[b+1])
    #         doMore(x, i, bcdays5[b])


# the .flow() command below generates batches of randomly transformed images
# and saves the results to the `preview/` directory
# i = 0
# for batch in datagen.flow(augmented, batch_size=10,
#                           save_to_dir='data/raw/riceRidgeAugmented', save_prefix='test', save_format='tif'):
#     i += 1
#     if i > 25:
#         break  # otherwise the generator would loop indefinitely
