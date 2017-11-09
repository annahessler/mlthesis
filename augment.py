from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import os
import cv2
import numpy as np
from lib import image
import time
from time import localtime, strftime
from scipy.misc import imsave
from decimal import *


  
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
    # data = np.loadtxt(fname, usecols=[5,12], skiprows=1, delimiter='')
    date_list = np.loadtxt(fname, usecols=range(0,2), dtype='float32', skiprows=1, delimiter=',')
    print('datelist is ', date_list)
    no_augment = np.loadtxt(fname, usecols=range(2,5), skiprows=1, delimiter=',', dtype='float32')
    print('no augment ', no_augment)
    augment = np.loadtxt(fname, usecols=range(5,12), skiprows=1, delimiter=',', dtype='float32').T
    print('augment ', augment)
    return date_list, no_augment, augment

# def createWeatherMetrics(weatherData):
#     temp, dewpt, temp2, wdir, wspeed, precip, hum = weatherData
#     avgWSpeed = sum(wspeed)/len(wspeed)
#     totalPrecip = sum(precip)
#     avgWDir= sum(wdir)/len(wdir)
#     avgHum = sum(hum)/len(hum)
#     return np.array( [max(temp), avgWSpeed, avgWDir, totalPrecip, avgHum])


# datagen = ImageDataGenerator(
#         rotation_range=300,
#         width_shift_range=0.2,
#         height_shift_range=0.2,
#         zoom_range=0.2,
#         horizontal_flip=True,
#         fill_mode='nearest')

def collectData(fireName, date, next_day):
    # still need landsat to be read in 
    dem = cv2.imread('data/raw/' + fireName + '/dem.tif', cv2.IMREAD_UNCHANGED)
    print('DEM SHAPE: ', dem.shape)
    # dem = np.array(dem, dtype='float16') 
    # print('directory is ', os.listdir())
    # np.savetxt('data/raw/dembefore.csv', dem, delimiter=',')
    aspect = cv2.imread('data/raw/'+ fireName + '/aspect.tif', cv2.IMREAD_UNCHANGED)
    print('ASPECT SHAPE: ', aspect.shape)
    print(fireName, date)
    perim = cv2.imread('data/raw/'+fireName+'/perims/'+date+'.tif', 0)
    print('PERIM SHAPE: ', perim.shape)
    print('perim_next day ' + next_day)
    # perim_next = cv2.imread('data/raw/' + fireName + '/perims/' + next_day + '.tif')
    # print('perim_next shape ', perim_next.shape)
    slope = cv2.imread('data/raw/'+ fireName + '/slope.tif', cv2.IMREAD_UNCHANGED)
    print('slope shape ', slope.shape)
    print('SHAPES: ', dem.shape, aspect.shape, perim.shape, slope.shape)# before slope.shape->, perim_next.shape
    # if(perim.shape != aspect.shape):
    #     xdiff = aspect.shape[0] - perim.shape[0]
    #     print('xdiff is ', xdiff)
    #     ydiff = aspect.shape[1] - perim.shape[1]
    #     print('ydiff is ', ydiff)
    #     perim = np.lib.pad(perim, ((xdiff, 0), (0,0)), 'constant')
    #     print('new perime shape is ', perim.shape)
    
    toAugment = np.dstack((dem, aspect, perim, slope)) #, perim_next
    # print('weather shape', weather.shape)
    print('toaugment shape ', toAugment.shape)
    print('x shape is ', toAugment.shape)
    return toAugment

def rotateWindDirection(theta, fire, date, int_index):
    date_list, no_augment, weather = openWeatherData(date, fire)
    print('date_list shape is ', date_list.shape)
    print('date_list is ', date_list)
    print('no_augment shape is ', no_augment.shape)
    print('no_augment is ', no_augment)
    print('weather shape is ', weather.shape)
    print('weather is ', weather)
    temp, dewpt, temp2, wdir, wspeed, precip, hum = weather
    wdir = (wdir + ((180/np.pi)+theta))%360
    weather = temp, dewpt, temp2, wdir, wspeed, precip, hum
    weather2 = np.transpose(weather)
    print('new weather shape is ', weather2.shape, weather2)
    all_weather = np.hstack([date_list, no_augment, weather2])
    print('all weather shape is ', all_weather.shape)
    headings = np.zeros(12, dtype='float32')
    print('headings shape is ', headings.shape)
    headings = headings.reshape( (1,) + headings.shape)
    print('new headings shape is ', headings.shape)
    result = np.vstack((headings, all_weather))
    print('result shape is ', result.shape)
    f = 'data/raw/' + fire+ 'Augmented' + int_index + '/weather/' + date + '.csv'
    # np.savetxt(f, headings, delimiter=',')
    # np.savetxt(f, all_weather, delimiter=',')
    # f = open(f)
    # for i in all_weather:
    #     np.savetxt(f, i, delimiter=',')
    print('result is ', result)
    np.savetxt('data/raw/' + fire+ 'Augmented' + int_index + '/weather/' + date + '.csv', result, delimiter=',')

    return weather 

def doMore(x, fire, date):
    infinity = Decimal('Infinity')
    oidg = image.ourImageDataGenerator(
            rotation_range=40,
            fill_mode='constant',
            cval=infinity, 
            data_format = 'channels_last'
        )

    x1 = np.lib.pad(x, ((1,1),(1,1),(0,0)), 'constant')
    augmented, theta = oidg.random_transform(x1, 7)
    print(augmented.shape)

    int_index = strftime("%d%b%H%M%S", localtime()) + str(np.random.randint(low=1, high=99)) + str(time.time())
    os.mkdir('data/raw/' + fire + 'Augmented' + int_index + '/')
    os.mkdir('data/raw/' + fire + 'Augmented' + int_index+ '/weather/' )
    os.mkdir('data/raw/' + fire + 'Augmented'  + int_index + '/perims/')

    weather = rotateWindDirection(theta, fire, date, int_index)
    # np.savetxt('data/raw/demafterreturn.csv', augmented[:,:,0], delimiter=',')
    dem = augmented[:,:,0]
    aspect = augmented[:,:,1]
    perim = augmented[:,:,2]
    # perim_next = 
    # slope = augmented[:,:,4]
    # cv2.imwrite('before'+ fire+ date+ '.png', before.reshape(before.shape[:2]))
 
    
    print('current dir is ', os.listdir())
    # os.chdir('data/raw/' + fire + 'Augmented' + int_index + '/')
    # cv2.imwrite('dem.tif', result.reshape(dem.shape[:2])) 
    print('dem before conversions is ', dem)
    print('dem converted is ', dem.astype(np.float32))
    print("dem dtype is ", dem.astype(np.float32).dtype)
    dem2 = dem.astype(np.float32)
    print(dem2.shape, dem2.dtype)
    folder = 'data/raw/' + fire + 'Augmented' + int_index
    imsave(folder + '/dem.tif', dem2)
    # cv2.imwrite('data/raw/' + fire + 'Augmented' + int_index + '/dem.tif', dem2) 
    # cv2.imwrite('data/raw/' + fire + 'Augmented' + int_index + '/aspect.tif', aspect)
    # # cv2.imwrite('data/raw/' + fire + 'Augmented' + int_index + '/slope.tif', slope)
    # cv2.imwrite('data/raw/' + fire + 'Augmented' + int_index + '/perims/' + date +'.tif', perim)
    # os.chdir('../../../..')
    # print(os.listdir())
    print('done with 1')

fires = ['riceRidge', 'coldSprings'] # ,  'coldSprings''riceRidge','coldSprings'
rrdays = ['0731', '0801', '0802', '0803']
csdays = ['0711', '0712', '0713', '0714']
# bcdays = ['0629', '0630']
# bcdays2 = ['0711', '0712', '0713', '0714', '0715', '0716']
# bcdays3 = ['0801', '0802']
# bcdays4 = ['0804', '0805']
# bcdays5 = ['0807', '0808', '0809', '0810']

for i in fires:
    if i == fires[0]:
        for r, value in enumerate(rrdays[:-1], 0):
            x = collectData(i, rrdays[r], rrdays[r+1])
            doMore(x, i, rrdays[r])
    if i == fires[1]:
        for c, value in enumerate(csdays[:-1], 0):
            x = collectData(i, csdays[c], csdays[c+1])
            doMore(x, i, csdays[c])
    # if i == fires[0]:
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
