from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import os
import cv2
import numpy as np
from lib import image
import time
from time import localtime, strftime
from scipy.misc import imsave
from decimal import *
import util



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

def collectData(fireName, days):
    days_arr = []
    dem = util.openImg('data/raw/' + fireName + '/dem.tif')
    aspect = util.openImg('data/raw/'+ fireName + '/aspect.tif')  
    landsat = util.openImg('data/raw/'+ fireName + '/landsat.png')  
    ndvi = util.openImg('data/raw/'+ fireName + '/ndvi.tif')
    slope = util.openImg('data/raw/'+ fireName + '/slope.tif')
    for day in days:
        print('day is ', day)
        days_arr.append(util.openImg('data/raw/'+fireName+'/perims/'+day+'.tif'))
    print('COLLECT DATA SHAPES: ', dem.shape, aspect.shape, landsat.shape, ndvi.shape, slope.shape)

    fire_tuple = (dem, aspect, slope, landsat, ndvi)

    for p in days_arr:
        fire_tuple = fire_tuple + (p,)
    print('FIRE TUPLE: ' + str(fire_tuple))


    toAugment = np.dstack(fire_tuple) #This needs to be tuple?????
    print('toaugment shape ', toAugment.shape)
    return toAugment, fire_tuple, days_arr

def rotateWindDirection(theta, fire, date, int_index):
    date_list, no_augment, weather = openWeatherData(date, fire)
    temp, dewpt, temp2, wdir, wspeed, precip, hum = weather
    wdir = (wdir + ((180/np.pi)+theta))%360
    weather = temp, dewpt, temp2, wdir, wspeed, precip, hum
    weather2 = np.transpose(weather)
    all_weather = np.hstack([date_list, no_augment, weather2])
    headings = np.zeros(12, dtype='float32')
    headings = headings.reshape( (1,) + headings.shape)
    result = np.vstack((headings, all_weather))
    f = 'data/raw/' + fire+ 'Augmented' + int_index + '/weather/' + date + '.csv'
    np.savetxt(f, result, delimiter=',')

    return weather 

# doMore(toaugment, fire, csdays, fire_tuple, day_arr)
def doMore(toaugment, fire, days, f_tuple, perim_array):
    infinity = Decimal('Infinity')
    oidg = image.ourImageDataGenerator(
            rotation_range=40,
            fill_mode='constant',
            cval=infinity, 
            data_format = 'channels_last'
        )

    toaugment = np.lib.pad(toaugment, ((1,1),(1,1),(0,0)), 'constant', constant_values=np.nan )
    augmented, theta = oidg.random_transform(toaugment, 7)

    int_index = makeFolders(fire)

    for day in days:
        weather = rotateWindDirection(theta, fire, day, int_index)
    # np.savetxt('data/raw/demafterreturn.csv', augmented[:,:,0], delimiter=',')
    saveFiles(augmented, int_index, perim_array, days)

    print('done with 1')

def saveFiles(augmented, int_index, perim_array, days):
    dem = augmented[:,:,0]
    aspect = augmented[:,:,1]
    #perim = augmented[:,:,2]
    # perim_next =
    landsat = augmented[:,:,2]
    # cv2.imwrite('before'+ fire+ date+ '.png', before.reshape(before.shape[:2]))
    print('current dir is ', os.listdir())
    folder = 'data/raw/' + fire + 'Augmented' + int_index
    for n, perim in enumerate(perim_array, 0): #figure out how to do this with two perims
        imsave(folder + '/perims/' + days[n] +'.tif', perim)
    imsave(folder + '/dem.tif', dem)

def makeFolders(fire):
    int_index = strftime("%d%b%H%M%S", localtime()) + str(np.random.randint(low=1, high=99)) + str(time.time())
    folder = os.mkdir('data/raw/' + fire + 'Augmented' + int_index + '/')
    fweather = os.mkdir('data/raw/' + fire + 'Augmented' + int_index+ '/weather/' )
    fperims = os.mkdir('data/raw/' + fire + 'Augmented'  + int_index + '/perims/')
    return int_index

fires = ['riceRidge', 'coldSprings'] # ,  'coldSprings''riceRidge','coldSprings'
rrdays = ['0731', '0801', '0802', '0803']
csdays = ['0711', '0712', '0713', '0714']
# bcdays = ['0629', '0630']
# bcdays2 = ['0711', '0712', '0713', '0714', '0715', '0716']
# bcdays3 = ['0801', '0802'] TAKE OUT
# bcdays4 = ['0804', '0805']
# bcdays5 = ['0807', '0808', '0809', '0810']

for fire in fires:
    if fire == fires[0]:
        # for r, value in enumerate(rrdays[:-1], 0):
        toaugment, fire_tuple, day_arr = collectData(fire, rrdays)
        doMore(toaugment, fire, rrdays, fire_tuple, day_arr)
    # if fire == fires[1]:
    #     # for c, value in enumerate(csdays[:-1], 0):
    #     toaugment, fire_tuple, day_arr = collectData(fire, csdays)
    #     doMore(toaugment, fire, csdays, fire_tuple, day_arr)
    # if i == fires[2]:



# the .flow() command below generates batches of randomly transformed images
# and saves the results to the `preview/` directory
# i = 0
# for batch in datagen.flow(augmented, batch_size=10,
#                           save_to_dir='data/raw/riceRidgeAugmented', save_prefix='test', save_format='tif'):
#     i += 1
#     if i > 25:
#         break  # otherwise the generator would loop indefinitely
