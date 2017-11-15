from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import os
import cv2
import numpy as np
from lib import image, util
import time
from time import localtime, strftime
from scipy.misc import imsave
from decimal import *



def openEndingPerim(dateString, fire):
    month, day = dateString[:2], dateString[2:]
    nextDay = str(int(day)+1).zfill(2)
    guess = month+nextDay
    perimFileName = 'data/'+ fire + '/perims/' + guess + '.tif'
    perim = cv2.imread(perimFileName, cv2.IMREAD_UNCHANGED)
    if perim is None:
        # overflowed the month, that file didnt exist
        nextMonth = str(int(month)+1).zfill(2)
        guess = nextMonth+'01'
        perimFileName = 'data/' + fire + '/perims/' + guess + '.tif'
        perim = cv2.imread(perimFileName, cv2.IMREAD_UNCHANGED)
        if perim is None:
            raise RuntimeError('Could not find a perimeter for the day after ' + dateString)
        return perim

def openWeatherData(dateString, fireName):
    fname = 'data/' + fireName + '/weather/' + dateString + '.csv'
    # the first row is the headers, and only cols 5-12 are actual data
    # data = np.loadtxt(fname, usecols=[5,12], skiprows=1, delimiter='')
    date_list = np.loadtxt(fname, usecols=range(0,2), dtype='float32', skiprows=1, delimiter=',')
    no_augment = np.loadtxt(fname, usecols=range(2,5), skiprows=1, delimiter=',', dtype='float32')
    augment = np.loadtxt(fname, usecols=range(5,12), skiprows=1, delimiter=',', dtype='float32').T
    return date_list, no_augment, augment

def collectData(fireName, days):
    days_arr = []
    dem = util.openImg('data/' + fireName + '/dem.tif')
    aspect = util.openImg('data/'+ fireName + '/aspect.tif')
    landsat4 = util.openImg('data/'+ fireName + '/band_4.tif')
    print('landsat4.tif shae', landsat4.shape)
    landsat3 = util.openImg('data/'+ fireName + '/band_3.tif')
    print('landsat3.tif shae', landsat3.shape)
    landsat2 = util.openImg('data/'+ fireName + '/band_2.tif')
    np.savetxt('band2lookatthis.csv', landsat2, delimiter=',')
    landsat5 = util.openImg('data/'+ fireName + '/band_5.tif')
    print('before landsat4 shape is ', landsat4.shape)

    ndvi = util.openImg('data/'+ fireName + '/ndvi.tif')
    slope = util.openImg('data/'+ fireName + '/slope.tif')
    for day in days:
        print('day is ', day)
        days_arr.append(util.openImg('data/'+fireName+'/perims/'+day+'.tif'))
    print('COLLECT DATA SHAPES: ', dem.shape, aspect.shape, landsat4.shape, landsat3.shape, landsat2.shape, landsat5.shape, ndvi.shape, slope.shape)

    fire_tuple = (dem, aspect, slope, landsat4, landsat3, landsat2, landsat5, ndvi)
    print('firetuple shape is ', len(fire_tuple))

    for p in days_arr:
        fire_tuple = fire_tuple + (p,)
    # print('FIRE TUPLE: ' + str(fire_tuple))
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
    f = 'data/' + fire+ 'Augmented' + int_index + '/weather/' + date + '.csv'
    np.savetxt(f, result, delimiter=',')

    return weather

# doMore(toaugment, fire, csdays, fire_tuple, day_arr)
def doMore(toaugment, fire, days, f_tuple, perim_array):
    infinity = Decimal('Infinity')
    oidg = image.ourImageDataGenerator(
            rotation_range=330, #40, 80, 120, 160, 200, 240, 300, 360, 280, 330
            # fill_mode='constant',
            # cval=np.nan,
            data_format = 'channels_last'
        )

    toaugment = np.lib.pad(toaugment, ((1,1),(1,1),(0,0)), 'constant', constant_values=np.nan )
    augmented, theta = oidg.random_transform(toaugment, 7)
    np.savetxt('landsat2directlyafterrandomtransformcall.csv', augmented[:,:,4], delimiter=',')
    print(np.array_equal(augmented[:,:,4], toaugment[:,:,4]))
    int_index = makeFolders(fire)

    for day in days:
        weather = rotateWindDirection(theta, fire, day, int_index)
    # np.savetxt('data/demafterreturn.csv', augmented[:,:,0], delimiter=',')
    saveFiles(augmented, int_index, perim_array, days)

    print('done with 1+++++++++++++++++++++++++++++++++++++++++++++++++++')

def saveFiles(augmented, int_index, perim_array, days):
    # print('COLLECT DATA SHAPES: ', dem.shape, aspect.shape, landsat.shape, ndvi.shape, slope.shape)
    dem = augmented[:,:,0]
    aspect = augmented[:,:,1]
    #perim = augmented[:,:,2]
    # perim_next =
    # others = augmented[:,:,2:]
    landsat = augmented[:,:,2:6]
    # others2 = others[:,:,4:]
    ndvi = augmented[:,:,6]
    slope = augmented[:,:,7]
    othersperims = augmented[:,:,8:]
    days = days
    print('days is ', days)
    print('other layers are' , othersperims.shape, othersperims)
    print('landsat shape is ', landsat.shape)
    print(landsat)
    # cv2.imwrite('before'+ fire+ date+ '.png', before.reshape(before.shape[:2]))
    print('current dir is ', os.listdir())
    folder = 'data/' + fire + 'Augmented' + int_index
    for n, perim in enumerate(cv2.split(othersperims)): #figure out how to do this with two perims
        util.saveImg(folder + '/perims/' + days[n] +'.tif', perim )
        # imsave(folder + '/perims/' + days[n] +'.tif', perim)
    util.saveImg(folder + '/dem.tif', dem)
    util.saveImg(folder + '/ndvi.tif', ndvi)
    util.saveImg(folder + '/aspect.tif', aspect)
    np.savetxt('landsatrightbeforesavetif.csv', landsat[:,:,2], delimiter=',')
    util.saveImg(folder + '/band_4.tif', landsat[:,:,0])
    util.saveImg(folder + '/band_3.tif', landsat[:,:,1])
    util.saveImg(folder + '/band_2.tif', landsat[:,:,2])
    util.saveImg(folder + '/band_5.tif', landsat[:,:,3])
    # landsat_test_write = util.openImg(folder + '/landsat.tif')
    # np.savetxt('landsataftersavereqrite1.csv', landsat_test_write[:,:,0], delimiter=',')
    # np.savetxt('landsataftersavereqrite2.csv', landsat_test_write[:,:,1], delimiter=',')
    # np.savetxt('landsataftersavereqrite3.csv', landsat_test_write[:,:,2], delimiter=',')
    # np.savetxt('landsataftersavereqrite4.csv', landsat_test_write[:,:,3], delimiter=',')

def makeFolders(fire):
    int_index = strftime("%d%b%H%M%S", localtime()) + str(np.random.randint(low=1, high=99)) + str(time.time())
    folder = os.mkdir('data/' + fire + 'Augmented' + int_index + '/')
    fweather = os.mkdir('data/' + fire + 'Augmented' + int_index+ '/weather/' )
    fperims = os.mkdir('data/' + fire + 'Augmented'  + int_index + '/perims/')
    return int_index

fires = ['riceRidge', 'coldSprings', 'beaverCreek', 'haydenPass', 'junkins', 'peekaboo', 'pineTree', 'redDirt', 'gutzler', 'ecklund', 'redDirt2']
rrdays = ['0731', '0801', '0802', '0803']
csdays = ['0711', '0712', '0713', '0714']
bcdays = ['0629', '0630', '0711', '0712', '0713', '0714', '0715', '0716', '0801', '0802', '0804', '0805', '0807', '0808', '0809', '0810']
# bcdays2 = ['0711', '0712', '0713', '0714', '0715', '0716']
# bcdays3 = ['0801', '0802'] #TAKE OUT
# bcdays4 = ['0804', '0805']
# bcdays5 = ['0807', '0808', '0809', '0810']
hpdays = ['0712', '0713', '0714', '0715', '0716', '0717', '0718', '0719']
jdays = ['1028','1029', '1030', '1020', '1021', '1023', '1024']
# jdays2 = ['1020', '1021']
# jdays3 = ['1023', '1024']
peekdays = ['0710', '0711']
ptdays = ['0911', '0912']
rddays = ['0719', '0720']
rd2days = ['0719', '0720']
gdays = ['0703', '0704', '0705', '0706']
eckdays = ['0628','0629', '0630']


for fire in fires:
    if fire == fires[0]:
        # for r, value in enumerate(rrdays[:-1], 0):
        toaugment, fire_tuple, day_arr = collectData(fire, rrdays)
        doMore(toaugment, fire, rrdays, fire_tuple, day_arr)
    if fire == fires[1]:
        # for c, value in enumerate(csdays[:-1], 0):
        toaugment, fire_tuple, day_arr = collectData(fire, csdays)
        doMore(toaugment, fire, csdays, fire_tuple, day_arr)
    # if fire == fires[2]:
    #     toaugment, fire_tuple, day_arr = collectData(fire, bcdays)
    #     doMore(toaugment, fire, bcdays, fire_tuple, day_arr)
    #     toaugment, fire_tuple, day_arr = collectData(fire, bcdays2)
    #     doMore(toaugment, fire, bcdays2, fire_tuple, day_arr)
    #     toaugment, fire_tuple, day_arr = collectData(fire, bcdays3)
    #     doMore(toaugment, fire, bcdays3, fire_tuple, day_arr)
    #     toaugment, fire_tuple, day_arr = collectData(fire, bcdays4)
    #     doMore(toaugment, fire, bcdays4, fire_tuple, day_arr)
    #     toaugment, fire_tuple, day_arr = collectData(fire, bcdays5)
    #     doMore(toaugment, fire, bcdays5, fire_tuple, day_arr)
    if fire == fires[3]:
        # for c, value in enumerate(csdays[:-1], 0):
        toaugment, fire_tuple, day_arr = collectData(fire, hpdays)
        doMore(toaugment, fire, hpdays, fire_tuple, day_arr)
    if fire == fires[4]:
        # for c, value in enumerate(csdays[:-1], 0):
        toaugment, fire_tuple, day_arr = collectData(fire, jdays)
        doMore(toaugment, fire, jdays, fire_tuple, day_arr)
        # toaugment, fire_tuple, day_arr = collectData(fire, jdays2)
        # doMore(toaugment, fire, jdays2, fire_tuple, day_arr)
        # toaugment, fire_tuple, day_arr = collectData(fire, jdays3)
        # doMore(toaugment, fire, jdays3, fire_tuple, day_arr)
    # if fire == fires[5]:
    #     toaugment, fire_tuple, day_arr = collectData(fire, peekdays)
    #     doMore(toaugment, fire, peekdays, fire_tuple, day_arr)
    if fire == fires[6]:
    # for c, value in enumerate(csdays[:-1], 0):
        toaugment, fire_tuple, day_arr = collectData(fire, ptdays)
        doMore(toaugment, fire, ptdays, fire_tuple, day_arr)
    if fire == fires[7]:
    # for c, value in enumerate(csdays[:-1], 0):
        toaugment, fire_tuple, day_arr = collectData(fire, rddays)
        doMore(toaugment, fire, rddays, fire_tuple, day_arr)
    # if fire == fires[8]:
    # # for c, value in enumerate(csdays[:-1], 0):
    #     toaugment, fire_tuple, day_arr = collectData(fire, gdays)
    #     doMore(toaugment, fire, gdays, fire_tuple, day_arr)
    if fire == fires[9]:
    # for c, value in enumerate(csdays[:-1], 0):
        toaugment, fire_tuple, day_arr = collectData(fire, eckdays)
        doMore(toaugment, fire, eckdays, fire_tuple, day_arr)
    if fire == fires[10]:
    # for c, value in enumerate(csdays[:-1], 0):
        toaugment, fire_tuple, day_arr = collectData(fire, rd2days)
        doMore(toaugment, fire, rd2days, fire_tuple, day_arr)
