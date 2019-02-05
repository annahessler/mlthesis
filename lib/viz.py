from time import localtime, strftime
import csv

import numpy as np
import cv2
import random
try:
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation
    print('Successfully imported pyplot')
except:
    print('Failed to import pyplot ')

from lib import dataset
from lib import util

def renderPredictions(dataset, predictions, preResu):
    day2pred = {}
    for pt, pred in predictions.items():
        burnName, date, location = pt
        day = (burnName, date)
        if day not in day2pred:
            day2pred[day] = []
        pair = (location, float(pred))
        # pair = (location, preResu[0])
        # preResu.pop(0)
        day2pred[day].append(pair)

    results = {}
    for (burnName, date), locsAndPreds in day2pred.items():
        locs, preds = zip(*locsAndPreds)
        xs,ys = zip(*locs)
        preds = [pred+1 for pred in preds]
        burn = dataset.data.burns[burnName]
        canvas = np.zeros(burn.layerSize, dtype=np.float32)
        canvas[(xs,ys)] = np.array(preds, dtype=np.float32)
        results[(burnName, date)] = canvas
    return results

#loads farsite .tif as a 2D array and compares with perims
def compare_farsite(dataset, loc, res, size, fireDate):
    total_f_score = 0
    total_precision = 0
    total_recall = 0
    perim_num = 0
    max_f_score = 0
    max_f_score_date = 0
    min_f_score = 1
    min_f_score_date = 0
    max_burn_name = ""
    min_burn_name = ""

    farsite = cv2.imread('First711BurnFloat01.tif', cv2.IMREAD_UNCHANGED)


    # plt.imshow(farsite, cmap = 'gray', interpolation = 'bicubic')
    # plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
    # plt.show()
    # exit()

    for burnName, date in dataset.getUsedBurnNamesAndDates():
        perim_num = perim_num + 1
        burn = dataset.data.burns[burnName]
        day = dataset.data.getDay(burnName, date)
        h,w = day.startingPerim.shape
        normedDEM = util.normalize(burn.layers['dem'])
        numOfFire = len(dataset.getUsedBurnNamesAndDates())

        curDay = []
        curRes = []
        pixelsForDate = 0
        while(len(fireDate) > 0 and date == fireDate[0]):
            curDay.append(loc[0])
            # if rand is True:
            #     curRes.append(random.uniform(0.0, 1.0))
            # else:
            curRes.append(res[0])
            loc.pop(0)
            res.pop(0)
            fireDate.pop(0)

        posx = []
        posy = []

        for tup in curDay:
            posx.append(tup[1])
            posy.append(tup[0])

        if date == '0711':
            new_burn = 0
            initial_perim = 0
            farsite_burn = 0

            for i in range(0, h):
                for j in range(0, w):
                    if day.startingPerim.astype(np.uint8)[i][j] == 1:
                        initial_perim = initial_perim + 1
                    if day.startingPerim.astype(np.uint8)[i][j] == 0 and day.endingPerim.astype(np.uint8)[i][j] == 1:
                        new_burn = new_burn + 1
                    if day.startingPerim.astype(np.uint8)[i][j] == 0 and farsite[i][j] == 1:
                        farsite_burn = farsite_burn + 1

            perc_burned = farsite / initial_perim



            correct = 0
            incorrect = 0
            shouldBeBurnt = 0
            didBurn = 0
            middle = 0
            TP=0
            FP=0
            TN=0
            FN=0

            for pos in range(0, len(posx)):
                # print(day.startingPerim.astype(np.uint8)[posy[pos]][posx[pos]])
                # print("far: ", farsite[posy[pos]][posx[pos]])
                if day.endingPerim.astype(np.uint8)[posy[pos]][posx[pos]] == 1:
                    shouldBeBurnt = shouldBeBurnt + 1
                    if farsite[posy[pos]][posx[pos]] == 1:
                        didBurn = didBurn + 1
                if farsite[posy[pos]][posx[pos]] == 1 and day.endingPerim.astype(np.uint8)[posy[pos]][posx[pos]] == 1:
                    correct = correct + 1
                    TP = TP + 1
                elif farsite[posy[pos]][posx[pos]] == 0 and day.endingPerim.astype(np.uint8)[posy[pos]][posx[pos]] == 0:
                    correct = correct + 1
                    TN = TN + 1
                elif farsite[posy[pos]][posx[pos]] == 0 and day.endingPerim.astype(np.uint8)[posy[pos]][posx[pos]] == 1: #.03
                    FN = FN + 1
                    incorrect = incorrect + 1
                elif farsite[posy[pos]][posx[pos]] == 1 and day.endingPerim.astype(np.uint8)[posy[pos]][posx[pos]] == 0: #.97
                    FP = FP + 1
                    incorrect = incorrect + 1

            try:
                precision = (TP/(TP + FP))
                recall = (TP/(TP + FN))
                f_frac = ((precision * recall)/(precision + recall))
                f_score = 2 * f_frac
                total_recall = total_recall + recall
                total_precision = total_precision + precision
            except:
                print("ERROR: Not enough points to calculate F-score")
                precision = 0
                recall = 0
                f_score = 0


            total_f_score = total_f_score + f_score

            print()
            print("pixels tested " , len(posx))

            try:
                burnAcc = float("%0.2f" % ((didBurn/shouldBeBurnt) * 100))
            except:
                burnAcc = 0
                print("ERROR: No pixels tested are inside of the next fire perimeter")

            f_score_print = float("%0.4f" % f_score)
            accCor = float("%0.2f" % ((correct/(correct + incorrect)) * 100))
            accIncor = float("%0.2f" % ((incorrect/len(posx)) * 100))

            if f_score > max_f_score:
                max_f_score = f_score_print
                max_f_score_date = date
                max_burn_name = burnName
            elif f_score < min_f_score:
                min_f_score = f_score_print
                min_f_score_date = date
                min_burn_name = burnName


            # print(didBurn)
            # print(shouldBeBurnt)
            print(date, " farsite fire: ", burnName)
            print(date , " farsite precision: ", precision)
            print(date , " farsite recall: ", recall)
            print(date , " farsite F score: ", f_score_print)
            #This is the percent of pixels that actually burned that the model was correct on (doesn't count what
            #the model said would burn that didnt)
            print(date , " farsite percent model thinks burned that actually did: ", burnAcc,  " %")
            print(date , " farsite accuracy correct: ", accCor,  " %")
            print(date , " farsite accuracy incorrect: ", accIncor, " %")
            print(date , " farsite percent fire grew: ", perc_burned)
            # print("TOTAL F SCORE AND PERIM NUM: ", total_f_score, " ", perim_num)
            exit()



#This function gathers a bunch of statistics about the predictions and prints them out
def getNumbers(dataset, loc, res, size, fireDate):
    total_f_score = 0
    total_precision = 0
    total_recall = 0
    perim_num = 0
    max_f_score = 0
    max_f_score_date = 0
    min_f_score = 1
    min_f_score_date = 0
    max_burn_name = ""
    min_burn_name = ""
    for burnName, date in dataset.getUsedBurnNamesAndDates():
        perim_num = perim_num + 1
        burn = dataset.data.burns[burnName]
        day = dataset.data.getDay(burnName, date)
        h,w = day.startingPerim.shape
        normedDEM = util.normalize(burn.layers['dem'])
        numOfFire = len(dataset.getUsedBurnNamesAndDates())

        new_burn = 0
        initial_perim = 0

        for i in range(0, h):
            for j in range(0, w):
                if day.startingPerim.astype(np.uint8)[i][j] == 1:
                    initial_perim = initial_perim + 1
                if day.startingPerim.astype(np.uint8)[i][j] == 0 and day.endingPerim.astype(np.uint8)[i][j] == 1:
                    new_burn = new_burn + 1

        perc_burned = new_burn / initial_perim

        curDay = []
        curRes = []
        pixelsForDate = 0
        while(len(fireDate) > 0 and date == fireDate[0]):
            curDay.append(loc[0])
            # if rand is True:
            #     curRes.append(random.uniform(0.0, 1.0))
            # else:
            curRes.append(res[0])
            loc.pop(0)
            res.pop(0)
            fireDate.pop(0)

        posx = []
        posy = []

        for tup in curDay:
            posx.append(tup[1])
            posy.append(tup[0])

        burnThePerim = False #this just burns the 20 pixels around the fire

        if burnThePerim:
            print("Burning the perimeter only.")
            for pos in range(0, len(curDay)):
                try:
                    if day.startingPerim.astype(np.uint8)[posy[pos] + 20][posx[pos] + 20] == 1:
                        curRes[pos] = 1
                    elif day.startingPerim.astype(np.uint8)[posy[pos]][posx[pos] + 20] == 1:
                        curRes[pos] = 1
                    elif day.startingPerim.astype(np.uint8)[posy[pos] - 20][posx[pos] + 20] == 1:
                        curRes[pos] = 1
                    elif day.startingPerim.astype(np.uint8)[posy[pos] + 20][posx[pos]] == 1:
                        curRes[pos] = 1
                    elif day.startingPerim.astype(np.uint8)[posy[pos]][posx[pos]] == 1:
                        curRes[pos] = 1
                    elif day.startingPerim.astype(np.uint8)[posy[pos] - 20][posx[pos]] == 1:
                        curRes[pos] = 1
                    elif day.startingPerim.astype(np.uint8)[posy[pos] + 20][posx[pos] - 20] == 1:
                        curRes[pos] = 1
                    elif day.startingPerim.astype(np.uint8)[posy[pos]][posx[pos] - 20] == 1:
                        curRes[pos] = 1
                    elif day.startingPerim.astype(np.uint8)[posy[pos] - 20][posx[pos] - 20] == 1:
                        curRes[pos] = 1
                    else:
                        curRes[pos] = 0
                except:
                    curRes[pos] = 0

        correct = 0
        incorrect = 0
        shouldBeBurnt = 0
        didBurn = 0
        middle = 0
        TP=0
        FP=0
        TN=0
        FN=0

        for pos in range(0, len(posx)):
            # print(day.startingPerim.astype(np.uint8)[posy[pos]][posx[pos]])
            if day.endingPerim.astype(np.uint8)[posy[pos]][posx[pos]] == 1:
                shouldBeBurnt = shouldBeBurnt + 1
                if curRes[pos] > 0.97:
                    didBurn = didBurn + 1
            if curRes[pos] > 0.5 and day.endingPerim.astype(np.uint8)[posy[pos]][posx[pos]] == 1:
                correct = correct + 1
                TP = TP + 1
            elif curRes[pos] < 0.5 and day.endingPerim.astype(np.uint8)[posy[pos]][posx[pos]] == 0:
                correct = correct + 1
                TN = TN + 1
            elif curRes[pos] < 0.5 and day.endingPerim.astype(np.uint8)[posy[pos]][posx[pos]] == 1: #.03
                FN = FN + 1
                incorrect = incorrect + 1
            elif curRes[pos] > 0.5 and day.endingPerim.astype(np.uint8)[posy[pos]][posx[pos]] == 0: #.97
                FP = FP + 1
                incorrect = incorrect + 1
            # elif curRes[pos] > 0.03 and curRes[pos] < 0.97:
            #     middle = middle + 1





            # else:
            #     incorrect = incorrect + 1
        #
        # print("these are here: ", TP, TN, FN, FP)
        # print("This is middle: ", middle)

        try:
            precision = (TP/(TP + FP))
            recall = (TP/(TP + FN))
            f_frac = ((precision * recall)/(precision + recall))
            f_score = 2 * f_frac
            total_recall = total_recall + recall
            total_precision = total_precision + precision
        except:
            print("ERROR: Not enough points to calculate F-score")
            precision = 0
            recall = 0
            f_score = 0


        total_f_score = total_f_score + f_score

        print()
        print("pixels tested " , len(posx))

        try:
            burnAcc = float("%0.2f" % ((didBurn/shouldBeBurnt) * 100))
        except:
            burnAcc = 0
            print("ERROR: No pixels tested are inside of the next fire perimeter")

        f_score_print = float("%0.4f" % f_score)
        accCor = float("%0.2f" % ((correct/(correct + incorrect)) * 100))
        accIncor = float("%0.2f" % ((incorrect/len(posx)) * 100))

        if f_score > max_f_score:
            max_f_score = f_score_print
            max_f_score_date = date
            max_burn_name = burnName
        elif f_score < min_f_score:
            min_f_score = f_score_print
            min_f_score_date = date
            min_burn_name = burnName


        # print(didBurn)
        # print(shouldBeBurnt)
        print(date, " fire: ", burnName)
        print(date , " precision: ", precision)
        print(date , " recall: ", recall)
        print(date , " F score: ", f_score_print)
        #This is the percent of pixels that actually burned that the model was correct on (doesn't count what
        #the model said would burn that didnt)
        print(date , " percent model thinks burned that actually did: ", burnAcc,  " %")
        print(date , " accuracy correct: ", accCor,  " %")
        print(date , " accuracy incorrect: ", accIncor, " %")
        print(date , " percent fire grew: ", perc_burned)
        print("TOTAL F SCORE AND PERIM NUM: ", total_f_score, " ", perim_num)

    #print avg f_score
    try:
        avg_f_score = float("%0.4f" % (total_f_score/perim_num))
    except:
        avg_f_score = 0


#fun exercise would be to do quicksort on a list/dict of these and print the whole thing out to see which fires are better

    print("AVERAGE recall: ", total_recall/perim_num)
    print("AVERAGE percision: ", total_precision/perim_num)

    print("AVERAGE F SCORE: ", avg_f_score)
    print("Maximum F-score: ", max_f_score)
    print("Maximum F-score date: ", max_f_score_date)
    print("Maximum F-score fire: ", max_burn_name)
    print("Minimum F-score: ", min_f_score)
    print("Minimum F-score date: ", min_f_score_date)
    print("Minimum F-score fire: ", min_burn_name)


#the dataset for this function MUST be made of two times (4 perims in _untrained) where the 2 pairs dont have to be adjacent
#but the perimeters inside the pairs do (e.g. 0716, 0717, 0801, 0802)
def getNumbersNonConsecutive(dataset, loc, res, size, fireDate):
    loopNum = 0
    initial_perim = 0
    new_burn = 0
    for burnName, date in dataset.getUsedBurnNamesAndDates():
        print("Starting loop number ", loopNum)
        loopNum = loopNum + 1
        burn = dataset.data.burns[burnName]
        day = dataset.data.getDay(burnName, date)
        h,w = day.startingPerim.shape
        normedDEM = util.normalize(burn.layers['dem'])
        numOfFire = len(dataset.getUsedBurnNamesAndDates())
        loopIt = int(size/numOfFire)


        for i in range(0, h):
            for j in range(0, w):
                if loopNum == 1:
                    if day.startingPerim.astype(np.uint8)[i][j] == 1:
                        initial_perim = initial_perim + 1
                if loopNum == 2:
                    if day.endingPerim.astype(np.uint8)[i][j] == 1:
                        new_burn = new_burn + 1


        if loopNum == 2:
            new_burn = new_burn - initial_perim
            perc_burned = new_burn / initial_perim
            print("new burn: ", new_burn)
            print("initial perim: ", initial_perim)
            print("day: ", day)

        if loopNum == 1:
            curDay = []
            curRes = []
            pixelsForDate = 0
            while(len(fireDate) > 0 and date == fireDate[0]):
                curDay.append(loc[0])
                # if rand is True:
                #     curRes.append(random.uniform(0.0, 1.0))
                # else:
                curRes.append(res[0])
                loc.pop(0)
                res.pop(0)
                fireDate.pop(0)

            posx = []
            posy = []

            for tup in curDay:
                posx.append(tup[1])
                posy.append(tup[0])

        correct = 0
        incorrect = 0
        shouldBeBurnt = 0
        didBurn = 0
        TP=0
        FP=0
        TN=0
        FN=0


        if loopNum == 2:
            for pos in range(0, len(posx)):
                if day.endingPerim.astype(np.uint8)[posy[pos]][posx[pos]] == 1:
                    shouldBeBurnt = shouldBeBurnt + 1
                    if curRes[pos] > 0.5:
                        didBurn = didBurn + 1
                if curRes[pos] > 0.5 and day.endingPerim.astype(np.uint8)[posy[pos]][posx[pos]] == 1:
                    correct = correct + 1
                    TP = TP + 1
                elif curRes[pos] < 0.5 and day.endingPerim.astype(np.uint8)[posy[pos]][posx[pos]] == 0:
                    correct = correct + 1
                    TN = TN + 1
                elif curRes[pos] < 0.5 and day.endingPerim.astype(np.uint8)[posy[pos]][posx[pos]] == 1:
                    FN = FN + 1
                    incorrect = incorrect + 1
                elif curRes[pos] > 0.5 and day.endingPerim.astype(np.uint8)[posy[pos]][posx[pos]] == 0:
                    FP = FP + 1
                    incorrect = incorrect + 1
                # else:
                #     incorrect = incorrect + 1


            try:
                precision = (TP/(TP + FP))
                recall = (TP/(TP + FN))
                f_frac = ((precision * recall)/(precision + recall))
                f_score = 2 * f_frac
            except:
                print("ERROR: Not enough points to calculate F-score")
                precision = 0
                recall = 0
                f_score = 0

            print()
            print(date, " Number of pixels tested " , len(posx))


            try:
                burnAcc = float("%0.2f" % ((didBurn/shouldBeBurnt) * 100))
            except:
                burnAcc = 0
                print("ERROR: No pixels tested are inside of the next fire perimeter")

            f_score_print = float("%0.4f" % f_score)
            accCor = float("%0.2f" % ((correct/len(posx)) * 100))
            accIncor = float("%0.2f" % ((incorrect/len(posx)) * 100))


            print(date , " precision: ", precision)
            print(date , " recall: ", recall)
            print(date , " F score: ", f_score_print)
            print(date , " percent model thinks burned that actually did: ", burnAcc,  " %")
            print(date , " accuracy correct: ", accCor,  " %")
            print(date , " accuracy incorrect: ", accIncor, " %")
            print(date , " percent fire grew: ", perc_burned)





def createCanvases(dataset):
    result = {}
    for burnName, date in dataset.getUsedBurnNamesAndDates():
        burn = dataset.data.burns[burnName]
        day = dataset.data.getDay(burnName, date)
        h,w = day.startingPerim.shape
        normedDEM = util.normalize(burn.layers['dem'])
        canvas = cv2.cvtColor(normedDEM, cv2.COLOR_GRAY2RGB)
        # coun = 0
        # for y in day.endingPerim.astype(np.uint8):
        #     cou = 0
        #     for x in y:
        #         print("this is y:", coun, " x:", cou , " " , x)
        #         cou = cou + 1
        #     coun = coun + 1

        im2, startContour, hierarchy = cv2.findContours(day.startingPerim.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        im2, endContour, heirarchy = cv2.findContours(day.endingPerim.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        #the code below puts the perimeters on the visual
        cv2.drawContours(canvas, endContour, -1, (0,0,1), 1)
        cv2.drawContours(canvas, startContour, -1, (0,1,0), 1)

        result[(burnName, date)] = canvas
    return result

def overlay(predictionRenders, canvases):
    result = {}
    for burnName, date in sorted(canvases):
        canvas = canvases[(burnName, date)].copy()
        render = predictionRenders[(burnName, date)]
        yellowToRed = np.dstack((np.ones_like(render), 1-(render-1), np.zeros_like(render)))
        canvas[render>1] = yellowToRed[render>1]
        result[(burnName, date)] = canvas
    return result

def visualizePredictions(dataset, predictions, preResu):
    predRenders = renderPredictions(dataset, predictions, preResu)
    canvases = createCanvases(dataset)
    overlayed = overlay(predRenders, canvases)
    return overlayed

def showPredictions(predictionsRenders):
    # sort by burn
    burns = {}
    for (burnName, date), render in predictionsRenders.items():
        if burnName not in burns:
            burns[burnName] = []
        burns[burnName].append((date, render))

    for burnName, frameList in burns.items():
        frameList.sort()
        fig = plt.figure(burnName, figsize=(8, 6))
        ims = []
        pos = (30,30)
        color = (0,0,1.0)
        size = 1
        thickness = 2
        for date, render in frameList:
            withTitle = render.copy()
            cv2.putText(withTitle,date, pos, cv2.FONT_HERSHEY_SIMPLEX, size, color, thickness=thickness)
            im = plt.imshow(withTitle)
            ims.append([im])
        anim = animation.ArtistAnimation(fig, ims, interval=300, blit=True,
                                repeat_delay=0)

        def createMyOnKey(anim):
            def onKey(event):
                if event.key == 'right':
                    anim._step()
                elif event.key == 'left':
                    saved = anim._draw_next_frame
                    def dummy(a,b):
                        pass
                    anim._draw_next_frame = dummy
                    for i in range(len(anim._framedata)-2):
                        anim._step()
                    anim._draw_next_frame = saved
                    anim._step()
                elif event.key =='down':
                    anim.event_source.stop()
                elif event.key =='up':
                    anim.event_source.start()
            return onKey

        fig.canvas.mpl_connect('key_press_event', createMyOnKey(anim))
        plt.show()


def show(*imgs, imm=True):
    try:
        for i, img in enumerate(imgs):
            plt.figure(i, figsize=(8, 6))
            plt.imshow(img)
        if imm:
            plt.show()
    except:
        print("Not able to show because plt not imported")

def save(img, name):
    fname = 'output/imgs/{}.png'.format(name)
    cv2.imwrite(fname, img)

def saveModel(model):
    from keras.utils import plot_model
    timeString = strftime("%d%b%H:%M", localtime())
    fname = 'output/modelViz/{}.png'.format(timeString)
    plot_model(model, to_file=fname, show_shapes=True)
