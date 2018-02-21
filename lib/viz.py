from time import localtime, strftime
import csv

import numpy as np
import cv2
try:
    import matplotlib
    matplotlib.use("TkAgg")
    from matplotlib import pyplot as plt
    import matplotlib.animation as animation
    print('Successfully imported pyplot')
except:
    print('Failed to import pyplot ')

from lib import dataset
from lib import util

def renderDataset(dataset):
    pass

def renderUsedPixels(dataset, burnName, date):
    # burnName, date = day.burn.name, day.date
    mask = dataset.points[burnName][date]
    # bg = day.burn.layers['dem']
    # background = cv2.merge((bg,bg,bg))
    return mask*127


def renderPredictions(dataset, predictions):
    # print('predictions are', predictions.values())
    day2pred = {}
    for pt, pred in predictions.items():
        burnName, date, location = pt
        day = (burnName, date)
        if day not in day2pred:
            day2pred[day] = []
        pair = (location, float(pred))
        # print('storing prediction', pair)
        day2pred[day].append(pair)

    # print('these are all the original days:', day2pred.keys())
    results = {}
    for (burnName, date), locsAndPreds in day2pred.items():
        # print('locs and preds', locsAndPreds)
        locs, preds = zip(*locsAndPreds)
        # print('reds:', preds)
        xs,ys = zip(*locs)
        preds = [pred+1 for pred in preds]
        # print((xs,ys))
        # print(max(preds), min(preds))
        # print(len(xs), len(preds))
        burn = dataset.data.burns[burnName]
        canvas = np.zeros(burn.layerSize, dtype=np.float32)
        # print(canvas)
        canvas[(xs,ys)] = np.array(preds, dtype=np.float32)
        results[(burnName, date)] = canvas
    return results

def createCanvases(dataset):
    result = {}
    for burnName, date in dataset.getUsedBurnNamesAndDates():
        burn = dataset.data.burns[burnName]
        day = dataset.data.getDay(burnName, date)
        h,w = day.startingPerim.shape
        # canvas = np.zeros((h,w,3), dtype=np.uint8)
        normedDEM = util.normalize(burn.layers['dem'])
        canvas = cv2.cvtColor(normedDEM, cv2.COLOR_GRAY2RGB)

        im2, startContour, hierarchy = cv2.findContours(day.startingPerim.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        im2, endContour, heirarchy = cv2.findContours(day.endingPerim.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(canvas, endContour, -1, (0,0,1), 1)
        cv2.drawContours(canvas, startContour, -1, (0,1,0), 1)

        result[(burnName, date)] = canvas

        # plt.imshow(canvas)
        # plt.show()
    return result

def overlay(predictionRenders, canvases):
    result = {}
    for burnName, date in sorted(canvases):
        canvas = canvases[(burnName, date)].copy()
        render = predictionRenders[(burnName, date)]
        yellowToRed = np.dstack((np.ones_like(render), 1-(render-1), np.zeros_like(render)))
        canvas[render>1] = yellowToRed[render>1]
        result[(burnName, date)] = canvas

        # plt.imshow(canvases[(burnName, date)])
        # plt.figure('render')
        # plt.imshow(render)
        # plt.figure(burnName +' '+date)
        # plt.imshow(canvas)
        # plt.show()
    return result

def visualizePredictions(dataset, predictions):
    # print('these are all the burns Im going to start rendering:', predictions.keys())
    predRenders = renderPredictions(dataset, predictions)
    canvases = createCanvases(dataset)
    overlayed = overlay(predRenders, canvases)
    return overlayed

def showPredictions(predictionsRenders):
    # sort by burn
    # print("Here are all the renders:", predictionsRenders.keys())
    burns = {}
    for (burnName, date), render in predictionsRenders.items():
        if burnName not in burns:
            burns[burnName] = []
        burns[burnName].append((date, render))

    # isRunning = {}
    # print("These are all the burns I'm showing:", burns.keys())
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
                    # print(success)
                    # if not success:
                    #     anim.frame_seq = anim.new_frame_seq()
                    #     anim._step()
                elif event.key =='down':
                    anim.event_source.stop()
                elif event.key =='up':
                    anim.event_source.start()
            return onKey

        # fig.canvas.mpl_connect('button_press_event', onClick)
        fig.canvas.mpl_connect('key_press_event', createMyOnKey(anim))
        plt.show()


    # anim = animation.FuncAnimation(fig, animfunc[,..other args])

    #pause
    # anim.event_source.stop()
    #
    # #unpause
    # anim.event_source.start()

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

def saveModelDiagram(model):
    from keras.utils import plot_model
    timeString = strftime("%d%b%H:%M", localtime())
    fname = 'output/modelViz/{}.png'.format(timeString)
    plot_model(model, to_file=fname, show_shapes=True)
