import keras
from sklearn.metrics import roc_auc_score
from hottopic import viz
import matplotlib
matplotlib.use("TkAgg")
from matplotlib import animation
from matplotlib import pyplot as plt

class Histories(keras.callbacks.Callback):

    def __init__(self, test):
        self.td = test
        self.losses = []
        self.predictions = []
        self.i = 0
        self.save_every = 1
        self.images = []
        self.toanimate = plt.plot([], 'ro', animated=True)



    def on_train_begin(self, logs={}):
        return

    def on_train_end(self, logs={}):
        return

    def on_epoch_begin(self, epoch, logs={}):
        return

    def on_epoch_end(self, epoch, logs={}):
        # return
        self.losses.append(logs.get('loss'))
        fig1 = plt.figure()
        self.i += 1
        if self.i % self.save_every == 0:
            pred = self.model.predict(self.td)
            self.predictions.append(pred)
            # res = viz.visualizePredictions(self.td, pred)
            # self.images.append(res)     doesnt work due to data change in shape
            # self.toanimate.set_data(self.images)
            # update(self.images, res)
            # viz.save(res, 'precictions_at_' +  str(self.i))    need for predictions

    def on_batch_begin(self, batch, logs={}):
        return

    def on_batch_end(self, batch, logs={}):
        # self.losses.append(logs.get('loss'))
        # fig1 = plt.figure()
        # self.i += 1
        # if self.i % self.save_every == 0:
        #     pred = self.model.predict(self.vd)
        #     self.predictions.append(pred)
        #     res = viz.visualizePredictions(self.vd, pred)
        #     self.images.append(res)
        #     # self.toanimate.set_data(self.images)
        #     # update(self.images, res)
        #     viz.save(res, 'precictions at ' +  str(self.i))
        # print("images are ", self.images)
        # viz.createGif(self.i)
        return


#     def on_batch_end(self, batch, logs={}):
#         self.losses.append(logs.get('loss'))
#         self.i += 1
#         if self.i % self.save_every == 0:
#             pred = model.predict(X_train)
#             self.predictions.append(pred)
