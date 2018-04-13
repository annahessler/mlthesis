import keras
from sklearn.metrics import roc_auc_score
import matplotlib as mpl
mpl.use('Agg')
from lib import viz
from matplotlib import animation
import matplotlib.pyplot as plt

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
        self.losses.append(logs.get('loss'))
        fig1 = plt.figure()
        self.i += 1
        if self.i % self.save_every == 0:
            pred = self.model.predict(self.td)
            self.predictions.append(pred)

    def on_batch_begin(self, batch, logs={}):
        return

    def on_batch_end(self, batch, logs={}):
        return
