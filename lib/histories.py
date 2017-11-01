import keras
from sklearn.metrics import roc_auc_score
from lib import viz

class Histories(keras.callbacks.Callback):

    def __init__(self, validateData):
        self.vd = validateData
        # self.testdata = testData
        self.losses = []
        self.predictions = []
        self.i = 0
        self.save_every = 1
        

    def on_train_begin(self, logs={}):
        # self.aucs = []
        # self.losses = []
        # self.losses = []
        # self.predictions = []
        # self.i = 0
        # self.save_every = 1
        return

    def on_train_end(self, logs={}):
        return

    def on_epoch_begin(self, epoch, logs={}):
        return

    def on_epoch_end(self, epoch, logs={}):
        # self.losses.append(logs.get('loss'))
        # inputs, output  = self.vd.getData()
        # y_pred = self.model.predict(self.vd)
        # self.aucs.append(roc_auc_score(output, y_pred))
        return

    def on_batch_begin(self, batch, logs={}):
        return

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))
        self.i += 1        
        if self.i % self.save_every == 0:        
            pred = self.model.predict(self.vd)
            self.predictions.append(pred)
            res = viz.visualizePredictions(self.vd, pred)
            viz.save(res, 'precictions at ' +  str(self.i))
            # print("history losses are this " , self.losses)




# def on_train_begin(self, logs={}):
#         self.losses = []
#         self.predictions = []
#         self.i = 0
#         self.save_every = 50

#     def on_batch_end(self, batch, logs={}):
#         self.losses.append(logs.get('loss'))
#         self.i += 1        
#         if self.i % self.save_every == 0:        
#             pred = model.predict(X_train)
#             self.predictions.append(pred)


