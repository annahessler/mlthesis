
from inspect import signature
from PyQt5 import QtCore

def async(func=None, callback=None):
    '''Decorator for making functions asynchronous, with or without callbacks.

    used as:
    def dealWithResults(result):
        print(result)

    @async(callback=dealWithResults)
    def workerWithResult():
        doWork()
        return 42

    OR

    @async
    def workerWithoutResult():
        doWork()

    '''
    def decorator(func):
        class Runner(QtCore.QThread):
            # we have to ensure the pyqtsignal uses the same arguments that the
            # callback function expects
            if callback:
                sig = signature(callback)
                callbackArgTypes = tuple([object] * len(sig.parameters))
                mySignal = QtCore.pyqtSignal(*callbackArgTypes,name="mySignal")

            def __init__(self, target, *args, **kwargs):
                super().__init__()
                self._target = target
                self._args = args
                self._kwargs = kwargs

            def run(self):
                result = self._target(*self._args, **self._kwargs)
                if callback:
                    self.mySignal.emit(*result)

        def inner(*args, **kwargs):
            runner = Runner(func, *args, **kwargs)
            # Keep the runner somewhere or it will be garbage collected
            func.__runner = runner
            if callback:
                runner.mySignal.connect(callback)
            runner.start()
        return inner
    if func is not None:
        return decorator(func)
    return decorator
