# preprocess.py
import numpy as np

def mergeSamples(samples):
    weather, spatial, outputs = [], [], []
    for sample in samples:
        w, s = sample.getInputs()
        weather.append(w)
        spatial.append(s)
        outputs.append(sample.getOutput())
    return [np.array(weather), np.array(spatial)], np.array(outputs)

def getInputsAndOutputs(dataset, inputSettings):
    # get the weather data for each day
    samples = dataset.getSamples(inputSettings)
    return mergeSamples(samples)
