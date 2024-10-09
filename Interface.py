import numpy as np
import pandas as pd
import dataPreProcess as dpp
import tenfoldCrossValidation as tfc
import KNN
import matplotlib.pyplot as plt

#file = r"C:\Users\tyler\OneDrive\Documents\GitHub\MachineLearningKNearestNeighbor\data\forestfires.data"
#file = r"enter you file path here and uncomment"


cf = tfc.tenfoldCrossValidation()
dataPre = dpp.dataPreProcess()

X,y = dataPre.getData(file)
#plt.scatter(X[:,4], y)
#plt.show()
X,y = dataPre.stratify(X=X,y=y,regression=True)

model = KNN.KNN
task = "regression"
defaultVal = [3] #ex for 2 different hyperparameters use [3, .1] as an example

hyperparamArr = np.array(defaultVal)
hyperparamArr = np.atleast_1d(hyperparamArr)

hyperParams = cf.changeHyperparamaters(hyperparamArr)

hyperParams = np.atleast_1d(hyperParams)


bestHypers = cf.HyperTest(model,hyperParams,X,y, task)

result = cf.tenFoldCrossFinal(model, bestHypers, X,y, task)


