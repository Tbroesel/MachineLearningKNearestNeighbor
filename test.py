import numpy as np
import pandas as pd

t = pd.read_csv(r"C:\Users\tyler\OneDrive\Documents\GitHub\MachineLearningKNearestNeighbor\data\forestfires.data", header=None)
print(t)

for row, element in enumerate(t.iloc[0]):
    if(isinstance(element, str)):
        t[row] = pd.factorize(t[row])[0]

print(t)