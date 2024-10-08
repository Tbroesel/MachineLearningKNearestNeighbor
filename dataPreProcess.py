import numpy as np
import pandas as pd

class dataPreProcess:

    def __init__(self):
        X = None
        y = None
        tuningArray = None


    def getData(filePath):

        data = pd.read_csv(filePath, header = None)

        for row, element in enumerate(data.iloc[0]):                 # go through all the first row of data

            if(isinstance(element, str)):                            # if the element is a string or subclass of string
                data[row] = pd.factorize(data[row])[0]               # we will make the entire column into 0s (ex {'a', 'b', 'c', 'a'} -> {0, 1, 2, 0})

        X = pd.DataFrame(data)

        X = X.sample(frac = 1)                                      # shuffle the data

        y = X[X.columns[-1]]                                        # move labels to y
        X.drop([X.columns[-1]], axis = 1, inplace = True)           # remove labels from X

        return X, y


    def stratify(X, y, regression=False, tuningParition=True, tuningArray=None):

        if(tuningParition):
            np.array_split(X,10)                                     # evenly splits the data into ten partitions (1 less normal partition for tuning partition)
            np.array_split(y,10)
            tuningArray = [X[9],y[9]]                                # makes tuningPartition instead of tenth partition
        else:
            np.array_split(X,10)                                     # evenly splits the data into ten partitions
            np.array_split(y,10)