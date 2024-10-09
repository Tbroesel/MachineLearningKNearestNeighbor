import numpy as np
import pandas as pd

class dataPreProcess(object):


    def __init__(self):
        X = None
        y = None
        tuningArray = None


    def getData(self, filePath):

        data = pd.read_csv(filePath, header = None)

        for row, element in enumerate(data.iloc[0]):                 # go through all the first row of data

            if(isinstance(element, str)):                            # if the element is a string or subclass of string
                data[row] = pd.factorize(data[row])[0]               # we will make the entire column into 0s (ex {'a', 'b', 'c', 'a'} -> {0, 1, 2, 0})

        X = pd.DataFrame(data)

        X = X.sample(frac = 1)                                      # shuffle the data

        y = X[X.columns[-1]]                                        # move labels to y
        X.drop([X.columns[-1]], axis = 1, inplace = True)           # remove labels from X

        X = X.to_numpy()
        y = y.to_numpy()
        return X,y


    def stratify(self, X, y, regression=False, tuningParition=True, tuningArray=None):


        

        if(not regression):
            if(tuningParition):
                X = np.array_split(X,10)                                 # evenly splits the data into ten partitions (1 less normal partition for tuning partition)
                y = np.array_split(y,10)
                tuningArray = [X[9],y[9]]                                # makes tuningPartition instead of tenth partition
           
            else:
                X = np.array_split(X,10)                                 # evenly splits the data into ten partitions
                y = np.array_split(y,10)
        
        else:                                                            #is regression
            
            X = np.hstack((X,y[:, None]))  
            t = X[1]                                      #re-add the response data to last column
            print(t)
            print(X)
            X = X[X[:, -1].argsort()]                                          #sort the data by the last column (sort data by response)
            print(X)
            numRows, numCols = X.shape
            numCols -= 1
            newXArr = np.array([[np.empty((numCols))],[np.empty((numCols))],[np.empty((numCols))],[np.empty((numCols))],[np.empty((numCols))],[np.empty((numCols))],[np.empty((numCols))],[np.empty((numCols))],[np.empty((numCols))],[np.empty((numCols))]])                        # make an empty array that will be used for the 10 data points
            newYArr = np.array([[np.empty((1))],[np.empty((1))],[np.empty((1))],[np.empty((1))],[np.empty((1))],[np.empty((1))],[np.empty((1))],[np.empty((1))],[np.empty((1))],[np.empty((1))]])                        # make an empty array that will be used for the 10 data points

            #I tried the way above but no matter what I did i would get a error when I would try to vstack inside the array (when I would newYArr[count] = np.vstack((newYArr[count], newY[row]))) i spent more then 3 hours on it best to move

            Y1,Y2,Y3,Y4,Y5,Y6,Y7,Y8,Y9,Y10 = np.empty((1)),np.empty((1)),np.empty((1)),np.empty((1)),np.empty((1)),np.empty((1)),np.empty((1)),np.empty((1)),np.empty((1)),np.empty((1))
            X1,X2,X3,X4,X5,X6,X7,X8,X9,X10 = np.empty((numCols)),np.empty((numCols)),np.empty((numCols)),np.empty((numCols)),np.empty((numCols)),np.empty((numCols)),np.empty((numCols)),np.empty((numCols)),np.empty((numCols)),np.empty((numCols))

            tempX = np.split(X, np.where(np.diff(X[:,-1]))[0]+1)          #splits the data when the response info changes(i.e. a new response type) | returns a list of np arrays
            
            lenX = len(newXArr)                                             #number of groups of different response
            
            count = 0
            
            first = True
            
            for tempVal, array in enumerate(tempX):
                
                arrayRow, tempArrY = array.shape
                newX = array
                newY, newX = newX[:,-1], np.delete(newX, -1, axis=1)                                 # move labels to y
                
                


                for row in range(arrayRow):
                    

                    if (count % 10 == 0) and (count != 0):                                  #If count is a multiple of ten
                       
                        count = 0
                        
                        if(first):
                            
                            first = False
                    
                    if(first):
                        match count:
                            case 1:
                                Y1 = newY[row]
                                X1 = newX[row]
                            case 2:
                                Y2 = newY[row]
                                X2 = newX[row]
                            case 3:
                                Y3 = newY[row]
                                X3 = newX[row]
                            case 4:
                                Y4 = newY[row]
                                X4 = newX[row]    
                            case 5:
                                Y5 = newY[row]
                                X5 = newX[row]
                            case 6:
                                Y6 = newY[row]
                                X6 = newX[row]
                            case 7:
                                Y7 = newY[row]
                                X7 = newX[row]
                            case 8:
                                Y8 = newY[row]
                                X8 = newX[row]
                            case 9:
                                Y9 = newY[row]
                                X9 = newX[row]
                            case 10:
                                Y10 = newY[row]
                                X10 = newX[row]
                                                       #change the first value to row

                        
                    
                    else:
                        testY = np.vstack((newYArr[count], newY[row]))
                        match count:
                            case 1:
                                Y1 = np.vstack((Y1, newY[row]))
                                X1 = np.vstack((X1, newX[row]))
                            case 2:
                                Y2 = np.vstack((Y2, newY[row]))
                                X2 = np.vstack((X2, newX[row]))
                            case 3:
                                Y3 = np.vstack((Y3, newY[row]))
                                X3 = np.vstack((X3, newX[row]))
                            case 4:
                                Y4 = np.vstack((Y4, newY[row]))
                                X4 = np.vstack((X4, newX[row]))    
                            case 5:
                                Y5 = np.vstack((Y5, newY[row]))
                                X5 = np.vstack((X5, newX[row]))
                            case 6:
                                Y6 = np.vstack((Y6, newY[row]))
                                X6 = np.vstack((X6, newX[row]))
                            case 7:
                                Y7 = np.vstack((Y7, newY[row]))
                                X7 = np.vstack((X7, newX[row]))
                            case 8:
                                Y8 = np.vstack((Y8, newY[row]))
                                X8 = np.vstack((X8, newX[row]))
                            case 9:
                                Y9 = np.vstack((Y9, newY[row]))
                                X9 = np.vstack((X9, newX[row]))
                            case 10:
                                Y10 = np.vstack((Y10, newY[row]))
                                X10 = np.vstack((X10, newX[row]))
                
                    count += 1
            X = np.array((X1,X2,X3,X4,X5,X6,X7,X8,X9,X10), dtype=object)
            y = np.array((Y1,Y2,Y3,Y4,Y5,Y6,Y7,Y8,Y9,Y10), dtype=object)

        return X, y


            