import numpy as np

def precision(y_true, y_pred):
        return float(sum(y_pred == y_true) / (sum(y_pred == y_true) + sum(y_pred != y_true)))
def recall(y_true, y_pred):
        return float(sum(y_pred == y_true) / (sum(y_pred == y_true) + sum(y_true != y_pred)))
def f1_score(y_true, y_pred):
    return float(2 * ((precision(y_true, y_pred) * recall(y_true, y_pred)) / (precision(y_true, y_pred) + recall(y_true, y_pred))) * 100)
def l2Error(y_true, y_pred):
    return np.sqrt(np.sum(np.power((y_true-y_pred),2)))

class tenfoldCrossValidation:

    def changeHyperparamaters(self, defaultArray):    #an array of hyper paramaters as well as the array of what those default values are
        hyperArray = np.empty(10)
        full = False
        for count, defaultVal in enumerate(defaultArray):
            
                if min(0, .99) <= defaultVal <= max(0, .99):
                    val = defaultVal
                    count = 0
                    step = .1
                    diff = round(.99 - defaultVal)
                    
                    for num in range(diff):         #every step that will be adding as a hyperparameter
                        val += step
                        hyperArray[count] = val
                        count+=1
                    
                    val = defaultVal
                    count += 1
                    hyperArray[count] = val


                    for num in range(10 - count):
                        val -= step
                        hyperArray[count] = val
                        count+=1


                elif min(1, 9.9) <= defaultVal <= max(1, 9.9):
                    val = defaultVal
                    count = 0
                    step = 1
                    diff = round(9.9 - defaultVal)
                    
                    for num in range(diff):         #every step that will be adding as a hyperparameter
                        val += step
                        hyperArray[count] = val
                        count+=1
                    
                    val = defaultVal
                    
                    hyperArray[count] = val

                    count += 1
                    for num in range(10 - count):
                        val -= step
                        hyperArray[count] = val
                        count+=1

                elif min(10, 99) <= defaultVal <= max(10, 99):
                    val = defaultVal
                    count = 0
                    step = 10
                    diff = round(99 - defaultVal)
                    
                    for num in range(diff):         #every step that will be adding as a hyperparameter
                        val += step
                        hyperArray[count] = val
                        count+=1
                    
                    val = defaultVal
                    count += 1
                    hyperArray[count] = val


                    for num in range(10 - count):
                        val -= step
                        hyperArray[count] = val
                        count+=1

        if(not full):
                fullHyperArray = hyperArray
                full = True
        else:
                fullHyperArray = np.append(fullHyperArray, hyperArray)

                
        return fullHyperArray



    def HyperTest(self, model, fullHyperArray, X,y, task="classification"):          # THIS WILL NEED TO BE CHANGED BETWEEN MODELS
        x = np.ndim(fullHyperArray)
        bestRunScore = 0
        
        if(x == 1):          # this is what will have to change
                for run, element in enumerate(fullHyperArray):
                    hyper = np.atleast_1d(np.array(element))
                    run = self.tenFold(model, hyper, X,y, task)
                    if run > bestRunScore:
                         bestRun = hyper
                         bestRunScore = run
                         
        if(x == 2):
                count = 0
                for index, element in np.ndenumerate(fullHyperArray[0]):
                     for secIndex, secElement in np.ndenumerate(fullHyperArray[1]):
                        hyper = np.array(element, secElement)
                        run = self.tenFold(model, hyper, X,y, task)
                        if run > bestRunScore:
                            bestRun = hyper
                            bestRunScore = run
        return bestRun
    
    def tenFold(self, model, hypers, X,y, task="classification"):
         result = 0
         if hypers.ndim == 1:
              for run in range(10):
                    X_test = X[run]
                    X_train = np.delete(X, run)
                    y_test = y[run]
                    y_train = np.delete(y, run)

                    modelClass = model(hypers[0])
                    modelClass.fit(X_train, y_train)

                    if(task == "regression"): 
                        y_pred = modelClass.predict(X_test, task)
                        result += l2Error(y_test, y_pred)
                        
                    else:
                         y_pred = model.predict(X_test, task)
                         result += f1_score(y_test,y_pred)
         else:
              count = 0
              for run in range(10):
                        X_test = X[count]
                        X_train = np.delete(X, count)
                        y_test = y[count]
                        y_train = np.delete(y, count)

                        modelClass = model(hypers[0], hypers[1])
                        modelClass.fit(X_train, y_train)

                        if(task == "regression"): 
                            y_pred = modelClass.predict(X_test, task)
                            result += l2Error(y_test, y_pred)
                           
                        else:
                            y_pred = model.predict(X_test, task)
                            result += f1_score(y_test,y_pred)
         result / 10.
         return result
                            
    def tenFoldCrossFinal(self, model,bestHypers, X,y,  task="classification"):
        x = np.ndim(bestHypers)
        if(x == 1):          # this is what will have to change
                    hyper = np.atleast_1d(np.array(bestHypers[0]))
                    run = self.tenFold(model, hyper, X,y, task)
                    
                         
        if(x == 2):
           
                        hyper = np.array(bestHypers[0], bestHypers[1])
                        run = self.tenFold(model, hyper, X,y, task)
                        
        return run   
            

            
