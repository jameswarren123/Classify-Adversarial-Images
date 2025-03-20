import numpy as np
import pandas as pd
import pickle
import time
from model import SimpleNN

mnist_mean = 0.1307
mnist_std = 0.3081

def normalize(X):
    X = X.astype(np.float32) / 255.0
    X = (X - mnist_mean) / mnist_std
    return X
def unnormalize(X):
    X = (X * mnist_std) + mnist_mean
    X *= 255
    return np.clip(X, 0, 255).astype(np.uint8)

def dfun(data, model):
    failsToMisclassify = 0
    startsMisclassified = 0
    retData = data.copy(deep=True)
    pixel_columns = retData.columns[1:]
    retData[pixel_columns] = retData[pixel_columns].astype(np.float32)
    
    for i in range(len(data)):
        x = normalize(data.iloc[i, 1:].values)
        y = np.array([int(data.iloc[i, 0])])
        probabilities = model.predict(x)
        firstLabel = np.argmax(probabilities)
        
        if(firstLabel != y):
            startsMisclassified += 1
            retData.iloc[i, 0] = y
            retData.iloc[i, 1:] = unnormalize(x)
            continue
        
        iteration = 0
        w = [[] for _ in range(10)]
        f = np.zeros(10)
        curClass = np.argmax(probabilities)
        k = 0
        max_iterations = 200
        while (curClass == y[0] and iteration < max_iterations):
            for i in range(10):
                if i != y[0]:
                    print(model.gradient(x, np.array([curClass])))
                    print(model.gradient(x, y))
                    w[i] = model.gradient(x, np.array([curClass])) - model.gradient(x, y)
                    print(w[i])
                    f[i] = probabilities[0, i] - probabilities[0, (y[0])]
            if 0 != y[0]:
                l = abs(f[0])/np.linalg.norm(w[0])
                k = 0
            else:
                l = abs(f[1])/np.linalg.norm(w[1])
                k = 1
            for i in range(10):
                if i != y[0]:
                    temp = abs(f[i])/np.linalg.norm(w[i])
                    if temp < l:
                        l = temp
                        k = i
            r = abs(f[k]) / np.linalg.norm(w[k]) * w[k]
            x = x + r
            iteration += 1
            probabilities = model.predict(x)
            curClass = np.argmax(probabilities)
                        
            
        if iteration >= max_iterations:
            failsToMisclassify += 1
        if iteration == 0:
            startsMisclassified += 1
        
        retData.iloc[i, 0] = y[0]
        retData.iloc[i, 1:] = unnormalize(x)
        
    print(f"Number failed to misclassify: {failsToMisclassify}")
    print(f"Number started miscalssified: {startsMisclassified}")
    retData[pixel_columns] = retData[pixel_columns].astype(np.uint8)
    
    return retData

try:
    with open('mnist_test.pkl', 'rb') as fid:
        mnist_test = pickle.load(fid)
    with open('model.pkl', 'rb') as fid:
        model = pickle.load(fid)
    
    print("training data")
    start_time = time.time()
    DeepFoolUntargeted = dfun(mnist_test[800:801], model)
    end_time = time.time()
    print(f"Create training data execution time: {end_time - start_time:.2f} seconds")
    DeepFoolUntargeted.to_pickle('dfun_train.pkl')
    
    print("testing data")
    start_time = time.time()
    DeepFoolUntargeted = dfun(mnist_test[6600:6601], model)
    end_time = time.time()
    print(f"Create testing data execution time: {end_time - start_time:.2f} seconds")
    DeepFoolUntargeted.to_pickle('dfun_test.pkl')
except FileNotFoundError as e:
    print(f"Error loading files: {e}")
except Exception as e:
    print(f"Error processing data: {e}")