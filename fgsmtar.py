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

def fgsmtar(data, model):
    failsToMisclassify = 0
    startsMisclassified = 0
    retData = data.copy(deep=True)
    pixel_columns = retData.columns[1:]
    retData[pixel_columns] = retData[pixel_columns].astype(np.float32)
    totalIterations = 0
    
    for i in range(len(data)):
        x = normalize(data.iloc[i, 1:].values)
        y = np.array([int(data.iloc[i, 0])])
        
        probabilities = model.predict(x)
        epsilon = 0.0075
        max_iterations = 200
        iteration = 0
        
        while (np.argmax(probabilities) != 0 and iteration < max_iterations):
            gradients = model.gradient(x, np.array([0]), True)
            if gradients is None:
                break
            x = x - epsilon * np.sign(gradients)
            probabilities = model.predict(normalize(unnormalize(x)))
            iteration += 1
        
        if iteration >= max_iterations:
            failsToMisclassify += 1
        if iteration == 0:
            startsMisclassified += 1
        totalIterations += iteration
        
        retData.iloc[i, 0] = y[0]
        retData.iloc[i, 1:] = unnormalize(x)
    
    print(f"Average iterations to end: {totalIterations/len(data)}")
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
    FGSMtargeted = fgsmtar(mnist_test[400:800], model)
    end_time = time.time()
    print(f"Create training data execution time: {end_time - start_time:.2f} seconds")
    FGSMtargeted.to_pickle('fgsmtar_train.pkl')
    
    print("testing data")
    start_time = time.time()
    FGSMtargeted = fgsmtar(mnist_test[6500:6600], model)
    end_time = time.time()
    print(f"Create testing data execution time: {end_time - start_time:.2f} seconds")
    FGSMtargeted.to_pickle('fgsmtar_test.pkl')
except FileNotFoundError as e:
    print(f"Error loading files: {e}")
except Exception as e:
    print(f"Error processing data: {e}")