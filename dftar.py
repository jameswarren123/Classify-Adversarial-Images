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
    return X.reshape(-1, 1, 28, 28)
def unnormalize(X):
    X = (X * mnist_std) + mnist_mean
    X *= 255
    return np.clip(X, 0, 255).astype(np.uint8).reshape(784,)

def dftar(data, model):
    failsToMisclassify = 0
    startsMisclassified = 0
    retData = data.copy(deep=True)
    pixel_columns = retData.columns[1:]
    retData[pixel_columns] = retData[pixel_columns].astype(np.float32)
    totalIterations = 0
    
    for i in range(len(data)):
        x = normalize(data.iloc[i, 1:].values)
        y = np.array([int(data.iloc[i, 0])])
        probabilities = model.predict(normalize(unnormalize(x)))
        curClass = np.argmax(probabilities)
        
        if(curClass != y[0]):
            startsMisclassified += 1
            retData.iloc[i, 0] = y[0]
            retData.iloc[i, 1:] = unnormalize(x)
            continue
        
        iteration = 0
        max_iterations = 200
        perturbation = np.zeros_like(x)
        alpha = .9
        while (curClass != 0 and iteration < max_iterations):
            w = model.gradient(x, np.array([0])) - model.gradient(x, y)
            f = probabilities[0, 0] - probabilities[0, y[0]]
            step_scale = 2.0 * (1.0 - probabilities[0, 0])
            r = (abs(f) / np.linalg.norm(w) * step_scale) * w
            
            # additions based on too high of a fail rate
            r_norm = .5
            r = r / np.linalg.norm(r) * r_norm if np.linalg.norm(r) > 0 else r
            
            perturbation = alpha * perturbation + (1 - alpha) * r
            x = x - perturbation
            iteration += 1
            probabilities = model.predict(normalize(unnormalize(x)))
            curClass = np.argmax(probabilities)
                        
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
    DeepFoolTargeted = dftar(mnist_test[1200:1600], model)
    end_time = time.time()
    print(f"Create training data execution time: {end_time - start_time:.2f} seconds")
    DeepFoolTargeted.to_pickle('dftar_train.pkl')
    
    print("testing data")
    start_time = time.time()
    DeepFoolTargeted = dftar(mnist_test[6700:6800], model)
    end_time = time.time()
    print(f"Create testing data execution time: {end_time - start_time:.2f} seconds")
    DeepFoolTargeted.to_pickle('dftar_test.pkl')
except FileNotFoundError as e:
    print(f"Error loading files: {e}")
except Exception as e:
    print(f"Error processing data: {e}")