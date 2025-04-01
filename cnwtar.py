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

def cnwtar(data, model):
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
        
        if(curClass == 0):
            startsMisclassified += 1
            retData.iloc[i, 0] = y[0]
            retData.iloc[i, 1:] = unnormalize(x)
            continue
        
        iteration = 0
        max_iterations = 200
        
        x_prime = x.copy()
        iteration = 0
        alpha = 1
        beta = -4
        learning_rate = .05
        epsilon = 1
        while curClass != 0 and iteration < max_iterations:
            t_o_h = np.zeros(10)
            t_o_h[0] = 1
            max_class = np.argmax(model.predict(normalize(unnormalize(x_prime))) * (1 - t_o_h))
            gradient = model.gradient(x_prime, np.array([0])) - model.gradient(x_prime, np.array([max_class]))
            gradient *= beta
            gradient += alpha * (-2*(x - x_prime))
            x_prime += learning_rate * gradient
            x_prime = np.clip(x_prime, x - epsilon, x + epsilon)
            probabilities = model.predict(normalize(unnormalize(x_prime)))
            curClass = np.argmax(probabilities)  
            iteration += 1
            
        #print(probabilities)            
        if iteration >= max_iterations:
            failsToMisclassify += 1
        if iteration == 0:
            startsMisclassified += 1
        print(iteration)
        totalIterations += iteration
        retData.iloc[i, 0] = y[0]
        retData.iloc[i, 1:] = unnormalize(x_prime)
        
    print(f"Average iterations to end: {totalIterations/len(data)}")
    print(f"Number failed to misclassify: {failsToMisclassify}")
    print(f"Number started classified as 0: {startsMisclassified}")
    retData[pixel_columns] = retData[pixel_columns].astype(np.uint8)
    
    return retData

try:
    with open('mnist_test.pkl', 'rb') as fid:
        mnist_test = pickle.load(fid)
    with open('model.pkl', 'rb') as fid:
        model = pickle.load(fid)
    
    print("training data")
    start_time = time.time()
    CarliniWagnerTargeted = cnwtar(mnist_test[2000:2400], model)
    end_time = time.time()
    print(f"Create training data execution time: {end_time - start_time:.2f} seconds")
    CarliniWagnerTargeted.to_pickle('cnwtar_train.pkl')
    
    print("testing data")
    start_time = time.time()
    CarliniWagnerTargeted = cnwtar(mnist_test[6900:7000], model)
    end_time = time.time()
    print(f"Create testing data execution time: {end_time - start_time:.2f} seconds")
    CarliniWagnerTargeted.to_pickle('cnwtar_test.pkl')
except FileNotFoundError as e:
    print(f"Error loading files: {e}")
except Exception as e:
    print(f"Error processing data: {e}")