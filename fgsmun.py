import numpy as np
import pandas as pd
import pickle
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
    return X.astype(np.uint8)

def fgsmun(data, model):
    retData = data.copy(deep=True)
    pixel_columns = retData.columns[1:]
    retData[pixel_columns] = retData[pixel_columns].astype(np.float32)
    
    for i in range(len(data)):
        x = normalize(data.iloc[i, 1:].values)
        y = np.array([int(data.iloc[i, 0])])
        #print(x)
        print(y)
        probabilities = model.predict(x)
        epsilon = 0.05
        max_iterations = 100
        iteration = 0
        
        while (np.argmax(probabilities) == y[0] and iteration < max_iterations):
            gradients = model.gradient(x, y)
            if gradients is None:
                break
            x = x + epsilon * np.sign(gradients)
            probabilities = model.predict(normalize(unnormalize(x)))
            iteration += 1
        
        print(iteration)
        # print("preun")
        # print(x)
        # print(model.predict(x))
        # print("postun")
        
        # print(x-normalize(unnormalize(x)))
        # print(model.predict(normalize(unnormalize(x))))
        x = unnormalize(x)
        # print(x)
        retData.iloc[i, 0] = y
        retData.iloc[i, 1:] = x
        #print(model.predict(normalize(retData.iloc[i, 1:].values)))
    
    retData[pixel_columns] = retData[pixel_columns].astype(np.uint8)
    
    return retData

try:
    with open('mnist_test.pkl', 'rb') as fid:
        mnist_test = pickle.load(fid)
    with open('model.pkl', 'rb') as fid:
        model = pickle.load(fid)
    #print(mnist_train[10000:10010])
    FGSMUntargeted = fgsmun(mnist_test[:10], model)
    FGSMUntargeted.to_pickle('fgsmun_train.pkl')
except FileNotFoundError as e:
    print(f"Error loading files: {e}")
except Exception as e:
    print(f"Error processing data: {e}")