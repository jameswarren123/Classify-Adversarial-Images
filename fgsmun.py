import numpy as np
import pandas as pd
import pickle
from model import SimpleNN

mnist_mean = 0.1307
mnist_std = 0.3081

def fgsmun(data, model):
    retData = data.copy(deep=True)
    pixel_columns = retData.columns[1:]
    retData[pixel_columns] = retData[pixel_columns].astype(np.float32)
    
    for i in range(len(data)):
        x = data.iloc[i, 1:].values.astype(float) / 255.0
        x = ((x - mnist_mean) / mnist_std).reshape(1, -1)
        y = int(data.iloc[i, 0])
        print(x)
        print(y)
        probabilities = model.predict(x)
        epsilon = 0.005
        max_iterations = 100
        iteration = 0
        
        while (np.argmax(probabilities) == y and iteration < max_iterations):
            gradients = model.gradient(x, y)
            if gradients is None:
                break
            x = x + epsilon * np.sign(gradients)
            probabilities = model.predict(x)
            iteration += 1
        
        x = (x * mnist_std) + mnist_mean
        x = np.clip(x * 255, 0, 255)
        
        retData.iloc[i, 0] = y
        retData.iloc[i, 1:] = x.flatten()
    
    retData[pixel_columns] = retData[pixel_columns].astype(np.uint8)
    
    return retData

try:
    with open('mnist_train.pkl', 'rb') as fid:
        mnist_train = pickle.load(fid)
    with open('model.pkl', 'rb') as fid:
        model = pickle.load(fid)
    print(mnist_train[10000:10010])
    FGSMUntargeted = fgsmun(mnist_train[10000:10010], model)
    FGSMUntargeted.to_pickle('fgsmun_train.pkl')
except FileNotFoundError as e:
    print(f"Error loading files: {e}")
except Exception as e:
    print(f"Error processing data: {e}")