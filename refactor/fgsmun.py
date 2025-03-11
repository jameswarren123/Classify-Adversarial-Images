import numpy as np
import pickle

mnist_train = None
model = None

mnist_mean = 0.1307
mnist_std = 0.3081

def fgsmun(data, model):
    retData = data.copy()
    for i in range(len(data)):
        x = data.iloc[i,1:] 
        x = np.array(x) #image as numpy array
        x = x / 255
        x = ((x - mnist_mean) / mnist_std).reshape(1, -1) #normalize image
        y = data.iloc[i,0] #image label
        probabilities = forward(x) #get probabilities of image
        epsilon = .005 #perturbation amount
        while(np.argmax(probabilities) == y):
            gradients = gradient(x, y) #get gradient of image
            x = x + epsilon * np.sign(gradients)
            probabilities = forward(x)
        x = (x*mnist_std) + mnist_mean #unnormalize image
        x = x * 255
        retData.loc[i] = np.append(y, x)
    return retData


with open('mnist_train.pkl', 'rb') as fid:
    mnist_train = pickle.load(fid)
with open('model.pkl', 'rb') as fid:
    model = pickle.load(fid)
FGSMUntargeted = fgsmun(mnist_train[10000:10400], model)
FGSMUntargeted.to_pickle('fgsmun_train.pkl')
