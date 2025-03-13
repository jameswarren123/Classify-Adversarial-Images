import numpy as np
import pickle



def main():
    mnist_train = None
    mnist_test = None
    model = None
    FGSMUntargeted = None
    # Load the data
    with open('mnist_train.pkl', 'rb') as fid:
        mnist_train = pickle.load(fid)
    with open('mnist_test.pkl', 'rb') as fid:
        mnist_test = pickle.load(fid)
    with open('model.pkl', 'rb') as fid:
        model = pickle.load(fid)
    with open('fgsmun_train.pkl', 'rb') as fid:
        FGSMUntargeted = pickle.load(fid)
    print(mnist_train)
    

if __name__ == '__main__':
    main()