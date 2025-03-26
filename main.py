import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt

mnist_mean = 0.1307
mnist_std = 0.3081

# returns precision recall in nx4 dimensional array for precision and recall
def testClass(data, model):
    return 0

# returns precision and recall for binary
def testBinary(data, model):
    return 0

#generate trained classifier on data on data classifiying as 0-6 based on class
def whatMethod(data):
    model = "trained model temp"
    return model


#generate trained classifier on data classifing as adversarial or not
def isAdversarial(data):
    model = "trained model temp"
    return model


##################################
#ignore these methods for testing#
##################################
def normalize(X):
    X = X.astype(np.float32) / 255.0
    X = (X - mnist_mean) / mnist_std

    return X.reshape(-1, 1, 28, 28)

def unnormalize(X):
    X = (X * mnist_std) + mnist_mean
    X *= 255
    return np.clip(X, 0, 255).astype(np.uint8).reshape(784,)

def visualize_example(x_img, y_probs, b_unnormalize=True, label=-1,
                      filename=None):
    """
    Parameters:
    ------------------------------
    x_img: 1D numpy array of length 784 containing the image to display
    b_unnormalize: boolean, If set true, the image will be unnormalized
                   (i.e., inverse of standardization will be applied)
    label: an integer value representing the class of given handwritten digit image
    filename: string, when provided, the resulting plot will be saved with
              the given name

    Returns:
    ------------------------------
    None
    """
    img = x_img.reshape(28, 28)

    if b_unnormalize:
        x_img = unnormalize(x_img)

    if y_probs.ndim > 1:
        y_probs = y_probs.ravel()

    fig, ax = plt.subplots(ncols=2, figsize=(6.6,3))
    ax[0].imshow(img, cmap='Greys')
    ax[0].set_axis_off()
    ax[0].set_title('Generated Image')
    
    x_class = np.arange(10)
    max_prob = np.amax(y_probs)
    mask = y_probs < max_prob
    
    ax[1].bar(x_class[mask], y_probs[mask], align='center', color='C0', alpha=0.8)
    ax[1].bar(x_class[~mask], y_probs[~mask], align='center', color='C1', alpha=0.8)
    ax[1].set_xticks(x_class, [str(c) for c in x_class])
    ax[1].set_xlabel("Classes", fontsize=13)
    ax[1].set_ylabel("Class Probability", fontsize=13)

    if label >= 0:
        ax[1].set_title(f'Class label: {label}')

    plt.tight_layout(pad=0.1)
    plt.subplots_adjust(top=0.9, bottom=0.18, wspace=0.3)

    if filename is None:
        plt.show()
    else:
        fig.savefig(filename)
    plt.close(fig)
####################################
# ignore these methods for testing #
####################################




def main():

    # ---------------------------------- #
    # data collection and initialization #
    # ---------------------------------- #
    ###### initialize variables of data to load ######
    mnist_train = None
    mnist_test = None
    model = None
    FGSMUntargeted = None
    FGSMTargeted = None
    DeepFoolUntargeted = None
    DeepFoolTargeted = None
    CarliniWagnerTargeted = None
    randUntargeted = None


    ###### Load the data ######
    #with open('mnist_train.pkl', 'rb') as fid:
    #    mnist_train = pickle.load(fid)
    with open('mnist_test.pkl', 'rb') as fid:
        mnist_test = pickle.load(fid)
    with open('model.pkl', 'rb') as fid:
        model = pickle.load(fid)
    
    # confirm different gradients #
    # print(model.gradient(normalize(mnist_test.iloc[0, 1:].values), np.array([2])))
    # print("MMMMMMMMMMMMMMMMMMMMM")
    # print(model.gradient(normalize(mnist_test.iloc[0, 1:].values), np.array([0])))
    # done confirming ----------- #
    
    # ---------------------------- #
    # confirm mnist model accuracy #
    # ---------------------------- #
    # correct = 0
    # total = 9999
    # class_counts = np.zeros(10)
    # class_correct = np.zeros(10)
    # for i in range(total):
    #     x = normalize(mnist_test.iloc[i, 1:].values)
    #     y = int(mnist_test.iloc[i, 0])
    #     class_counts[y] += 1
    #     label = np.argmax(model.predict(x))
    #     if label == y:
    #         correct += 1
    #         class_correct[y] += 1
    # print(correct/total) # 0.982998299829983
    # print(class_correct/class_counts) # [0.99183673 0.99559471 0.99515504 0.98811881 0.96232179 
    #                                   # 0.98206278 0.98329854 0.9805258  0.97022587 0.97819623]
    
    # ------------------------------------------ #
    # back to our regularly scheduled programing #
    # ------------------------------------------ #
    
    with open('fgsmun_train.pkl', 'rb') as fid:             # 0 fail to misclassify, 5 start misclassified 
        FGSMUntargeted = pickle.load(fid)                   # 140.73 seconds build time, # average iterations
        
    with open('fgsmtar_train.pkl', 'rb') as fid:            # 24 failed to misclassify, 38 started at 0
        FGSMTargeted = pickle.load(fid)                     # 168.64 seconds build time, # average iterations
    
    with open('dfun_train.pkl', 'rb') as fid:               # 0 fail to misclassify, 10 start misclassified
        DeepFoolUntargeted = pickle.load(fid)               # 321.40 seconds build time, 14.57 averave iterations
    
    with open('dftar_train.pkl', 'rb') as fid:              # 20 fail to misclassify, 40 start misclassified
        DeepFoolTargeted = pickle.load(fid)                 # 241.83 seconds build time, 53.305 averave iterations
    
    # with open('carliniwagnertar_train.pkl', 'rb') as fid: #
    #     CarliniWagnerTargeted = pickle.load(fid)          #
    
    with open('randun_train.pkl', 'rb') as fid:             # 4 fail to misclassify, 8 start misclassified
        randUntargeted = pickle.load(fid)                   # 139.99 seconds build time, 32.7675 averave iterations
    
    # ------------------------- #
    # data visualization output #
    # ------------------------- #
    for i in range(10):
        visualize_example(mnist_test.iloc[i, 1:].values, model.predict(normalize(mnist_test.iloc[i, 1:].values)), label=mnist_test.iloc[i, 0], filename=f'normal_images/example{i}.png')

    for i in range(10):
        visualize_example(FGSMUntargeted.iloc[i, 1:].values, model.predict(normalize(FGSMUntargeted.iloc[i, 1:].values)), label=FGSMUntargeted.iloc[i, 0], filename=f'fgsmun_images/example{i}.png')
    
    for i in range(10):
        visualize_example(FGSMTargeted.iloc[i, 1:].values, model.predict(normalize(FGSMTargeted.iloc[i, 1:].values)), label=FGSMTargeted.iloc[i, 0], filename=f'fgsmtar_images/example{i}.png')
    
    for i in range(10):
        visualize_example(DeepFoolUntargeted.iloc[i, 1:].values, model.predict(normalize(DeepFoolUntargeted.iloc[i, 1:].values)), label=DeepFoolUntargeted.iloc[i, 0], filename=f'dfun_images/example{i}.png')

    for i in range(10):
        visualize_example(DeepFoolUntargeted.iloc[i, 1:].values, model.predict(normalize(DeepFoolTargeted.iloc[i, 1:].values)), label=DeepFoolTargeted.iloc[i, 0], filename=f'dftar_images/example{i}.png')
    
    for i in range(10):
        visualize_example(randUntargeted.iloc[i, 1:].values, model.predict(normalize(randUntargeted.iloc[i, 1:].values)), label=randUntargeted.iloc[i, 0], filename=f'randun_images/example{i}.png')
    # ------------------------- #
    
    return 1
    #combine data to train classifier on 4000 true images and 400*6 adversarial images
    trainingData = pd.concat([mnist_test[:4000],FGSMUntargeted, FGSMTargeted, DeepFoolUntargeted, DeepFoolTargeted, CarliniWagnerTargeted, randUntargeted])
   
    #label training data as true or perturbed 0 is real 1 is perturbed
    tfValues = []
    for i in range(6400):
        if i < 4000:
            tfValues.append(0)
        else:
            tfValues.append(1)
    trainingDataBinary = trainingData.copy()
    trainingDataBinary['True/Perturbed'] = tfValues

    #combine data to train classifier on 4000 true images and  1-6 for each class of adversarial images
    classValues = []
    for i in range(6400):
        if i < 4000:
            classValues.append(0)
        elif i < 4400:
            classValues.append(1)
        elif i < 4800:
            classValues.append(2)
        elif i < 5200:
            classValues.append(3)
        elif i < 5600:
            classValues.append(4)
        elif i < 6000:
            classValues.append(5)
        else:
            classValues.append(6)
    trainingDataClass = trainingData.copy()
    trainingDataClass['Class'] = classValues


    # --------------------------- #
    # Training and implementation #
    # --------------------------- #
    classifier1 = isAdversarial(trainingDataBinary)
    classifier2 = whatMethod(trainingDataClass)


    # ------- #
    # Testing #
    # ------- #

    #test classifier1 on 1000 true images and 100*6 adversarial images
    with open('fgsmun_test.pkl', 'rb') as fid:           # 0 failed to miscalssify, 1 started misclassified
        FGSMUntargeted = pickle.load(fid)                # 33.65 seconds to build data
        
    with open('fgsmtar_test.pkl', 'rb') as fid:          # 5 failed to classify, 10 started classified
        FGSMTargeted = pickle.load(fid)                  # 43.18
        
    with open('dfun_test.pkl', 'rb') as fid:             # 0 fail to misclassify, 2 start misclassified
        DeepFoolUntargeted = pickle.load(fid)            # 92.81 seconds build time, 14.91 averave iterations
    
    with open('dftar_test.pkl', 'rb') as fid:            # 2 fail to misclassify, 7 start misclassified
        DeepFoolTargeted = pickle.load(fid)              # 45.92 seconds build time, 41.27 averave iterations
    
    with open('carliniwagnertar_test.pkl', 'rb') as fid: #
        CarliniWagnerTargeted = pickle.load(fid)         #
    
    with open('randun_test.pkl', 'rb') as fid:           # 1 fail to misclassify, 0 start misclassified
        randUntargeted = pickle.load(fid)                # 41.12 seconds build time, 39.36 averave iterations
    
    #combine data to train classifier on 4000 true images and 400*6 adversarial images
    testingData = pd.concat([mnist_test[4000:5000],FGSMUntargeted, FGSMTargeted, DeepFoolUntargeted, DeepFoolTargeted, CarliniWagnerTargeted, randUntargeted])
   
    #label training data as true or perturbed 0 is real 1 is perturbed
    tfValues = []
    for i in range(1600):
        if i < 1000:
            tfValues.append(0)
        else:
            tfValues.append(1)
    testingDataBinary = testingData.copy()
    testingDataBinary['True/Perturbed'] = tfValues

    #combine data to train classifier on 4000 true images and  1-6 for each class of adversarial images
    classValues = []
    for i in range(1600):
        if i < 1000:
            classValues.append(0)
        elif i < 1100:
            classValues.append(1)
        elif i < 1200:
            classValues.append(2)
        elif i < 1300:
            classValues.append(3)
        elif i < 1400:
            classValues.append(4)
        elif i < 1500:
            classValues.append(5)
        else:
            classValues.append(6)
    testingDataClass = trainingData.copy()
    testingDataClass['Class'] = classValues

    
    pnrBinary = testBinary(testingDataBinary, classifier1)
    pnrMulticlass = testClass(testingDataBinary, classifier2)


    # ------------------------------- #
    # visualize classifiers and tests #
    # ------------------------------- #


    



if __name__ == '__main__':
    main()