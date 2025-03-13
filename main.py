import numpy as np
import pickle
import matplotlib.pyplot as plt

mnist_mean = 0.1307
mnist_std = 0.3081

def unnormalize(X):
    X = (X * mnist_std) + mnist_mean
    X *= 255

    return X.astype(np.uint8)

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





def main():
    #mnist_train = None
    #mnist_test = None
    model = None
    FGSMUntargeted = None
    # Load the data
    #with open('mnist_train.pkl', 'rb') as fid:
    #    mnist_train = pickle.load(fid)
    #with open('mnist_test.pkl', 'rb') as fid:
    #    mnist_test = pickle.load(fid)
    with open('model.pkl', 'rb') as fid:
        model = pickle.load(fid)
    with open('fgsmun_train.pkl', 'rb') as fid:
        FGSMUntargeted = pickle.load(fid)
    visualize_example(FGSMUntargeted.iloc[0, 1:].values, model.predict(FGSMUntargeted.iloc[0, 1:].values), label=FGSMUntargeted.iloc[0, 0], filename='example.png')

if __name__ == '__main__':
    main()