"""General utils needed for manipulating the data."""

import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt


def train_val_split(X, Y, val_size=0.2):
    """Splits array and list into train and validation sets.

    Arguments
    ---------
    X = array size n x d (n = number of elements).
    Y = labels vector size n x 1.
    val_size = percentage of data to be validation.

    Returns
    -------
    X_train = training portion of the features.
    X_val = validation portion of the features.
    Y_train = training portion of the labels.
    Y_val = validation portion of the labels.
    """

    n = X.shape[0]  # size of data
    indexes_shuffled = list(range(n))
    random.shuffle(indexes_shuffled)  # shuffle elements
    train_indexes = indexes_shuffled[0:int((1-val_size)*n)]
    val_indexes = indexes_shuffled[int((1-val_size)*n):]

    X_train = X[train_indexes,:]
    X_val = X[val_indexes,:]
    Y_train = Y[train_indexes]
    Y_val = Y[val_indexes]

    return X_train, X_val, Y_train, Y_val


def load_data(val_size=0.5, data_augmentation=True):
    """Loads raw data and splits it into training and validation subsets.
    
    Arguments
    ---------
        val_size (float): percentage of validation data. Default is 0.5 for 
        performance reasons.
        data_augmentation (bool): option to augment data with horizontal flip. 
    
    Returns
    -------
        X_train_final (np.array(N_train,d)): training covariates.
        Y_train_final (np.array(N_train, )): training labels.
        X_val (np.array(N_val, d)): validation covariates.
        Y_val (np.array(N_val, )): validation labels.
    """

    # Raw data loading
    Xtr = np.array(pd.read_csv('data/Xtr.csv',
                   header=None,sep=',',
                   usecols=range(3072)))
    
    Xte = np.array(pd.read_csv('data/Xte.csv',
                   header=None,sep=',',
                   usecols=range(3072)))
    
    Ytr = np.array(pd.read_csv('data/Ytr.csv',
                   sep=',',
                   usecols=[1])).squeeze()

    # Splitting 
    if val_size == 0:
        X_train = Xtr
        Y_train = Ytr
        X_val = None
        Y_val = None
    else:
        X_train, X_val, Y_train, Y_val = train_val_split(Xtr, Ytr,
                                                         val_size=val_size)
    
    # Data augmentation
    if data_augmentation:
        X_train_final = np.vstack([X_train,
                                   np.apply_along_axis(flip_img, 1, X_train)])
        Y_train_final = np.hstack([Y_train,
                                   Y_train])
    else:
        X_train_final = X_train
        Y_train_final = Y_train
    
    return X_train_final, Y_train_final, X_val, Y_val, Xte


def normalize(tensor):
    """Given a tensor (line image reshaped to height x width x channels), 
    normalizes each channel to be in the range [0, 1].

    Arguments
    ---------
    tensor = image represented by np.array with shape (h, w, c).

    Returns
    -------
    normalized_tensor = channel-wise normalized tensor.
    """

    normalized_tensor = np.zeros(tensor.shape)
    for ch in range(3):
        normalized_tensor[:,:,ch] = (tensor[:,:,ch] - np.min(tensor[:,:,ch]))\
        / (np.max(tensor[:,:,ch]) - np.min(tensor[:,:,ch]))

    return normalized_tensor


def line_to_tensor(img):
    """Given an image as a line (in our challenge, a row of Xtr of shape 
    (3072,)), transforms it into a tensor, i.e., represents it as an np.array
    of shape (h, w, c). Not completely general, we hard coded the output
    dimensions for our challenge, but is easily adaptable.

    Arguments
    ---------
    img = image as a np.array of shape (h*w*c, ).

    Returns
    -------
    tensor = image as a np.array of shape (h,w,c).
    """

    tensor = np.zeros((32,32,3))
    for c in range(3):
        tensor[:,:,c] = np.reshape(img[c*1024:(c+1)*1024], (32,32))
    return tensor


def tensor_to_line(tensor):
    """Given a 32x32x3 tensor, returns a line with 1024 pixel intensities
    over three channels.

    Arguments
    ---------
    tensor = image as a np.array of shape (h,w,c).

    Returns
    -------
    img = image as a line, i.e., an np.array of shape (h*w*c, ).
    """
    
    img = np.empty((3072,))
    for i in range(tensor.shape[2]):
        img[i*1024:(i+1)*1024] = tensor[:,:,i].reshape(-1,)
    return img


def visualize_image(img):
    """Given an image as a line, i.e., a np.array of shape (h*w*c, ), transforms
    it and shows the image.
    
    Arguments
    ---------
    img = np.array of shape (h*w*c, ).

    Returns
    -------
    void function.
    """

    plt.imshow(normalize(line_to_tensor(img)))


def flip_img(img):
    """Takes a line image and flips the corresponding tensor along axis=1. This
    visually corresponds to horizontally flipping an image. It returns a line 
    image. Used to perform data augmentation.

    Arguments
    ---------
    img = np.array of shape (h*w*c, ).
     
    Returns
    -------
    tensor_as_line = np.array of shape (h*w*c, ) representing horizontally
    flipped version of img.
    """
    
    tensor = line_to_tensor(img)
    k = int(tensor.shape[0]/2)
    for i in range(k):
        tensor[:, [i, -1-i], :] = tensor[:,[-1-i, i], :]
    tensor_as_line = tensor_to_line(tensor)
    return tensor_as_line


def vec2onehot(y, n_classes=10):
    """Transforms vector y with m labels into array Y of shape (m, k) where k 
    is the number of classes and Y_ij = 1 if i belongs to class j, 0 otherwise.

    Arguments
    ---------
    y = vector of labels, np.array of shape (m,)
    n_classes = number of unique labels to encode, default is 10 for our case.
    """

    m = len(y)
    Y = np.zeros((m, n_classes))
    # we suppose classes range from 0 to n_classes-1
    for i in range(m):
      Y[i,y[i]] = 1
  
    return Y


def save_test_preds(model, test_data, file_name='Yte.csv'):
    """Predicts test data labels and saves them in csv format.

    Arguments
    ---------
        model (model object): one of the models implemented in model.py
        test_data (np.array (N_test, d)): test data.

    Returns
    -------
        void function, saves a csv file in working directory.
    """

    Yte = model.predict(test_data)
    Yte = {'Prediction' : Yte}
    dataframe = pd.DataFrame(Yte)
    dataframe.index += 1
    print('Saving predictions as ', file_name)
    dataframe.to_csv(file_name, index_label='Id')
