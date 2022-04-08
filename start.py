"""Fit a model and predict the image classes using kernel methods."""

from utils import *
from models import *
from descriptor import *
from kernels import *
import numpy as np
import time


# 1. Load data
print('Loading data...', end=' ', flush=True)
X_train, Y_train, X_val, Y_val, Xte = load_data(val_size=0.5, 
                                                data_augmentation=False)
print('loaded !')


# 2. Extract features
load_pre_comp_hog =\
        input('Load pre-computed HOG? (True/False, Enter to default) ')
assert load_pre_comp_hog.lower()=="true"\
       or load_pre_comp_hog.lower()=="false"\
       or not load_pre_comp_hog, "Invalid choice"  # default

if not load_pre_comp_hog or load_pre_comp_hog.lower()=="false":
    print('Extracting features with HOG (might take some minutes)')
    
    start_time = time.time()
    hists_train = np.apply_along_axis(hog, 1, X_train, n_orientations=9,
                                      cell_size=8, block_size=2)
    print('Total time for HOG on train: ', time.time()-start_time, 's')
    
    start_time = time.time()
    hists_val = np.apply_along_axis(hog, 1, X_val, n_orientations=9,
                                    cell_size=8, block_size=2)
    print('Total time for HOG on validation: ', time.time()-start_time, 's')
    
    start_time = time.time()
    hists_test = np.apply_along_axis(hog, 1, Xte, n_orientations=9,
                                     cell_size=8, block_size=2)
    print('Total time for HOG on test: ', time.time()-start_time, 's')

    # Save pre-computed versions
    with open('pre_computed_hogs.npy', 'wb') as f:
        np.save(f, hists_train)
        np.save(f, hists_val)
        np.save(f, hists_test)
    print('Saved HOGs')

elif load_pre_comp_hog.lower()=="true":  # Load pre-computed versions
    print('Loading pre-computed HOG descriptors.')
    with open('pre_computed_hogs.npy', 'rb') as f:
        hists_train = np.load(f)
        hists_val = np.load(f)
        hists_test = np.load(f)


# 3. Select model, train, and predict
model_choice = input("Select a model, 'onevsall'/'ridge': (Enter to default) ")\
                     or 'onevsall'
assert (model_choice=="ridge" or model_choice=="onevsall"), "Invalid choice !"

# Model 1: kernel ridge regression.
if model_choice=="ridge":
    # Parameters
    lambd = float(input("Enter regularization parameter lambda: ") or "1")
    sigma = float(input("Enter sigma: ") or "1.8")
    model = KernelRR(kernel=RBF(sigma=sigma).kernel, lambd=lambd)
    model.fit(X_train, Y_train)
    print('Validation score: ', model.score(X_val, Y_val), '\n')
    save_test_preds(model, Xte)

# Model 2: one vs. all multiclass SVM.
if model_choice=="onevsall":
    # Parameters
    C = float(input('Enter regularization parameter C: (Enter to default) ') 
                  or '10')
    sigma = float(input('Enter variance parameter sigma: (Enter to default) ')
                  or '1')
    degree = int(input('Enter degree of polynomial: (Enter to default) ') 
                  or '3')

    # Choose kernel
    kernel_dict = {  # since instantiating untrained models is cheap
                'RBF':RBF(sigma=sigma),
                'Linear':Linear(),
                'Poly':Poly(degree=degree),
                'Chi2':Chi2(),
            }

    kernel_name = input('Enter desired kernel (RBF, Linear, Poly, Chi2): ')\
                  or 'RBF'
    assert kernel_name in ['RBF', 'Linear', 'Poly', 'Chi2'], 'Invalid choice !'

    kernel = kernel_dict[kernel_name].kernel

    print('Computing kernel matrix...', end=' ', flush=True)
    pre_K = kernel(hists_train, hists_train)  # pre-computed kernel matrix
    print('done !')

    model = svmOneVsAll(C=C, kernel=kernel, pre_K=pre_K)
    # Fit
    print('Fitting model...', flush=True)
    model.fit(hists_train, Y_train)
    print('done !')
    # Validate
    print('(Simple) validation score: ', model.score(hists_val, Y_val))
    # Predict
    save_test_preds(model, hists_test)
