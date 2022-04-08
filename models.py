"""Different kernel methods models."""

import numpy as np
from cvxopt import matrix
from cvxopt import solvers
from utils import *


class KernelSVC:
    """Binary SVM, which is fitted using cvxopt's solver.

    Arguments (constructor)
    -----------------------
        C (float): regularization parameter for C-SVM.
        kernel (function): kernel function to be used.
        epsilon (float): tubular tolerance for support vectors.
        pre_K (np.array): pre-computed kernel matrix, to not recalculate it.
    """

    def __init__(self, C, kernel, epsilon = 1e-3, pre_K=None):
        self.C = C
        self.kernel = kernel
        self.alpha = None
        self.a = None
        self.support = None
        self.epsilon = epsilon
        self.K = pre_K

    def fit(self, X, y):
        """Given data, fits the SVM model.

        Arguments
        ---------
            X (np.array (N, d)): training data covariates.
            y (np.array (N,)): training data labels.

        Returns
        -------
            void function, parameters are stored as attributes.
        """

        # Set up
        N = len(y)
        diag_y = np.diag(y)
        C_vect = self.C*np.ones(N)
        if self.K is None:
            self.K = self.kernel(X,X)

        # Set up for cvxopt solver
        P = matrix(diag_y @ self.K @ diag_y, tc='d')
        q = matrix(-np.ones(N), tc='d')
        A = matrix(y.reshape(1,N), tc='d')
        G = matrix(np.block([[np.identity(N)],[-np.identity(N)]]), tc='d')
        h = matrix(np.concatenate([C_vect, np.zeros(N)]), tc='d')
        b = matrix(np.zeros(1), tc='d')

        # Solve dual minimization problem
        solvers.options['show_progress'] = False
        sol = solvers.qp(P, q, G, h, A, b)
        self.alpha = np.ravel(sol['x'])  # extract optimal value

        # Get support vectors
        sv = self.alpha > self.epsilon
        self.support = X[sv]
        ind = np.arange(len(self.alpha))[sv]
        self.a = (diag_y @ self.alpha)[sv]
        self.sv_label = y[sv]

        # Calculate classifier's off-set
        self.b = 0*1.0;
        for i in range(len(self.a)):
            self.b += self.sv_label[i]
            self.b -= np.sum(self.a * self.K[sv,ind[i]])
        self.b /= len(self.a)

    def separating_function(self, x):
        """Calculates the value of the separating function at an array of 
        points.

        Arguments
        ---------
            x (np.array (N,d))

        Returns
        -------
            f + self.b (np.array(N,)): sep. function values at each point of x.
        """

        self.K = self.kernel(x, self.support)
        f = self.K @ self.a

        return f + self.b

    def predict(self, X):
        """Predict y values in {-1, 1} for given test data X.

        Arguments
        ---------
            X (np.array (N,d)): test data covariates.
        
        Returns
        -------
            2*(d>0)-1 (np.array(N,)): labels attributed.
        """

        d = self.separating_function(X)

        return 2 * (d> 0) - 1


class svmOneVsAll: 
    """Aggregation of multiply binary SVMs to perform multiclass classification
    according to the ''one vs. all'' voting scheme.

    Arguments (constructor)
    -----------------------
        kernel (function): common kernel function for all SVMs.
        C (float): common regularization parameter for all C-SVMs, default is 1.
        gamma (float): becomes gamma parameter for the kernel function. Its
        meaning depends on the kernel function and it is sometimes not used.
        degree (int): to be used in the case of a polynomial kernel.
        pre_K (np.array (N, M)): pre-computed kernel matrix.
    """

    def __init__(self, kernel, C=1, gamma=None, degree=None, pre_K=None):
        self.C = C
        self.kernel = kernel
        self.n_classes = None
        self.submodels = None
        self.K = pre_K

    def fit(self, X, Y):
        """Fits n_classes SVM models to distinguish each class from the others.

        Arguments
        ---------
            X (np.array (N,d)): training data covariates.
            Y (np.array (N,)): training data labels.
        """

        # Set up
        unique_classes = np.unique(Y)
        self.n_classes = len(unique_classes)

        # Create n_classes svm submodels
        self.submodels = [KernelSVC(C=self.C,
                                    kernel=self.kernel,
                                    pre_K=self.K) 
                          for i in range(self.n_classes)]
  
        # Fit the submodels
        for class_ in range(self.n_classes):
            binary_labels = -np.ones_like(Y)
            indices_class = np.where(Y == class_)
            binary_labels[indices_class] = 1
            self.submodels[class_].fit(X, binary_labels)
            print(f'Fitted class {class_ + 1}')


    def predict(self, X):
        """Use all submodels to predict classes of test samples X.

        Arguments
        ---------
            X (np.array(N,d)): test sample covariates.

        Returns
        -------
            y_pred (np.array(N,)): predicted labels for X.
        """

        scores = np.zeros((self.n_classes, X.shape[0]))
        for i in range(self.n_classes):
            scores[i] = self.submodels[i].separating_function(X)
        y_pred = np.argmax(scores, axis=0)
  
        return y_pred

    
    def score(self, x, y):
        """Calculates mean accuracy score given covariates to predict and true
        labels.

        Arguments
        ---------
            x (np.array(N,d)): covariates to predict.
            y_pred (np.array(N, )): true labels.

        Returns
        -------
            (float): average accuracy score.
        """
        y_pred = self.predict(x)
  
        return np.sum(y_pred == y)/y.shape[0]


class KernelRR:    
    """Kernel ridge regression class.

    Arguments (constructor)
    -----------------------
        kernel (function): kernel function to use.
        lambd (float): regularization parameter.
    """

    def __init__(self, kernel, lambd=0.1):
        self.lambd = lambd     
        self.kernel = kernel        
        self.a = None
        self.data = None
    
    def fit(self, X, y_vec):
        """Fits a kernel ridge regression model for given training data.

        Arguments
        ---------
            X (np.array(N,d)): training data covariates.
            y_vec (np.array(N,)): training data labels.

        Returns
        -------
            void function, parameters are stored as attributes.
        """

        y = vec2onehot(y_vec)
        n = X.shape[0]
        K = self.kernel(X,X)
        self.a = y.T @ np.linalg.inv(K + self.lambd *np.identity(n)) 
        self.data = X.copy()

    def predict(self, x):
        """Predict labels for given test data.

        Arguments
        ---------
            x (np.array (N, d)): test data for which we want labels.

        Returns
        -------
            (np.array(N,)): predicted labels.
        """

        K = self.kernel(self.data, x) 
        f_x = self.a @ K  # array of shape (10, nb_points_x)

        return np.argmax(f_x, axis=0)

    def score(self, x, y):
        """Calculates mean accuracy score given covariates to predict and true
        labels.

        Arguments
        ---------
            x (np.array(N,d)): covariates to predict.
            y (np.array(N, )): true labels.

        Returns
        -------
            (float): average accuracy score.
        """

        print('predicting...', end=' ', flush=True)
        y_pred = self.predict(x)
        print('predicted!')
        return np.mean(y_pred == y)
