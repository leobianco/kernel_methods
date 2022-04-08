"""Kernels to use in the models."""

import numpy as np


class RBF:
    def __init__(self, sigma=1.):
        """Creates an instance of an RBF kernel, i.e., fixes its parameters.
        
        Arguments
        ---------
        sigma = variance (not std) of the RBF kernel.

        Returns
        -------
        RBF object with given variance.
        """

        self.sigma = sigma  # variance, not std, of the kernel

    def kernel(self,X,Y):
        """Computes pairwise kernel between points in input matrices.

        Arguments
        ---------
        X = np.array of shape (N, d)
        Y = np.array of shape (M, d)

        Returns
        -------
        kernels = np.array of shape (N, M) with pairwise kernel evaluations.
        """

        A = np.sum(np.multiply(X, X), 1) 
        A = np.outer(A, np.ones(Y.shape[0])) 

        B = np.sum(np.multiply(Y,Y), 1) 
        B = np.outer(B, np.ones(X.shape[0])).T 

        dist = A - 2*np.dot(X,Y.T) + B 
        kernels = np.exp(-dist/(2*self.sigma))

        return kernels 


class Linear:
    """Linear kernel. Default constructor is used as this kernel has no 
    parameter.
    """

    def kernel(self,X,Y):
        """Computes pairwise kernel between points in input matrices.

        Arguments
        ---------
        X = np.array of shape (N, d)
        Y = np.array of shape (M, d)

        Returns
        -------
        kernels = np.array of shape (N, M) with pairwise kernel evaluations.
        """

        kernels = np.dot(X,Y.T)
        return kernels


class Poly:
    def __init__(self, gamma=None, coef=1, degree=3):
        """Creates an instance of a polynomial kernel, fixing its parameters.
        
        Arguments
        ---------
        gamma = float, default is None which sets it adaptatively on data to
        1/dimension_of_data when calling kernel evaluation.
        coef = float, independent term to add in the polynomial. Default is 1.
        degree = int, degree of the polynomial, default is 3.

        Returns
        -------
        Polynomial kernel object with given parameters.
        """

        self.gamma = gamma
        self.coef = coef
        self.degree = degree

    def kernel(self, X, Y):
        """Computes pairwise kernel between points in input matrices.

        Arguments
        ---------
        X = np.array of shape (N, d)
        Y = np.array of shape (M, d)

        Returns
        -------
        kernels = np.array of shape (N, M) with pairwise kernel evaluations.
        """

        if self.gamma == None:
            self.gamma = 1/X.shape[1]  # adaptive setting of gamma parameter
        kernels = (self.gamma*np.dot(X, Y.T) + self.coef)**self.degree
        return kernels 


class Chi2:
    def __init__(self, gamma=1.0):
        """Creates an instance of a chi-squared kernel, fixing its parameters.
        
        Arguments
        ---------
        gamma = scaling factor, float, default is 1 (no scaling).

        Returns
        -------
        Chi-squared kernel object with given parameters.
        """

        self.gamma = gamma

    def kernel(self, X, Y):
        """Computes pairwise kernel between points in input matrices.

        Arguments
        ---------
        X = np.array of shape (N, d)
        Y = np.array of shape (M, d)

        Returns
        -------
        kernels = np.array of shape (N, M) with pairwise kernel evaluations.
        """

        N = X.shape[0]
        M = Y.shape[0]
        chi_distance = np.zeros((N, M))
        for i in range(N):
            for j in range(M):
                num = (X[i,:] - Y[j,:])**2
                den = X[i,:] + Y[j,:]
                ind = np.where(den != 0)[0]

                chi_distance[i,j] = -np.sum(num[ind]/ den[ind])

        K = self.gamma * chi_distance
        kernels = np.exp(K,K)

        return kernels
