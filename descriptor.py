"""Our own implementation of the HOG descriptor."""

import numpy as np
from utils import *


def convolution(filt, img2d):
    """Stride one convolution between a given filter and an image. Assumes that
    the image has a single channel.

    Arguments
    ---------
    filt = filter to apply, np.array of shape (n_row_f, n_col_f).
    img2d = some channel of image as tensor, np.array of shape (h, w).

    Returns
    -------
    result = result of the convolution, np.array of shape 
    (h - n_row_f + 1, w - n_col_f + 1).
    """

    n_row_f, n_col_f = filt.shape[0], filt.shape[1]
    n_row_i, n_col_i = img2d.shape[0], img2d.shape[1]
    result = np.zeros((n_row_i - n_row_f + 1, n_col_i - n_col_f + 1))

    for curr_row in range(n_row_i - n_row_f + 1):
        for curr_col in range(n_col_i - n_col_f + 1):
            result[curr_row, curr_col] = np.sum(
                    np.multiply(filt, img2d[curr_row:(curr_row + n_row_f),
                                            curr_col:(curr_col + n_col_f)])
                    )

    return result


def gradients(tensor):
    """Given an image as tensor of shape 32x32x3, returns the magnitudes and
    angles of its gradients. We pad of zeros on the borders to preserve shape.

    Arguments
    ---------
    tensor = image as np.array of shape (32,32,3)

    Returns
    -------
    magnitudes = magnitude of gradients at each pixel, np.array of shape
    (32,32,3). 
    angles = angle of the gradients at each pixel, np.array of shape (32,32,3)
    """

    grad_x = np.outer([1], [1,0,-1])
    grad_y = np.outer([1,0,-1], [1])
    magnitudes = np.empty((32,32,3))
    convs_x = np.empty((32,32,3))
    convs_y = np.empty((32,32,3))

    for ch in range(3):
        conv_x = convolution(grad_x, tensor[:,:,ch])
        conv_y = convolution(grad_y, tensor[:,:,ch])
        # Pad with zeros, bringing them to 32x32
        conv_x = np.hstack([np.zeros(32).reshape(-1,1),
                            conv_x,
                            np.zeros(32).reshape(-1,1)])
        conv_y = np.vstack([np.zeros(32),
                            conv_y,
                            np.zeros(32)])
        magnitudes[:,:,ch] = np.sqrt(conv_x**2 + conv_y**2)
        convs_x[:,:,ch] = conv_x
        convs_y[:,:,ch] = conv_y

    angles = np.arctan2(np.max(convs_y, axis=2),
                        np.max(convs_x, axis=2))
    magnitudes = np.max(magnitudes, axis=2)

    return magnitudes, angles


def hist_on_cell(c_magnitudes, c_angles, n_orientations):
    """Given a cell, i.e., one element of the partitioning of the image, this
    function calculates the weighted histogram of orientations of gradients
    inside that cell.

    Arguments
    ---------
    c_magnitudes = magnitudes of gradients at the cell, np.array of shape
    (n_rows_c, n_cols_c), which are the dimensions of the cell.
    c_angles = angles of gradients at the cell, np.array of shape
    (n_rows_c, n_cols_c), which are the dimensions of the cell.

    Returns
    -------

    """

    # Angles go from -pi to pi radians.
    sep_points = np.linspace(-np.pi, np.pi, n_orientations+1)
    bins = [[] for _ in range(n_orientations)]
    weighted_bins = [[] for _ in range(n_orientations)]

    # Loop intra-cell
    n_rows_c, n_cols_c = c_magnitudes.shape
    for p_row in range(n_rows_c):
        for p_col in range(n_cols_c):
            # For each pixel, find the correct bin and put the magnitude there
            for i in range(n_orientations):
                if ((sep_points[i] <= c_angles[p_row, p_col]) 
                and (c_angles[p_row, p_col] <= sep_points[i+1])):
                    bins[i].append(c_magnitudes[p_row, p_col])
                    break

    # Weighted histogram
    M = np.sum(c_magnitudes)

    for i in range(n_orientations):
        # Sum all the weights on the bin and normalize by n. of pixels
        bins[i] = np.sum(bins[i])/(n_rows_c*n_cols_c)

    return bins


def hog(img, n_orientations=9, cell_size=8, block_size=3, eps=1e-5):
    """Calculate the histograms of oriented gradients for all cells on a given
    image. 

    Arguments
    ---------
    img = image as a line, i.e., a np.array of shape (h*w*c, ).
    n_orientations = number of angle ranges, i.e., number of bins on each 
    histogram, default is 9.
    cell_size = length in pixels for a side of the cell, default is 8. We will
    build square cells for simplicity.
    block_size = length in cells for a side of a block, default is 3.
    eps = constant to avoid division by zero when normalizing blocks, default
    is 1e-5.

    Returns
    -------
    blocks.ravel() = 1d array containing the concatenation of block-wise 
    normalization of the histograms.
    """

    # Filter image
    magnitudes, angles = gradients(normalize(line_to_tensor(img)))
    n_rows_f, n_cols_f = magnitudes.shape  # normally 32,32

    # Arithmetic
    assert n_rows_f%cell_size==0, 'Take a cell divisor of filtered image size'
    ratio = int(n_rows_f/cell_size)
    hist = np.empty((ratio, ratio, n_orientations))
    blocks = np.empty((ratio - block_size + 1,
                       ratio - block_size + 1,
                       n_orientations*(block_size**2)))

    # Cells
    for c_row in range(ratio):
        for c_col in range(ratio):
            c_magnitudes = magnitudes[c_row*cell_size:(c_row+1)*cell_size,
                                      c_col*cell_size:(c_col+1)*cell_size]
            c_angles = angles[c_row*cell_size:(c_row+1)*cell_size,
                              c_col*cell_size:(c_col+1)*cell_size]
            hist[c_row, c_col, :] = hist_on_cell(c_magnitudes, c_angles,
                                                 n_orientations)

    # Blocks
    for b_row in range(ratio - block_size + 1):
        for b_col in range(ratio - block_size + 1):
            block = hist[b_row:(b_row+block_size),
                         b_col:(b_col+block_size), :].ravel()
            for i in range(len(block)):
                block[i] /= np.sqrt(np.dot(block, block) + eps**2)
            blocks[b_row, b_col, :] = block

    return blocks.ravel()
