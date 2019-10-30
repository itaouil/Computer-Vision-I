import numpy as np
import cv2 as cv
import random


##############################################
#     Task 1        ##########################
##############################################


def task_1_a():
    print("Task 1 (a) ...")
    img = cv.imread('../images/shapes.png')
    '''
    ...
    your code ...
    ...
    '''


def myHoughLines(img_edges, d_resolution, theta_step_sz, threshold):
    """
    Your implementation of HoughLines
    :param img_edges: single-channel binary source image (e.g: edges)
    :param d_resolution: the resolution for the distance parameter
    :param theta_step_sz: the resolution for the angle parameter
    :param threshold: minimum number of votes to consider a detection
    :return: list of detected lines as (d, theta) pairs and the accumulator
    """
    accumulator = np.zeros((int(180 / theta_step_sz), int(np.linalg.norm(img_edges.shape) / d_resolution)))
    detected_lines = []
    '''
    ...
    your code ...
    ...
    '''
    return detected_lines, accumulator


def task_1_b():
    print("Task 1 (b) ...")
    img = cv.imread('../images/shapes.png')
    img_gray = None  # convert the image into grayscale
    edges = None  # detect the edges
    # detected_lines, accumulator = myHoughLines(edges, 1, 2, 50)
    '''
    ...
    your code ...
    ...
    '''


##############################################
#     Task 2        ##########################
##############################################


def task_2():
    print("Task 2 ...")
    img = cv.imread('../images/line.png')
    img_gray = None  # convert the image into grayscale
    edges = None  # detect the edges
    theta_res = None  # set the resolution of theta
    d_res = None  # set the distance resolution
    # _, accumulator = myHoughLines(edges, d_res, theta_res, 50)
    '''
    ...
    your code ...
    ...
    '''


##############################################
#     Task 3        ##########################
##############################################


def myKmeans(data, k):
    """
    Your implementation of k-means algorithm
    :param data: list of data points to cluster
    :param k: number of clusters
    :return: centers and list of indices that store the cluster index for each data point
    """
    centers = np.zeros((k, data.shape[1]))
    index = np.zeros(data.shape[0], dtype=int)
    clusters = [[] for i in range(k)]

    # initialize centers using some random points from data
    # ....

    convergence = False
    iterationNo = 0
    while not convergence:
        # assign each point to the cluster of closest center
        # ...

        # update clusters' centers and check for convergence
        # ...

        iterationNo += 1
        print('iterationNo = ', iterationNo)

    return index, centers


def task_3_a():
    print("Task 3 (a) ...")
    img = cv.imread('../images/flower.png')
    '''
    ...
    your code ...
    ...
    '''


def task_3_b():
    print("Task 3 (b) ...")
    img = cv.imread('../images/flower.png')
    '''
    ...
    your code ...
    ...
    '''


def task_3_c():
    print("Task 3 (c) ...")
    img = cv.imread('../images/flower.png')
    '''
    
    ...
    your code ...
    ...
    '''


##############################################
#     Task 4        ##########################
##############################################


def task_4_a():
    print("Task 4 (a) ...")
    D = [
        # A, B, C, D, E, F, G, H
        [2.2, 0, 0, 0, 0, 0, 0, 0],  # A
        [0, 2.1, 0, 0, 0, 0, 0, 0],  # B
        [0, 0, 2.6, 0, 0, 0, 0, 0],  # C
        [0, 0, 0, 3, 0, 0, 0, 0],  # D
        [0, 0, 0, 0, 3, 0, 0, 0],  # E
        [0, 0, 0, 0, 0, 3, 0, 0],  # F
        [0, 0, 0, 0, 0, 0, 3.3, 0],  # G
        [0, 0, 0, 0, 0, 0, 0, 2],  # H
    ]
    W = [
        # A, B, C, D, E, F, G, H
        [0, 1, .2, 1, 0, 0, 0, 0],  # A
        [1, 0, .1, 0, 1, 0, 0, 0],  # B
        [.2, .1, 0, 1, 0, 1, .3, 0],  # C
        [1, 0, 1, 0, 0, 1, 0, 0],  # D
        [0, 1, 0, 0, 0, 0, 1, 1],  # E
        [0, 0, 1, 1, 0, 0, 1, 0],  # F
        [0, 0, .3, 0, 1, 1, 0, 1],  # G
        [0, 0, 0, 0, 1, 0, 1, 0],  # H
    ]

    W = np.array(W, dtype=np.float32)
    D = np.array(D, dtype=np.float32)
    retval, eigenvalues, eigenvectors = cv.eigen(W)
    first_eigen = (98, 98)
    sec_eigen = (99, 99)
    # Laplacian matrix
    L = D - W

    # get the second smallest eigenvalue
    for idx, val in enumerate(eigenvalues):
        y = eigenvectors[idx, :]
        # calculate the minNcut
        cut = np.dot(np.dot(y.T, L), y) / np.dot(np.dot(y.T, D), y)
        # store the first eigenvalue
        first_eigen = (idx, cut) if cut < first_eigen[1] else first_eigen
        # store the second eigenvalue
        sec_eigen = (idx, cut) if first_eigen[1] < cut < sec_eigen[1] else sec_eigen

    # ascii code of the letter A
    ascii = 65
    cluster1 = []
    cluster2 = []
    # retrieve the second smallest eigenvector
    min_cut = eigenvectors[sec_eigen[0]]
    for val in min_cut:
        if val < 0:
            cluster1.append(chr(ascii))
        else:
            cluster2.append(chr(ascii))
        ascii += 1
    print('C1: {}\nC2: {}'.format(cluster1, cluster2))
    print('The cost of the normalized cut is O(n), where n is the number of the eigenvalues.')


##############################################
##############################################
##############################################

if __name__ == "__main__":
    task_1_a()
    task_1_b()
    task_2()
    task_3_a()
    task_3_b()
    task_3_c()
    task_4_a()
