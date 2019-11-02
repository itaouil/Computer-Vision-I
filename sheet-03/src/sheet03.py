import numpy as np
import cv2 as cv
import random


def display_image(window_name, img):
    """
        Displays image with given window name.
        :param window_name: name of the window
        :param img: image object to display
    """
    cv.imshow(window_name, img)
    cv.waitKey(0)
    cv.destroyAllWindows()


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


def my_k_means(data, k, pos=False):
    """
    My implementation of k-means algorithm.
    :param histograms: histogram
    :param k: number of clusters
    :return: centers and ndarray of clusters.
    """
    shape = data.shape
    clusters = [[] for _ in range(k)]

    # initialize centers using some random points
    centers = np.zeros((data.shape[2], k))
    for i in range(centers.shape[1]):
        j = 0
        if pos:
            centers[0, i] = random.randint(0, shape[0])  # Y
            centers[1, i] = random.randint(0, shape[1])  # X
            j += 2
        for j in range(j, shape[2]):
            # Randomize: I / R / G / B
            centers[j, i] = random.randint(0, 255)
    print(centers)

    convergence = False
    iteration_no = 0
    while not convergence:
        iteration_no += 1
        print('iteration_no = ', iteration_no)

        # assign each point to the cluster of closest center
        for y in range(shape[0]):
            for x in range(shape[1]):
                dists = []
                point_img = data[y, x].reshape((data.shape[2], 1))

                for idx_k in range(k):
                    features_cnt = centers[:, idx_k].reshape((data.shape[2], 1))
                    dist = euclidian_dist(point_img, features_cnt)
                    dists.append(dist)

                idx_cluster = dists.index(min(dists))
                clusters[idx_cluster].append(point_img)

        # update clusters' centers and check for convergence
        print("someting")
        new_centers = np.zeros((data.shape[2], k))
        for idx_center, cluster in enumerate(clusters):
            print(idx_center)
            tmp_center = np.zeros((data.shape[2], 1))
            for points in cluster:
                tmp_center += points
            print(tmp_center)
            print(len(cluster))
            tmp_center /= len(cluster)
            print(tmp_center)

            new_centers[:, idx_center] = tmp_center.reshape((data.shape[2]))

        # get distance from original centers and new centers
        print('centers')
        print(centers)
        print(new_centers)

        centers_dist = euclidian_dist(centers, new_centers)
        new_centers = new_centers.astype(np.int)
        print(centers_dist)
        if centers_dist < 1:
            convergence = True
            centers = new_centers
        else:
            centers = new_centers
            clusters = [[] for _ in range(k)]

    print(centers)
    return centers, clusters


def euclidian_dist(a, b):
    dist = (a - b) ** 2
    val = np.sum(dist) ** 0.5

    return val


def generate_data(image, pos=False):
    shape = image.shape
    # reshape to tree dimension the original image
    if image.ndim == 2:
        n_feautures = 1
        image = image[:, :, np.newaxis].copy()
    elif image.ndim == 3:
        n_feautures = 3
        image = image.copy()
    else:
        raise Exception('Image dimension not allowed.')

    # create the zeros matrix with the right number
    # for the third dimension
    if pos:
        n_feautures += 2
    img_data = np.zeros((shape[0], shape[1], n_feautures))

    # insert values to the final image
    for y in range(image.shape[0]):
        for x in range(image.shape[1]):
            features = []
            data = image[y, x]

            if pos:
                # store the coordinates
                features.extend([y, x])

            features.extend(data.tolist())
            img_data[y, x] = np.array(features)
    return img_data


def paint_clusters(image, centers, clusters):
    for idx_center, cluster in enumerate(clusters):
        for feature in cluster:
            print(centers)
            image[feature[0], feature[1]] = centers[-1, idx_center]
    return image


def task_3_a():
    print("Task 3 (a) ...")
    img_gray = cv.imread('./images/flower.png', 0)
    for k in [2, 4, 6]:
        data = generate_data(img_gray, pos=True)
        centers, clusters = my_k_means(data, k, pos=True)
        img_clusters = paint_clusters(img_gray, centers, clusters)
        display_image('K-mean image', img_clusters)


def task_3_b():
    print("Task 3 (b) ...")
    img_color = cv.imread('./images/flower.png')
    for k in [2, 4, 6]:
        data = generate_data(img_color)
        # centers, clusters = my_k_means(data, k)
        # img_clusters = paint_clusters(img_gray, centers, clusters)
        # display_image('K-mean image', img_clusters)


def task_3_c():
    print("Task 3 (c) ...")
    print("Task 3 (a) ...")
    img_gray = cv.imread('./images/flower.png', 0)
    for k in [2, 4, 6]:
        data = generate_data(img_gray, pos=True)
        # centers, clusters = my_k_means(data, k)
        # img_clusters = paint_clusters(img_gray, centers, clusters)
        # display_image('K-mean image', img_clusters)


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
    # task_1_a()
    # task_1_b()
    # task_2()
    task_3_a()
    # task_3_b()
    # task_3_c()
    # task_4_a()
