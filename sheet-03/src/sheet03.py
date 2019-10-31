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


def create_histogram(image):
    """
    Create the intensity histogram of the image.
    :param image: the image
    :return: the histogram
    """
    histo = np.zeros((1, 256), dtype=np.int32)
    for y in range(image.shape[0]):
        for x in range(image.shape[1]):
            histo[[0], image[y, x]] += 1
    return histo


def euclidean_distances(a, b):
    dist = (a - b) ** 2
    ret = np.sum(dist) ** 0.5
    return ret


def my_k_means(image, k):
    """
    My implementation of k-means algorithm.
    :param histogram: histogram
    :param k: number of clusters
    :return: centers and ndarray of clusters.
    """
    shape = image.shape
    data = get_features_image(image)
    clusters = [[] for _ in range(k)]

    # initialize centers using some random points from histogram
    centers = np.zeros((data.shape[0], k))
    for i in range(centers.shape[1]):
        centers[0, i] = random.randint(0, shape[0])  # Y
        centers[1, i] = random.randint(0, shape[1])  # X
        if len(shape) == 2:
            centers[2, i] = random.randint(0, 255)  # I
        else:
            centers[2, i] = random.randint(0, 255)  # R
            centers[3, i] = random.randint(0, 255)  # G
            centers[4, i] = random.randint(0, 255)  # B

    convergence = False
    iteration_no = 0
    while not convergence:
        # assign each point to the cluster of closest center
        for idx_features_pixel in range(data.shape[1]):
            featu_pxl = data[:, idx_features_pixel]
            # index and distance from the closest center
            idx_n_dist = (0, 10000000)
            for idx_features_center in range(centers.shape[1]):
                featu_ctr = centers[:, idx_features_center]
                dist = euclidean_distances(featu_pxl, featu_ctr)
                if dist < idx_n_dist[1]:
                    idx_n_dist = (idx_features_center, dist)

            idx_center = idx_n_dist[0]
            clusters[idx_center].append(featu_pxl)

        # update clusters' centers and check for convergence
        centers_new = []
        for clust in clusters:
            final_coord = clust[0]
            for coord_clust in clust:
                final_coord += coord_clust
            print(final_coord)
            final_coord /= len(clust)
            print(final_coord)
            centers_new.append(final_coord)
        centers_new = np.array(centers_new).T

        print(centers_new)
        print(clusters[0][:5])
        print(clusters[1][:5])
        iteration_no += 1
        print('iteration_no = ', iteration_no)

        center_diff = (centers_new - centers)
        diff = np.where(center_diff < 1)
        if diff[0].size >= k * 3:
            convergence = True
        else:
            clusters = [[] for _ in range(k)]
        centers = centers_new
    return centers.astype(np.int32), clusters


def get_features_image(image):
    shape = image.shape
    if len(shape) == 2:
        # gray scale
        n_features = 3
    else:
        # differen color
        n_features = shape[2] + 2
    data = np.zeros((n_features, shape[0] * shape[1]))
    i = 0
    for y in range(image.shape[0]):
        for x in range(image.shape[1]):
            features = [x, y]
            if n_features > 3:
                features.extend(image[y, x].tolist())
            else:
                features.append(image[y, x])
            features = np.array(features)
            data[:, i] = features
            i += 1

    return data


def paint_clusters(image, centers, clusters):
    for idx, clust in enumerate(clusters):
        for val in clust:
            np.place(image, image == val, centers[idx])
    return image


def paint_clusters_data(image, centers, clusters):
    print(centers)
    for idx, clust in enumerate(clusters):
        for val in clust:
            val = val.astype(np.int8)
            y, x = val[0], val[1]
            image[y, x] = centers[2, idx]
    return image


def task_3_a():
    print("Task 3 (a) ...")

    img_gray = cv.imread('./images/flower.png', 0)
    # histo = create_histogram(img_gray)
    centers, clusters = my_k_means(img_gray, 2)
    img_clusters = paint_clusters_data(img_gray, centers, clusters)
    display_image('K-mean image', img_clusters)


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
    # task_1_a()
    # task_1_b()
    # task_2()
    task_3_a()
    # task_3_b()
    # task_3_c()
    # task_4_a()
