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
    img_gray = None # convert the image into grayscale
    edges = None # detect the edges
    #detected_lines, accumulator = myHoughLines(edges, 1, 2, 50)
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
    img_gray = None # convert the image into grayscale
    edges = None # detect the edges
    theta_res = None # set the resolution of theta
    d_res = None # set the distance resolution
    #_, accumulator = myHoughLines(edges, d_res, theta_res, 50)
    '''
    ...
    your code ...
    ...
    '''


##############################################
#     Task 3        ##########################
##############################################


def my_kmeans_intensity(img, k):

    centers = [random.randint(0, 255) for _ in range(k)]
    clustered_image = img.copy()
    clusters = [[] for _ in range(k)]

    convergence = False
    iteration_num = 0
    while not convergence:
        # assign each point to the cluster of closest center
        for y, row in enumerate(img):
            for x, p in enumerate(row):
                min_distance = 256
                cluster_num = -1

                for i, c in enumerate(centers):
                    distance_for_this_center = abs(p - c)
                    if distance_for_this_center < min_distance:
                        min_distance = distance_for_this_center
                        cluster_num = i

                clusters[cluster_num].append(p)  # add intensity value to the list of values in the cluster
                clustered_image[y, x] = cluster_num  # cluster number -> temporary value in the clustered image

        # update clusters' centers and check for convergence
        start_anew = False
        num_converged_centers = 0
        for i, c in enumerate(centers):
            # if some cluster ended up having no elements
            if len(clusters[i]) == 0:
                # that means random initialization was bad
                # initialize new random centers and iterate again
                centers = [random.randint(0, 255) for _ in range(k)]
                iteration_num = 0
                print('1 cluster had 0 elements due to bad random initialization of the centers. Starting anew')
                start_anew = True
            # else if all clusters have at least 1 pixel
            else:
                c_new = sum(v for v in clusters[i]) / len(clusters[i])
                if abs(c - c_new) < 1:
                    num_converged_centers += 1
                centers[i] = c_new
        if start_anew:
            continue

        iteration_num += 1
        print('iterationNo = ', iteration_num)

        if num_converged_centers == k:
            convergence = True
            print('converged')

    # after it converged
    # turn cluster numbers into final central values in the output image
    for y, row in enumerate(clustered_image):
        for x, p in enumerate(row):
            clustered_image[y, x] = centers[p]

    return clustered_image, centers


def euclidian_distance(a, b):
    assert len(a) == len(b)
    sum_sq = 0
    for i in range(len(a)):
        sum_sq += (a[i] - b[i]) ** 2
    return sum_sq ** 0.5


def my_kmeans_rgb(img, k):
    centers = [[random.randint(0, 255) for _ in range(3)] for _ in range(k)]
    pixel_to_cluster = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    clustered_image = img.copy()
    clusters = [[] for _ in range(k)]

    convergence = False
    iteration_num = 0
    while not convergence:
        # assign each point to the cluster of closest center
        for y, row in enumerate(img):
            for x, p in enumerate(row):
                min_distance = 256
                cluster_num = -1

                for i, c in enumerate(centers):
                    distance_for_this_center = euclidian_distance(p, c)
                    if distance_for_this_center < min_distance:
                        min_distance = distance_for_this_center
                        cluster_num = i

                clusters[cluster_num].append(p)  # add color value to the list of values in the cluster
                pixel_to_cluster[y, x] = cluster_num  # cluster number -> temporary value in the clustered image

        # update clusters' centers and check for convergence
        start_anew = False
        num_converged_centers = 0
        for i, c in enumerate(centers):
            # if some cluster ended up having no elements
            if len(clusters[i]) == 0:
                # that means random initialization was bad
                # initialize new random centers and iterate again
                centers = [[random.randint(0, 255) for _ in range(3)] for _ in range(k)]
                iteration_num = 0
                print('1 cluster had 0 elements due to bad random initialization of the centers. Starting anew')
                start_anew = True
            # else if all clusters have at least 1 pixel
            else:
                c_new = []
                for color in range(3):
                    c_new.append(sum(v[color] for v in clusters[i]) / len(clusters[i]))
                if euclidian_distance(c, c_new) < 1:
                    num_converged_centers += 1
                centers[i] = c_new
        if start_anew:
            continue

        iteration_num += 1
        print('iterationNo = ', iteration_num)

        if num_converged_centers == k:
            convergence = True
            print('converged')

    # after it converged
    # turn cluster numbers into final central values in the output image
    for y, row in enumerate(pixel_to_cluster):
        for x, p in enumerate(row):
            clustered_image[y, x] = centers[p]

    return clustered_image, centers


def my_kmeans_intensity_position(img, k):
    centers = [[random.randint(0, 255) for _ in range(3)] for _ in range(k)]
    clustered_image = img.copy()
    height, width = img.shape[0], img.shape[1]
    data = np.zeros((height, width, 3), dtype=float)
    for y, row in enumerate(img):
        for x, p in enumerate(row):
            data[y, x, 0] = img[y, x]
            data[y, x, 1] = y * 255 / height
            data[y, x, 2] = x * 255 / width
    clusters = [[] for _ in range(k)]

    convergence = False
    iteration_num = 0
    while not convergence:
        # assign each point to the cluster of closest center
        for y, row in enumerate(data):
            for x, p in enumerate(row):
                min_distance = 256
                cluster_num = -1

                for i, c in enumerate(centers):
                    distance_for_this_center = euclidian_distance(p, c)
                    if distance_for_this_center < min_distance:
                        min_distance = distance_for_this_center
                        cluster_num = i

                clusters[cluster_num].append(p)  # add data value to the list of values in the cluster
                clustered_image[y, x] = cluster_num  # cluster number -> temporary value in the clustered image

        # update clusters' centers and check for convergence
        start_anew = False
        num_converged_centers = 0
        for i, c in enumerate(centers):
            # if some cluster ended up having no elements
            if len(clusters[i]) == 0:
                # that means random initialization was bad
                # initialize new random centers and iterate again
                centers = [[random.randint(0, 255) for _ in range(3)] for _ in range(k)]
                iteration_num = 0
                print('1 cluster had 0 elements due to bad random initialization of the centers. Starting anew')
                start_anew = True
            # else if all clusters have at least 1 pixel
            else:
                c_new = []
                for feature_num in range(3):
                    c_new.append(sum(v[feature_num] for v in clusters[i]) / len(clusters[i]))
                if euclidian_distance(c, c_new) < 1:
                    num_converged_centers += 1
                centers[i] = c_new
        if start_anew:
            continue

        iteration_num += 1
        print('iterationNo = ', iteration_num)

        if num_converged_centers == k:
            convergence = True
            print('converged')

    # after it converged
    # turn cluster numbers into final central values in the output image
    for y, row in enumerate(clustered_image):
        for x, p in enumerate(row):
            clustered_image[y, x] = centers[p][0]

    return clustered_image, centers


def task_3_a():
    ks = (2, 4, 6)
    print("Task 3 (a) ...")
    img = cv.imread('../images/flower.png', cv.IMREAD_GRAYSCALE)

    for k in ks:
        clustered_image, centers = my_kmeans_intensity(img, k)
        print(f'3a - {k} clusters. centers:{", ".join(str(c) for c in centers)}')
        display_image(f'3a - {k} clusters', clustered_image)


def task_3_b():
    ks = (2, 4, 6)
    print("Task 3 (b) ...")
    img = cv.imread('../images/flower.png')

    for k in ks:
        clustered_image, centers = my_kmeans_rgb(img, k)
        print(f'3b - {k} clusters. centers:{", ".join(str(c) for c in centers)}')
        display_image(f'3b - {k} clusters', clustered_image)


def task_3_c():
    ks = (6,)
    print("Task 3 (c) ...")
    img = cv.imread('../images/flower.png', cv.IMREAD_GRAYSCALE)

    for k in ks:
        clustered_image, centers = my_kmeans_intensity_position(img, k)
        print(f'3c - {k} clusters. centers:{", ".join(str(c) for c in centers)}')
        display_image(f'3c - {k} clusters', clustered_image)


##############################################
#     Task 4        ##########################
##############################################


def task_4_a():
    print("Task 4 (a) ...")
    D = None  # construct the D matrix
    W = None  # construct the W matrix
    '''
    ...
    your code ...
    ...
    '''


##############################################
##############################################
##############################################

if __name__ == "__main__":
    # task_1_a()
    # task_1_b()
    # task_2()
    # task_3_a()
    # task_3_b()
    # task_3_c()
    # task_4_a()

