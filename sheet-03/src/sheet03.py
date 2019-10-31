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

    # Read image in grayscale
    img = cv.imread('../images/shapes.png', cv.IMREAD_GRAYSCALE)

    # Perform gaussian blur
    # followed by canny edge
    # detector to find edges
    edges = cv.Canny(cv.GaussianBlur(img.copy(), (0, 0), 1), 100, 200)

    # Perform hough transform
    # on the computed edges
    lines = cv.HoughLines(edges, 1, 2, 110, None, 0, 0)

    # Draw lines on the
    # original image
    if lines is not None:
        for line in lines:
            # Get line params
            r = line[0][0]
            theta = line[0][1]

            # Trig values
            a = np.cos(theta)
            b = np.sin(theta)

            # Compute x and y
            x = r * a
            y = r * b

            # Create sample point
            pt1 = (int(x + 1000*(-b)), int(y + 1000*(a)))
            pt2 = (int(x - 1000*(-b)), int(y - 1000*(a)))

            # Draw line
            cv.line(img, pt1, pt2, (120,105,120), 3, 2)

    display_image("OpenCV Hough Transform", img)

def myHoughLines(img_edges, d_resolution, theta_step_sz, threshold):
    """
    Your implementation of HoughLines
    :param img_edges: single-channel binary source image (e.g: edges)
    :param d_resolution: the resolution for the distance parameter
    :param theta_step_sz: the resolution for the angle parameter
    :param threshold: minimum number of votes to consider a detection
    :return: list of detected lines as (d, theta) pairs and the accumulator
    """
    # Quantizations and resolutions
    theta_q = int(180 / theta_step_sz)
    d_resolution = int(np.linalg.norm(img_edges.shape) / d_resolution)

    # Create our accumulator
    accumulator = np.zeros((theta_q, d_resolution))

    # Get indices where
    # edge point occurs
    indices = np.argwhere(img_edges == 255)

    # Populate our accumulator
    for tuple in indices:
        # Unpack our tuple
        # values y and x
        y, x = tuple

        # Compute for each theta
        # resolution value
        for theta in range(theta_q):
            d = int(x * np.cos(theta) - y * np.sin(theta))
            accumulator[theta, d] += 1

    # Find votes that satisfy
    # the threshold passed by
    # the user
    detected_lines = np.argwhere(accumulator > threshold)

    return detected_lines, accumulator

def task_1_b():
    print("Task 1 (b) ...")

    # Read image in grayscale
    img = cv.imread('../images/shapes.png', cv.IMREAD_GRAYSCALE)

    # Perform gaussian blur
    # followed by canny edge
    # detector to find edges
    edges = cv.Canny(cv.GaussianBlur(img.copy(), (0, 0), 1), 100, 200)

    # Perform hough transform
    # on the computed edges
    detected_lines, accumulator = myHoughLines(edges, 1, 2, 110)

    # Draw lines on the
    # original image
    if detected_lines is not None:
        for line in detected_lines:
            # Get line params
            r = line[1]
            theta = line[0]

            # Trig values
            a = np.cos(theta)
            b = np.sin(theta)

            # Compute x and y
            x = r * a
            y = r * b

            # Create sample point
            pt1 = (int(x + 1000*(-b)), int(y + 1000*(a)))
            pt2 = (int(x - 1000*(-b)), int(y - 1000*(a)))

            # Draw line
            cv.line(img, pt1, pt2, (120,105,120), 3, 2)

    display_image("Custom Hough Transform", img)

##############################################
#     Task 2        ##########################
##############################################
def euclid_distance(x, xi):
    return np.sqrt(np.sum(np.power(x - xi, 2)))

def kernel_density_estimation(distance, bandwidth):
    c = 1 / (bandwidth * np.power(2 * np.pi, 0.5))
    e = np.exp(-0.5 * (np.power(distance, 2) / bandwidth))
    return c * e

def get_neighbours(points, x, neighbouring_distance = 5):
    # Neighouring points
    neighbouring_points = []

    # Find points which are
    # within the neighouring
    # distance
    for point in points:
        if euclid_distance(x, point) <= neighbouring_distance:
            neighbouring_points.append(point)

    return neighbouring_points

def mean_shift(data):
    # Make copy of data
    points = data.copy()

    # Old points
    old_points = points

    # Convergence flag
    converged = False

    while not converged:
        # Iterate over points
        for i, point in enumerate(points):
            # Get neighouring points
            # based on a distance metric
            neighbours = get_neighbours(points, point, 6)

            # Numerator and denominator
            # of the weighted sum
            numerator, denominator = 0, 0

            # Iterate over neighbours
            for n in neighbours:
                # Compute euclidean distance
                # between n and the actual point
                distance = euclid_distance(point, n)

                # Compute density, which
                # is nothing more than just
                # a weight
                weight = kernel_density_estimation(distance, 4)

                # Get numerator and denominator
                # of our new m(x)
                numerator += (n * weight)
                denominator += weight

            # Compute new point m
            m = numerator / denominator

            # Replace points
            points[i] = m
            print("Point: ", m, point)

            # Check for convergence
            converged = np.array_equal(points, old_points)
            print("Converged: ", converged)

            # Update old points
            old_points = points

    return points

def task_2():
    print("Task 2 ...")
    # Read image and convert it to grayscale
    img = cv.imread('../images/line.png', cv.IMREAD_GRAYSCALE)

    # Perform gaussian blur
    # followed by canny edge
    # detector to find edges
    edges = cv.Canny(cv.GaussianBlur(img.copy(), (0, 0), 1), 100, 200)

    # Resolutions (not for the new year though
    d_res = 1
    theta_res = 1
    _, accumulator = myHoughLines(edges, d_res, theta_res, 110)

    # Get clusters of accumulator
    print("Size: ", len(np.argwhere(accumulator > 10)))
    clusters = np.unique(mean_shift(np.argwhere(accumulator > 10)), axis=0)

    # Draw lines on the
    # original image
    if clusters is not None:
        for cluster in clusters:
            # Get accumulator point
            print("Cluster: ", cluster)
            y,x = cluster

            # Get line params
            r = x
            theta = y

            # Trig values
            a = np.cos(theta)
            b = np.sin(theta)

            # Compute x and y
            x = r * a
            y = r * b

            # Create sample point
            pt1 = (int(x + 1000*(-b)), int(y + 1000*(a)))
            pt2 = (int(x - 1000*(-b)), int(y - 1000*(a)))

            # Draw line
            cv.line(img, pt1, pt2, (120,105,120), 3, 2)

    # Visualize mean shift
    display_image("Lines with mean shift", img)

    # Visualize accumulator
    accumulator = ((accumulator - accumulator.min()) / accumulator.max()) * 255
    display_image("Accumulator", accumulator.astype(np.uint8))

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
    task_1_a()
    task_1_b()
    task_2()
    # task_3_a()
    # task_3_b()
    # task_3_c()
    # task_4_a()
