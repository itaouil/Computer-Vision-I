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


def display_image(window_name, img):
    """
        Displays image with given window name.
        :param window_name: name of the window
        :param img: image object to display
    """
    cv.imshow(window_name, img)
    cv.waitKey(0)
    cv.destroyAllWindows()


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
            pt1 = (int(x + 1000 * (-b)), int(y + 1000 * (a)))
            pt2 = (int(x - 1000 * (-b)), int(y - 1000 * (a)))

            #  Draw line
            cv.line(img, pt1, pt2, (120, 105, 120), 3, 2)

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
            pt1 = (int(x + 1000 * (-b)), int(y + 1000 * (a)))
            pt2 = (int(x - 1000 * (-b)), int(y - 1000 * (a)))

            #  Draw line
            cv.line(img, pt1, pt2, (120, 105, 120), 3, 2)

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


def get_neighbours(points, x, neighbouring_distance=5):
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
            y, x = cluster

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
            pt1 = (int(x + 1000 * (-b)), int(y + 1000 * (a)))
            pt2 = (int(x - 1000 * (-b)), int(y - 1000 * (a)))

            #  Draw line
            cv.line(img, pt1, pt2, (120, 105, 120), 3, 2)

    # Visualize mean shift
    display_image("Lines with mean shift", img)

    # Visualize accumulator
    accumulator = ((accumulator - accumulator.min()) / accumulator.max()) * 255
    display_image("Accumulator", accumulator.astype(np.uint8))


##############################################
#     Task 3        ##########################
##############################################


def get_distance(a, b):
    assert len(a) == len(b)
    sum_sq = 0
    for i in range(len(a)):
        sum_sq += (a[i] - b[i]) ** 2
    return sum_sq ** 0.5


def my_kmeans_gray(image, k):
    final_img = image.copy()
    centers = [random.randint(0, 255) for _ in range(k)]
    clusters = [[] for _ in range(k)]

    convergence = False
    iteration_num = 0
    while not convergence:
        iteration_num += 1
        print('iteration num = ', iteration_num)

        # assign each point to the cluster of closest center
        for y in range(image.shape[0]):
            for x in range(image.shape[1]):
                min_distance = 256
                idx_cluster = -1

                for idx, center in enumerate(centers):
                    dist = abs(image[y, x] - center)
                    if dist < min_distance:
                        min_distance = dist
                        idx_cluster = idx

                clusters[idx_cluster].append(image[y, x])
                final_img[y, x] = idx_cluster

        # update clusters' centers and check for convergence
        restart = False
        center_found = 0
        for idx, center in enumerate(centers):
            if len(clusters[idx]) == 0:
                # Some cluster doesn't have elements
                restart = True
                print('Restarting again: one cluster has no values')
                centers = [random.randint(0, 255) for _ in range(k)]
                iteration_num = 0
            else:
                new_center = sum(v for v in clusters[idx]) / len(clusters[idx])
                if abs(new_center - center) < 1:
                    center_found += 1
                centers[idx] = new_center
        if restart:
            continue

        if center_found == k:
            convergence = True

    # convert the image to the center colors
    for y in range(final_img.shape[0]):
        for x in range(final_img.shape[1]):
            final_img[y, x] = centers[final_img[y, x]]

    return final_img, centers


def my_kmeans_color(image, k):
    centers = [[random.randint(0, 255) for _ in range(3)] for _ in range(k)]
    final_img = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    clusters = [[] for _ in range(k)]

    convergence = False
    iteration_num = 0
    while not convergence:
        # assign each point to the cluster of closest center
        for y in range(image.shape[0]):
            for x in range(image.shape[1]):
                min_dist = 256
                idx_cluster = -1

                for idx, center in enumerate(centers):
                    dist = get_distance(image[y, x], center)
                    if dist < min_dist:
                        min_dist = dist
                        idx_cluster = idx

                clusters[idx_cluster].append(image[y, x])
                final_img[y, x] = idx_cluster

        # update clusters' centers and check for convergence
        restart = False
        center_found = 0
        for idx, center in enumerate(centers):
            if len(clusters[idx]) == 0:
                # Some cluster doesn't have elements
                restart = True
                print('Restarting again: one cluster has no values')
                centers = [[random.randint(0, 255) for _ in range(3)] for _ in range(k)]
                iteration_num = 0
            else:
                new_centers = []
                for color in range(3):
                    new_centers.append(sum(v[color] for v in clusters[idx]) / len(clusters[idx]))
                if get_distance(center, new_centers) < 1:
                    center_found += 1
                centers[idx] = new_centers
        if restart:
            continue

        iteration_num += 1
        print('iteration num = ', iteration_num)

        if center_found == k:
            convergence = True

    # convert the image to the center colors
    for y, row in enumerate(final_img):
        for x, p in enumerate(row):
            final_img[y, x] = centers[p]

    return final_img, centers


def my_kmeans_position(image, k):
    final_img = image.copy()
    h, w = image.shape[0], image.shape[1]
    centers = [[random.randint(0, 255) for _ in range(3)] for _ in range(k)]
    data = np.zeros((h, w, 3), dtype=float)
    for y, row in enumerate(image):
        for x, p in enumerate(row):
            data[y, x, 0] = image[y, x]
            data[y, x, 1] = y * 255 / h
            data[y, x, 2] = x * 255 / w
    clusters = [[] for _ in range(k)]

    convergence = False
    iteration_num = 0
    while not convergence:
        # assign each point to the cluster of closest center
        for y in range(image.shape[0]):
            for x in range(image.shape[1]):
                min_dist = 256
                idx_cluster = -1

                for idx, center in enumerate(centers):
                    dist = get_distance(image[y, x], center)
                    if dist < min_dist:
                        min_dist = dist
                        idx_cluster = idx

                clusters[idx_cluster].append(image[y, x])
                final_img[y, x] = idx_cluster

        # update clusters' centers and check for convergence
        restart = False
        center_found = 0
        for idx, center in enumerate(centers):
            if len(clusters[idx]) == 0:
                # some cluster doesn't have elements
                restart = True
                print('Restarting again: one cluster has no values')
                centers = [[random.randint(0, 255) for _ in range(3)] for _ in range(k)]
                iteration_num = 0
            else:
                new_centers = []
                for feature_num in range(3):
                    new_centers.append(sum(i[feature_num] for i in clusters[idx]) / len(clusters[idx]))
                if get_distance(center, new_centers) < 1:
                    center_found += 1
                centers[idx] = new_centers

        if restart:
            continue

        iteration_num += 1
        print('iteration num = ', iteration_num)

        if center_found == k:
            convergence = True

    # convert the image to the center colors
    for y, row in enumerate(final_img):
        for x, p in enumerate(row):
            final_img[y, x] = centers[p]

    return final_img, centers


def task_3_a():
    cluster_n = [2, 4, 6]
    print("Task 3 (a) ...")
    img = cv.imread('./images/flower.png', 0)

    for k in cluster_n:
        clustered_image, centers = my_kmeans_gray(img, k)
        print('displaying image with {} clusters'.format(k))
        display_image('Cluster: {}'.format(k), clustered_image)


def task_3_b():
    cluster_n = [2, 4, 6]
    print("Task 3 (b) ...")
    img = cv.imread('./images/flower.png')

    for k in cluster_n:
        clustered_image, centers = my_kmeans_color(img, k)
        print('displaying image with {} clusters'.format(k))
        display_image('Cluster: {}'.format(k), clustered_image)


def task_3_c():
    cluster_n = [2, 4, 6]
    print("Task 3 (c) ...")
    img = cv.imread('./images/flower.png', 0)

    for k in cluster_n:
        clustered_image, centers = my_kmeans_position(img, k)
        print('displaying image with {} clusters'.format(k))
        display_image('Cluster: {}'.format(k), clustered_image)


##############################################
#     Task 4        ##########################
##############################################


def task_4_ab():
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
    cluster1 = {}
    cluster2 = {}
    # retrieve the second smallest eigenvector
    min_cut = eigenvectors[sec_eigen[0]]
    for idx, val in enumerate(min_cut):
        if val < 0:
            cluster1[idx] = (chr(ascii))
        else:
            cluster2[idx] = (chr(ascii))
        ascii += 1
    print('C1: {}\nC2: {}'.format(list(cluster1.values()), list(cluster2.values())))

    cost = 0
    for idx_1, _ in cluster1.items():
        for idx_2, _ in cluster2.items():
            cost += W[idx_1, idx_2]

    print('Cost of the normalized cut: {:.2f}'.format(cost))


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
    task_4_ab()
