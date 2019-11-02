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

def my_kmeans_intensity(img, k):
    centers = [random.randint(0, 255) for _ in range(k)]
    clustered_image = img.copy()
    clusters = [[] for _ in range(k)]

    convergence = False
    iteration_num = 0
    while not convergence:
        iteration_num += 1
        print('iteration num = ', iteration_num)

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
        print('iteration num = ', iteration_num)

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
        print('iteration num = ', iteration_num)

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
        print('iteration num = ', iteration_num)

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
    img = cv.imread('./images/flower.png', cv.IMREAD_GRAYSCALE)

    for k in ks:
        clustered_image, centers = my_kmeans_intensity(img, k)
        print(f'3a - {k} clusters. centers:{", ".join(str(c) for c in centers)}')
        display_image(f'3a - {k} clusters', clustered_image)


def task_3_b():
    ks = (2, 4, 6)
    print("Task 3 (b) ...")
    img = cv.imread('./images/flower.png')

    for k in ks:
        clustered_image, centers = my_kmeans_rgb(img, k)
        print(f'3b - {k} clusters. centers:{", ".join(str(c) for c in centers)}')
        display_image(f'3b - {k} clusters', clustered_image)


def task_3_c():
    ks = (2, 4, 6)
    print("Task 3 (c) ...")
    img = cv.imread('./images/flower.png', cv.IMREAD_GRAYSCALE)

    for k in ks:
        clustered_image, centers = my_kmeans_intensity_position(img, k)
        print(f'3c - {k} clusters. centers:{", ".join(str(c) for c in centers)}')
        display_image(f'3c - {k} clusters', clustered_image)


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
    # task_1_a()
    # task_1_b()
    # task_2()
    # task_3_a()
    # task_3_b()
    # task_3_c()
    task_4_a()
