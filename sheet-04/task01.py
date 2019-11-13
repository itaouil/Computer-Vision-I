import matplotlib.pyplot as plt
import numpy.linalg as la
import numpy as np
import cv2
import random

def display_image(window_name, img):
    """
        Displays image with given window name.
        :param window_name: name of the window
        :param img: image object to display
    """
    cv2.imshow(window_name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def plot_snake(ax, V, fill='green', line='red', alpha=1, with_txt=False):
    """ plots the snake onto a sub-plot
    :param ax: subplot (fig.add_subplot(abc))
    :param V: point locations ( [ (x0, y0), (x1, y1), ... (xn, yn)]
    :param fill: point color
    :param line: line color
    :param alpha: [0 .. 1]
    :param with_txt: if True plot numbers as well
    :return:
    """
    V_plt = np.append(V.reshape(-1), V[0, :]).reshape((-1, 2))
    ax.plot(V_plt[:, 0], V_plt[:, 1], color=line, alpha=alpha)
    ax.scatter(V[:, 0], V[:, 1], color=fill,
               edgecolors='black',
               linewidth=2, s=50, alpha=alpha)
    if with_txt:
        for i, (x, y) in enumerate(V):
            ax.text(x, y, str(i))

def load_data(fpath, radius):
    """
    :param fpath:
    :param radius:
    :return:
    """
    Im = cv2.imread(fpath, 0)
    h, w = Im.shape
    n = 20  # number of points
    u = lambda i: radius * np.cos(i) + w / 2
    v = lambda i: radius * np.sin(i) + h / 2
    V = np.array(
        [(u(i), v(i)) for i in np.linspace(0, 2 * np.pi, n + 1)][0:-1],
        'int32')

    return Im, V


# ===========================================
# RUNNING
# ===========================================

def pairs_distance(V):
    """
        Compute mean distance
        of vertices.
    """
    pairs_distance = []
    for x in range(1, len(V)):
        pairs_distance.append(euclidean_distance(V[x-1], V[x]))

    return pairs_distance

def get_exponent(x, y, sigma):
    return -1 * (x * x + y * y) / (2 * sigma)

def get_derivative_of_gaussian_kernel(size, sigma):
    assert size > 0 and size % 2 == 1 and sigma > 0

    kernel_x = np.zeros((size, size))
    kernel_y = np.zeros((size, size))

    size_half = size // 2

    for i in range(size):
        y = i - size_half
        for j in range(size):
            x = j - size_half
            kernel_x[i, j] = (
                    -1
                    * (x / (2 * np.pi * sigma * sigma))
                    * np.exp(get_exponent(x, y, sigma))
            )
            kernel_y[i, j] = (
                    -1
                    * (y / (2 * np.pi * sigma * sigma))
                    * np.exp(get_exponent(x, y, sigma))
            )

    return kernel_x, kernel_y

def get_gradient_image(img):
    kernel_x, kernel_y = get_derivative_of_gaussian_kernel(5, 0.6)
    edges_x = cv2.filter2D(img, ddepth=cv2.CV_32F, kernel=kernel_x)
    edges_y = cv2.filter2D(img, ddepth=cv2.CV_32F, kernel=kernel_y)
    return np.float32(np.sqrt(edges_x * edges_x + edges_y * edges_y))

def get_external_w_points(img_gradient, vertices, k_size):
    """
    Get the external energy of each point neighbor of each
    vertices from the gradient image (with respective
    coordinates of the points).
    :param img_gradient:
    :param vertices:
    :param k_size:
    :return: External energies and respective points.
    """
    # total number of k (with a kernel
    # of 3*3, k_total = 9)
    k_total = k_size ** 2
    # half of the kernel size (with a kernel
    # of 3*3, k_half = 1)
    k_half = (k_size // 2)
    U_external = np.zeros((k_total, vertices.shape[0]))
    U_points = np.zeros((k_total, vertices.shape[0], 2))

    # for each iteration 2, ..., n
    for n in range(vertices.shape[0]):
        k = 0
        x_vert = vertices[n, 0]
        y_vert = vertices[n, 1]
        for y_kern in range(-k_half, k_half + 1):
            for x_kern in range(-k_half, k_half + 1):
                y = y_vert + y_kern
                x = x_vert + x_kern
                U_points[k, n] = [y, x]
                U_external[k, n] = - (img_gradient[y, x] ** 2)
                k += 1
    return U_external, U_points

def euclidean_distance(a, b):
    """
    Calculate the euclidean
    distance between two point.
    :param a:
    :param b:
    :return: the distance.
    """
    return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5

def get_distances(points_n, points_n_1, k_size, pairs_distance, alpha=10):
    """
    Get distance between each connection from 1 to n.
    :param points_n:
    :param points_n_1:
    :param k_size: size of the neighbour kernel.
    :param alpha:
    :return:
    """
    # total number of k (with a kernel
    # of 3*3, k_total = 9)
    k_total = k_size ** 2
    Pn = np.zeros((k_total, k_total))
    for k in range(points_n.shape[0]):
        for l in range(points_n_1.shape[0]):
            Pn[k, l] = euclidean_distance(points_n[k], points_n_1[l])

    Pn = alpha * ((Pn - np.mean(pairs_distance)) ** 2)
    return Pn

def compute_new_vertices(vertices, U_external, U_points, k_size):
    """
    Compute the new vertices of a Snake Active Contours
    step.
    :param vertices: np.array of the current vertices.
    :param U_external: External energy of each point
    neighbor of each vertices.
    :param k_size: size of the neighbour kernel.
    :return: a np.array with a list of all the new
    vertices of size (n_vertices x coordinates).
    """
    # total number of k (with a kernel
    # of 3*3, k_total = 9)
    k_total = k_size ** 2
    matrix_shape = (k_total, vertices.shape[0])
    S_energies = np.zeros(matrix_shape)
    S_paths = np.zeros(matrix_shape)
    S_energies[:, 0] = U_external[:, 0]
    new_vertices = np.zeros((S_energies.shape[1], 2), dtype=np.int)
    # for each iteration 2, ..., n
    # (the first iteration is skipped)
    for n in range(1, vertices.shape[0]):
        # get distances between n and n-1
        Pn = get_distances(U_points[:, n], U_points[:, n - 1], k_size, pairs_distance(vertices))
        S_n_1 = S_energies[:, n - 1].T
        Pn = np.add(Pn, S_n_1)
        min_idxs = np.argmin(Pn, axis=1)
        min_values = Pn[np.arange(Pn.shape[0]), min_idxs]
        final_Sn = U_external[:, n].T + min_values
        S_energies[:, n] = final_Sn.T
        S_paths[:, n] = min_idxs

    # retrieve the final path
    idx_final_min = np.argmin(S_energies[:, -1])
    idx_vertices = np.zeros((S_energies.shape[1]), dtype=np.int)
    idx_vertices[-1] = idx_final_min
    for n_rev in range(S_paths.shape[1] - 1, 0, -1):
        idx_vertices[n_rev - 1] = S_paths[idx_vertices[n_rev], n_rev]

    # retrieve final coordinates by indexes
    for n in range(S_energies.shape[1]):
        idx_n = idx_vertices[n]
        new_vertices[n] = U_points[idx_n, n]
    new_vertices = np.roll(new_vertices, 1, axis=1)
    return new_vertices

def run(fpath, radius):
    """ run experiment
    :param fpath:
    :param radius:
    :return:
    """
    img, vertices = load_data(fpath, radius)
    img_gradient = get_gradient_image(img)
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111)
    n_steps = 200
    k_size = 9

    for t in range(n_steps):
        U_external, U_points = get_external_w_points(img_gradient, vertices, k_size)
        new_vertices = compute_new_vertices(vertices, U_external, U_points, k_size)
        ax.clear()
        ax.imshow(img, cmap='gray')
        ax.set_title('frame ' + str(t))
        plot_snake(ax, new_vertices)
        plt.pause(0.01)

        if (new_vertices == vertices).all():
            print('found')
            break

        roll_rand = random.randint(0, new_vertices.shape[0] - 1)
        vertices = np.roll(new_vertices, roll_rand, axis=0)

    plt.pause(2)

if __name__ == '__main__':
    run('images/ball.png', radius=120)
    run('images/coffee.png', radius=100)
