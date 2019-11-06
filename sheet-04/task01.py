import matplotlib.pyplot as plt
import numpy.linalg as la
import numpy as np
import cv2


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

# FUNCTIONS
# ------------------------
# your implementation here

# ------------------------


def get_gradient_image(img):
    return None


def get_external(img_gradient, vertices, k_size):
    matrix_shape = (k_size, vertices.shape[0])
    U_external = np.zeros(matrix_shape)
    k_half = (k_size // 2)
    for n in range(vertices.shape[0]):
        y_vert = vertices[n, 0]
        x_vert = vertices[n, 1]
        k = 0
        for y_kern in range(-k_half, k_half + 1):
            for x_kern in range(-k_half, k_half + 1):
                U_external[k, n] = img_gradient[y_vert + y_kern, x_vert + x_kern]
                k += 1
    return U_external


def euclidian_distance(a, b):
    return 0


def get_distances(vertices, k_size):
    k_total = k_size ** 2
    k_half = (k_size // 2)
    Pnk = np.zeros((k_total, k_total))
    for k in enumerate(range(k_half)):
        l = 0

    return None


def compute_new_vertices(vertices, U_external, k_size):
    matrix_shape = (k_size, vertices.shape[0])
    S_energies = np.zeros(matrix_shape)
    S_paths = np.zeros(matrix_shape)
    S_energies[:0] = U_external[:, 0]
    k_total = k_size ** 2
    for n in range(1, vertices.shape[0]):
        Pnk = get_distances(vertices, k_size)

    return np.array([[]])


def run(fpath, radius):
    """ run experiment
    :param fpath:
    :param radius:
    :return:
    """
    img, vertices = load_data(fpath, radius)
    img_gradient = cv2.Laplacian(img, cv2.CV_64F)
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111)
    n_steps = 200
    k_size = 3

    for t in range(n_steps):
        U_external = get_external(img_gradient, vertices, k_size)
        new_vertices = compute_new_vertices(vertices, U_external, k_size)
        ax.clear()
        ax.imshow(img, cmap='gray')
        ax.set_title('frame ' + str(t))
        plot_snake(ax, new_vertices)
        plt.pause(0.01)

        if new_vertices == vertices:
            break

    plt.pause(2)


if __name__ == '__main__':
    run('images/ball.png', radius=120)
    run('images/coffee.png', radius=100)
