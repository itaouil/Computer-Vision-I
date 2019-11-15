import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib import rc


# rc('text', usetex=True)  # if you do not have latex installed simply uncomment this line + line 75

def load_data():
    """ loads the data for this task
    :return:
    """
    fpath = 'images/ball.png'
    radius = 70
    Im = cv2.imread(fpath, 0).astype('float32') / 255  # 0 .. 1

    # we resize the image to speed-up the level set method
    Im = cv2.resize(Im, dsize=(0, 0), fx=0.5, fy=0.5)

    height, width = Im.shape

    centre = (width // 2, height // 2)
    Y, X = np.ogrid[:height, :width]
    phi = radius - np.sqrt((X - centre[0]) ** 2 + (Y - centre[1]) ** 2)

    return Im, phi


def get_contour(phi):
    """ get all points on the contour
    :param phi:
    :return: [(x, y), (x, y), ....]  points on contour
    """
    eps = 1
    A = (phi > -eps) * 1
    B = (phi < eps) * 1
    D = (A - B).astype(np.int32)
    D = (D == 0) * 1
    Y, X = np.nonzero(D)
    return np.array([X, Y]).transpose()


def display_image(window_name, img):
    """
        Displays image with given window name.
        :param window_name: name of the window
        :param img: image object to display
    """
    cv2.imshow(window_name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def grad(g, split=False):
    kernel_x = np.array([[-1, 1]])
    kernel_y = np.array([[-1], [1]])
    dx = cv2.filter2D(g.copy(), -1, kernel_x)
    dy = cv2.filter2D(g.copy(), -1, kernel_y)
    if split:
        return dx, dy
    else:
        return np.array([dx, dy])


def grad_flipped(g):
    # Add a border to the top and left
    border_left = np.zeros((g.shape[0], 1))
    border_top = np.zeros((g.shape[1]))
    gx = np.hstack((border_left, g))
    gy = np.vstack((border_top, g))

    # Computation of the partial derivatives
    kernel = np.array([[-1, 1]])
    dx = cv2.filter2D(gx.copy(), -1, kernel)
    dy = cv2.filter2D(gy.copy(), -1, kernel.T)
    return dx[:, :-1], dy[:-1, :]


def der_first(g):
    kernel = np.array([[-1, 0, 1]])
    dx = cv2.filter2D(g.copy(), -1, kernel) * (1 / 2)
    dy = cv2.filter2D(g.copy(), -1, kernel.T) * (1 / 2)
    return dx, dy


def der_sec(g):
    kernel = np.array([[1, -2, 1]])
    dxx = cv2.filter2D(g.copy(), -1, kernel)
    dyy = cv2.filter2D(g.copy(), -1, kernel.T)
    return dxx, dyy


def der_first_matched(g):
    kernel = np.array([[1, 0, -1], [0, 0, 0], [-1, 0, 1]])
    dg = cv2.filter2D(g.copy(), -1, kernel) * (1 / 4)
    return dg


def magnitude(dg):
    return np.sqrt(dg[0] ** 2 + dg[1] ** 2)


def compute_curvature(p):
    dphi_x, dphi_y = der_first(p)
    dphi_xx, dphi_yy = der_sec(p)
    dphi_xy = der_first_matched(p)

    return ((dphi_xx * (dphi_y ** 2) - 2 * dphi_x * dphi_y * dphi_xy +
             dphi_yy * (dphi_x ** 2)) / ((dphi_x ** 2) + (dphi_y ** 2) + (10 ** -4)))


def compute_propagation(w, phi):
    dw_x, dw_y = grad(w)
    max_x = np.where(dw_x < 0, 0, dw_x)
    max_y = np.where(dw_y < 0, 0, dw_y)
    min_x = np.where(dw_x < 0, dw_x, 0)
    min_y = np.where(dw_y < 0, dw_y, 0)
    dphi_x, dphi_y = grad(phi, True)
    dphi_x_f, dphi_y_f = grad_flipped(phi)

    return (max_x * dphi_x + min_x * dphi_x_f +
            max_y * dphi_y + min_y * dphi_y_f)


def geodesic(gradient):
    """
        Compute geodesic
        function values
        relative to the
        image gradient
    """
    # Compute the magnitude
    magnitude_gradient = magnitude(gradient)

    # Return geodesic function
    return 1 / (magnitude_gradient ** 2 + 1)


def start_level_set():
    # Define number of steps
    n_steps = 20000
    plot_every_n_step = 100

    # Image and relative phi
    Im, phi = load_data()

    # Plotting figure
    fig = plt.figure(figsize=(16, 8))
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)

    # Compute w
    w = geodesic(grad(Im))

    # Tau/Step size
    tau = 0.05

    for t in range(n_steps):
        # Compute mean curvature motion
        curvature = compute_curvature(phi)

        # Compute propagation term
        propagation = compute_propagation(w, phi)

        # Update phi
        phi += tau * w * (curvature + propagation)

        if t % plot_every_n_step == 0:
            ax1.clear()
            ax1.imshow(Im, cmap='gray')
            ax1.set_title('frame ' + str(t))

            contour = get_contour(phi)
            if len(contour) > 0:
                ax1.scatter(contour[:, 0], contour[:, 1], color='red', s=1)

            ax2.clear()
            ax2.imshow(phi)
            ax2.set_title(r'$\phi$', fontsize=22)
            plt.pause(0.01)

    plt.show()


if __name__ == '__main__':
    start_level_set()
