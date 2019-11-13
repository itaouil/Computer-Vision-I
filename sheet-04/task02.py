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


def grad(g):
    kernel_x = np.array([[-1, 1]])
    kernel_y = np.array([[-1], [1]])
    dx = cv2.filter2D(g.copy(), -1, kernel_x)
    dy = cv2.filter2D(g.copy(), -1, kernel_y)
    return dx, dy


def magnitude(dx, dy):
    return np.sqrt(dx ** 2 + dy ** 2)


def compute_curvature(phi):
    phi_x, phi_y = grad(phi)
    phi_xy = magnitude(phi_x, phi_y)
    phi_xx, phi_yy = grad(phi_xy)

    return ((phi_xx * (phi_y ** 2) - 2 * phi_x * phi_y * phi_xy +
             phi_yy * (phi_x ** 2)) / (phi_x ** 2 + phi_y ** 2))


# ===========================================
# RUNNING
# ===========================================


if __name__ == '__main__':

    n_steps = 20000
    plot_every_n_step = 100

    Im, phi = load_data()

    fig = plt.figure(figsize=(16, 8))
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)


    pass
    for t in range(n_steps):

        # ------------------------
        # your implementation here

        # ------------------------

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
