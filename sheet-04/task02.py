import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib import rc
import sys
np.set_printoptions(threshold=sys.maxsize)

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

def magnitude(dx, dy):
    """
        Computes magnitude of an image
    """
    return np.sqrt(dx ** 2 + dy ** 2)

def compute_derivative(image, kernel):
    """
        Compute derivative with correlation
    """
    return cv2.filter2D(image.copy(), -1, kernel)

def compute_curvature(phi):
    """
        Mean curvature motion
    """
    # Compute first derivatives
    phi_x = compute_derivative(phi, np.array([[-1, 0, 1]])) * 0.5
    phi_y = compute_derivative(phi, np.array([[-1], [0], [1]])) * 0.5

    # Compute second derivatives
    phi_xx = compute_derivative(phi, np.array([[1, -2, 1]]))
    phi_yy = compute_derivative(phi, np.array([[1], [-2], [1]]))

    # Compute derivative xy
    phi_xy = compute_derivative(phi, np.array([[1, 0, -1], [0, 0, 0], [-1, 0, 1]])) * 0.25

    # Curvature motion terms
    first_term = phi_xx * (phi_y ** 2)
    second_term = 2 * phi_x * phi_y * phi_xy
    third_term = phi_yy * (phi_x ** 2)
    fourth_term = (phi_x ** 2) + (phi_y ** 2) + (10 ** -4)

    return ((first_term + second_term + third_term) / fourth_term)

def compute_propagation(w, phi):
    """
        Propagation term
    """
    # Compute derivatives x and y
    wx = compute_derivative(w, np.array([[-1, 1]]))
    wy = compute_derivative(w, np.array([[-1], [1]]))

    # Get max and min of wx
    wx_max = np.maximum(wx, 0)
    wx_min = np.minimum(wx, 0)

    # Get max and min of wy
    wy_max = np.maximum(wy, 0)
    wy_min = np.minimum(wy, 0)

    # Compute derivative of wx shift
    wx_shift = np.zeros((w.shape[0], w.shape[1] + 1))
    wx_shift[:, 1:] = wx
    wx_shift = compute_derivative(wx_shift, np.array([[-1, 1]]))[:, 1:]

    # Compute derivative of wy shift
    wy_shift = np.zeros((w.shape[0] + 1, w.shape[1]))
    wy_shift[1:, :] = wy
    wy_shift = compute_derivative(wy_shift, np.array([[-1], [1]]))[1:, :]

    return wx_max * wx + wx_min * wx_shift + wy_max * wy + wy_min * wy_shift

def geodesic(dx, dy):
    """
        Compute geodesic function
    """
    return 1 / np.sqrt(((magnitude(dx, dy) ** 2) + 1))

if __name__ == '__main__':
    # Define number of steps
    n_steps = 20000
    plot_every_n_step = 100

    # Image and relative phi
    Im, phi = load_data()

    # Plotting figure
    fig = plt.figure(figsize=(16, 8))
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)

    # Compute image gradient x and y
    im_x = compute_derivative(Im, np.array([[-1, 1]]))
    im_y = compute_derivative(Im, np.array([[-1], [1]]))

    w = geodesic(im_x, im_y)

    # Tau/Step size
    tau = 0.5

    for t in range(n_steps):
        print(t)
        # Compute mean curvature motion
        curvature = compute_curvature(phi)

        # Compute propagation term
        propagation = compute_propagation(w, phi)

        # Update phi
        phi += tau * w * curvature + propagation

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
