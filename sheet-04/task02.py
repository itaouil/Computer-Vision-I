import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib import rc
rc('text', usetex=True)  # if you do not have latex installed simply uncomment this line + line 75

def load_data():
    """ loads the data for this task
    :return:
    """
    fpath = 'images/ball.png'
    radius = 70
    Im = cv2.imread(fpath, 0).astype('float32')/255  # 0 .. 1

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

# ===========================================
# RUNNING
# ===========================================
def geodesic(image_gradient):
    """
        Compute geodesic
        function values
        relative to the
        image gradient
    """
    # Compute the magnitude
    magnitude_gradient = magnitude(image_gradient)

    # Return geodesic function
    return 1 / (magnitude_gradient + 1)

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

    # Compute w
    w = geodesic(grad(Im))

    # Compute gradient of w
    dw = grad(w)

    # Tau/Step size
    tau = 0.5

    for t in range(n_steps):
        # Compute mean curvature motion
        curvature = compute_curvature(phi)

        # Compute propagation term
        propagation = dw * grad(phi)

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
