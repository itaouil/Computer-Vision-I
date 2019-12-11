import cv2
import numpy as np
import matplotlib.pylab as plt


def display_image(window_name, img):
    """
        Displays image with given window name.
        :param window_name: name of the window
        :param img: image object to display
    """
    cv2.imshow(window_name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def get_structural_tensor(img, ksize):
    """
        Compute M for img.
    """
    # Smooth image
    img = cv2.GaussianBlur(img, (3, 3), 1)

    # Compute derivatives for img
    kernel_x = np.array([[-1, 1]])
    kernel_y = np.array([[-1], [1]])
    I_x = cv2.filter2D(img, -1, kernel_x).astype(np.int16)
    I_y = cv2.filter2D(img, -1, kernel_y).astype(np.int16)

    # Compute derivative products
    I_xx = I_x ** 2
    I_yy = I_y ** 2
    I_xy = I_x * I_y

    # Apply box filter to products
    I_xx = cv2.boxFilter(I_xx, -1, ksize)
    I_yy = cv2.boxFilter(I_yy, -1, ksize)
    I_xy = cv2.boxFilter(I_xy, -1, ksize)

    # Create Harris response matrix M
    M = np.zeros((img.shape[0], img.shape[1], 4))

    # Populate matrix M
    for y in range(img.shape[0]):
        for x in range(img.shape[1]):
            M[y, x, :] = np.array([I_xx[y, x], I_xy[y, x], I_xy[y, x], I_yy[y, x]])

    return M


def local_maxima(img):
    cv2.meanShift(img, 20, criteria)


def harris_detector(M):
    """
        Compute harris responses.
    """
    # Harries responses
    responses = np.zeros((M.shape[0], M.shape[1]))

    # Compute responses for each pixel
    for y in range(M.shape[0]):
        for x in range(M.shape[1]):
            # Reshape array into
            # a 2x2 matrix
            M_temp = np.reshape(M[y, x], (2, 2))

            # Compute f value
            responses[y, x] = np.linalg.det(M_temp) - 0.04 * np.trace(M_temp) ** 2

    # Threshold responses
    responses = np.where(responses > 100, 255, 0)

    # responses = img_mean_shift(responses)

    return responses


def forstner_detector(M):
    """
        Compute Forstner responses.
    """
    # Forstner responses
    w = np.zeros((M.shape[0], M.shape[1]))
    q = np.zeros((M.shape[0], M.shape[1]))

    # Compute responses for each pixel
    for y in range(M.shape[0]):
        for x in range(M.shape[1]):
            # Reshape array into
            # a 2x2 matrix
            M_temp = np.reshape(M[y, x], (2, 2))

            # Compute f value
            w[y, x] = np.linalg.det(M_temp) / (np.trace(M_temp) + .00001)
            q[y, x] = 4 * np.linalg.det(M_temp) / (np.trace(M_temp) ** 2 + .00001)

    q = (q - q.min()) / (q.max() - q.min()) * 255

    # Threshold responses
    w = np.where(w > 100, 255, 0)
    q = np.where(q > 254.99995, 255, 0)

    return w, q


def main():
    # Load the image
    img = cv2.imread("./data/exercise2/building.jpeg", 0)
    # img = cv2.imread("/Users/dailand10/Desktop/Computer-Vision-I/sheet-08/data/exercise2/building.jpeg", 0)

    # Compute Structural Tensor
    ksize = (3, 3)
    M = get_structural_tensor(img, ksize)

    # Harris Corner Detection
    corners = harris_detector(M)
    display_image("Harris' corners", corners.astype(np.uint8))

    # Forstner Corner Detection
    w, q = forstner_detector(M)
    display_image("Forstner's corners w", w.astype(np.uint8))
    display_image("Forstner's corners q", q.astype(np.uint8))


if __name__ == '__main__':
    main()
