import cv2
import time
import sys
import numpy as np
from matplotlib import pyplot as plt

np.set_printoptions(threshold=sys.maxsize)


def display_image(window_name, img):
    """
        Displays image with given window name.
        :param window_name: name of the window
        :param img: image object to display
    """
    cv2.imshow(window_name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def mean_absolute_difference(img1, img2):
    """
        Computes the mean
        difference between
        two images.
    """
    # Compute mean of the
    # two images
    mean_img1 = np.sum(img1) / np.size(img1)
    mean_img2 = np.sum(img2) / np.size(img2)

    return np.absolute(mean_img1 - mean_img2)


def get_convolution_using_fourier_transform(image, kernel):
    # Compute FFT of image
    #  and shift 0 frequency
    # components
    ftimage = np.fft.fft2(image)
    ftshift = np.fft.fftshift(ftimage)
    magnitude_spectrum = 20 * np.log(np.abs(ftshift))
    magnitude_spectrum = np.asarray(
        magnitude_spectrum, dtype=np.uint8)
    display_image('Task1: Magn. Spect', magnitude_spectrum)

    # Blur image with kernel
    # in the spectrum domain
    ftimagep = ftimage * kernel

    # Return image as
    # matrix
    return np.abs(f_blur)


def task1():
    # Read image
    image = cv2.imread("./data/einstein.jpeg", 0)
    # display_image('Task1: Image', image)

    # Get gaussian kernel
    kernel = cv2.getGaussianKernel(7, 1)

    # Convolute image gaussian kernel
    conv_result = cv2.filter2D(image, -1, kernel)
    # display_image('Task1: Blur kernel', conv_result)

    # Convolute image using FFT
    # fft_result = get_convolution_using_fourier_transform(
    #     image, kernel)
    # display_image('Task1: FFT', fft_result)

    # compare results
    # print("(Task1) - Mean absolute difference: ",
    # mean_absolute_difference(conv_result, fft_result))


def draw_rectangles(image, positions, height, width):
    point_saved = (-1, -1)
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    for point in zip(*positions[::-1]):  # draw the rectangle around the matched template
        if point_saved == (-1, -1) or point[0] > (point_saved[0] + height) or point[1] > (point_saved[1] + width):
            cv2.rectangle(image, (point[0], point[1]), (point[0] + width, point[1] + height), (255, 0, 153), 1)
            point_saved = point
    return image


def ncc(image, template):
    dg = template - template.mean()
    df = image - image.mean()
    num = np.sum(dg * df)
    den = np.sqrt(np.sum(np.power(dg, 2)) * np.sum(np.power(df, 2)))
    return num / den


def normalized_cross_correlation(image, template):
    height, width = template.shape
    image_norm = np.zeros((image.shape[0] - height + 1, image.shape[1] - width + 1), dtype=np.float64)
    for y in range(image_norm.shape[0]):
        for x in range(image_norm.shape[1]):
            image_norm[y, x] = ncc(image[y: (y + height), x: (x + width)], template)

    points = np.where(image_norm >= 0.7)
    image = draw_rectangles(image, points, height, width)
    return image


def task2():
    image = cv2.imread("./data/lena.png", 0)
    template = cv2.imread("./data/eye.png", 0)

    # TODO: remove comment
    result_ncc = normalized_cross_correlation(image, template)
    display_image('NCC', result_ncc)

    resutl_cv_ncc = cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED)
    display_image('NCC', resutl_cv_ncc)


def gaussian_blur(img):
    kernel = np.array([[1, 4, 6, 4, 1],
                       [4, 16, 24, 16, 4],
                       [6, 24, 36, 24, 6],
                       [4, 16, 24, 16, 4],
                       [1, 4, 6, 4, 1]], dtype=np.float64)
    kernel /= 256
    return cv2.filter2D(img, -1, kernel)


def build_gaussian_pyramid(image, num_levels):
    levels = []
    next_level = image.copy()
    for _ in range(num_levels):
        levels.append(next_level)
        next_level = gaussian_blur(next_level)[::2, ::2]
    return levels


def build_gaussian_pyramid_opencv(image, num_levels):
    levels = []
    next_level = image.copy()
    for _ in range(num_levels):
        levels.append(next_level)
        next_level = cv2.pyrDown(next_level)
    return levels


def mean_abs_error(img1, img2):
    img1 = img1.copy().astype(np.int32)
    img2 = img2.copy().astype(np.int32)
    return abs(img1 - img2).mean()


def template_matching_multiple_scales(pyramid, template):
    return None


def task3():
    levels = 4
    image = cv2.imread("./data/traffic.jpg", 0)
    template = cv2.imread("./data/template.jpg", 0)

    mine_pyramid = build_gaussian_pyramid(image, levels)
    cv_pyramid = build_gaussian_pyramid_opencv(image, levels)

    for lvl in range(levels):
        mae = mean_abs_error(mine_pyramid[lvl], cv_pyramid[lvl])
        print('Mean abs error (level: {}): {}'.format(lvl, mae))

    # compare and print mean absolute difference at each level
    # result = template_matching_multiple_scales(pyramid, template)

    # show result


def get_derivative_of_gaussian_kernel(size, sigma):
    return None, None


def task4():
    image = cv2.imread("./data/einstein.jpeg", 0)

    kernel_x, kernel_y = get_derivative_of_gaussian_kernel(5, 0.6)

    edges_x = None  # convolve with kernel_x
    edges_y = None  # convolve with kernel_y

    magnitude = None  # compute edge magnitude
    direction = None  # compute edge direction

    cv2.imshow("Magnitude", magnitude)
    cv2.imshow("Direction", direction)


def l2_distance_transform_2D(edge_function, positive_inf, negative_inf):
    return None


def task5():
    image = cv2.imread("./data/traffic.jpg", 0)

    edges = None  # compute edges
    edge_function = None  # prepare edges for distance transform

    dist_transfom_mine = l2_distance_transform_2D(
        edge_function, positive_inf, negative_inf
    )
    dist_transfom_cv = None  # compute using opencv

    # compare and print mean absolute difference


if __name__ == "__main__":
    task1()
    task2()
    task3()
    task4()
    task5()
