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


def draw_rectangles(image, yxs, temp_shape):
    height, width = temp_shape
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    for yx in yxs:  # draw the rectangle around the matched template
        y = yx[0]
        x = yx[1]
        cv2.rectangle(image, (x, y), (x + width, y + height), (255, 0, 153), 1)
    return image


def ncc(image, template):
    """
    Normalized Cross Correlation for
    the single pixel.
    :param image: the interested image
    :param template: the template
    :return: the computed value of
    the pixel.
    """
    dg = template - template.mean()
    df = image - image.mean()
    num = np.sum(dg * df)
    den = np.sqrt(np.sum(np.power(dg, 2)) * np.sum(np.power(df, 2)))
    return num / den


def normalized_cross_correlation(image, template, threshold=0.7):
    """
    Normalized Cross Correlation
    :param image: the image
    :param template: the template
    :param threshold: the threshold
    :return: the matched indexes and the
    image with drown rectangles.
    """
    height, width = template.shape
    image_norm = np.zeros((image.shape[0] - height + 1, image.shape[1] - width + 1), dtype=np.float64)
    for y in range(image_norm.shape[0]):
        for x in range(image_norm.shape[1]):
            image_norm[y, x] = ncc(image[y: (y + height), x: (x + width)], template)
    points = np.argwhere(image_norm >= threshold)
    yxs = get_yxs_tuples(points, template.shape)
    return yxs, draw_rectangles(image, yxs, template.shape)


def task2():
    print('\nTask 2')
    image = cv2.imread("./data/lena.png", 0)
    template = cv2.imread("./data/eye.png", 0)

    _, result_ncc = normalized_cross_correlation(image, template)
    display_image('NCC', result_ncc)


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


def template_matching_multiple_scales(pyr_image, pyr_template):
    found_yxs = []
    first_level = len(pyr_image) - 1
    final_image = pyr_image[0]
    template = pyr_template[0]

    # get the matching coordinate of the last layer
    yxs, image_match = normalized_cross_correlation(pyr_image[-1], pyr_template[-1], 0.8)
    for yx in yxs:
        found_yx = next_pyramid_iter(pyr_image, pyr_template, yx, first_level)
        # saving the found templates
        if found_yx is not None:
            found_yxs.append(found_yx)

    return draw_rectangles(final_image, found_yxs, template.shape)


def next_pyramid_iter(pyr_image, pyr_template, yx, level):
    # successive iteration of the pyramid
    level -= 1
    image = pyr_image[level]
    template = pyr_template[level]

    # get the interested area and matching the template
    yxs_area, inter_area = get_interest_area(yx, image, template.shape)
    yx_temp, image_match = normalized_cross_correlation(inter_area, template, 0.8)
    # matching not successful
    if not yx_temp:
        return None

    # update the real yx coordinate
    # from the interest coordinate
    yx = yxs_area[0] + yx_temp[0]
    # found: return the coordinate
    # of the found pattern
    if level == 0:
        return yx

    return next_pyramid_iter(pyr_image, pyr_template, yx, level)


def get_interest_area(yx, image, t_shape):
    """
    Get the interested area that contain the
    matched template.
    :param yx: starting point of the
    found template.
    :param image: image of the current layer.
    :param t_shape: shape of the current template.
    :return: return a matrix with the coordinate
    of the interested area.
    """
    t_height, t_width = t_shape
    h2, w2 = int(t_shape[0] / 2), int(t_shape[1] / 2)
    # retrieve the coordinate of the next level
    yx = np.array(yx, dtype=np.int64) * 2
    # creation of the weighted matrix
    w_match = np.array([[-h2, -w2], [h2, w2]], dtype=np.int64)
    # creation of the matrix containing the
    # coordinate of the matching template
    yxs_am = np.array([yx, [yx[0] + t_height, yx[1] + t_width]], dtype=np.int64)
    # calculation of the indexes of the interested
    # area of the current layer
    yxs = yxs_am + w_match
    # cut the interested area from the image
    area_inter = image[yxs[0, 0]: yxs[1, 0], yxs[0, 1]: yxs[1, 1]]
    return yxs, area_inter


def get_yxs_tuples(points, shape):
    """
    Get a matrix of found points.
    :param shape: shape of the template.
    :param points: iterable of points.
    :return: return a matrix with y and x as
    row for every found matching.
    """
    yxs = []
    yx_saved = (-1, -1)
    height, width = shape
    for y, x in points:
        if yx_saved == (-1, -1) or y > (yx_saved[0] + height) or x > (yx_saved[1] + width):
            yx_saved = [y, x]
            yxs.append(yx_saved)
    return yxs


def task3():
    print('\nTask 3')
    levels = 4
    image = cv2.imread("./data/traffic.jpg", 0)
    template = cv2.imread("./data/traffic-template.png", 0)

    mine_pyramid = build_gaussian_pyramid(image, levels)
    mine_pyramid_t = build_gaussian_pyramid(template, levels)
    cv_pyramid = build_gaussian_pyramid_opencv(image, levels)

    # compare and print mean absolute difference at each level
    for lvl in range(levels):
        mae = mean_abs_error(mine_pyramid[lvl], cv_pyramid[lvl])
        print('Mean abs error (level: {}): {}'.format(lvl, mae))

    start = time.time()
    _, image_match = normalized_cross_correlation(image, template)
    print('Time elapsed for NCC: {}'.format(time.time() - start))
    display_image('Time elapsed NCC', image_match)

    start = time.time()
    result = template_matching_multiple_scales(mine_pyramid, mine_pyramid_t)
    print('Time elapsed for Pyramid NCC: {}'.format(time.time() - start))
    display_image('Time elapsed NCC', result)


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
