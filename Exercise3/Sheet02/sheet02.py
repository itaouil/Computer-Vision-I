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
    yx_saved = (-1, -1)
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    for y, x in yxs:  # draw the rectangle around the matched template
        cv2.rectangle(image, (x, y), (x + width, y + height), (255, 0, 153), 1)
    return image


def ncc(image, template):
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
    :param draw: True if we want to draw a rectangle over the image.
    :param threshold: this define the threshold to get the point
    of matching.
    :return:
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

    # _, result_ncc = normalized_cross_correlation(image, template)
    # display_image('NCC', result_ncc)


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


def template_matching_multiple_scales(pyr_image, pyr_template, yx=(-1, -1), level=-1):
    first_level = len(pyr_image) - 1
    final_image = pyr_image[0]
    template = pyr_template[0]
    if level == -1:
        yxs_final = []
        # first iteration of the pyramid
        yxs, image_match = normalized_cross_correlation(pyr_image[first_level], pyr_template[first_level], 0.8)
        for yx in yxs:
            yxs_final.append(template_matching_multiple_scales(pyr_image, pyr_template, yx=yx, level=first_level))
        for yx in yxs_final:
            if yx is not None:
                final_image = draw_rectangles(final_image, yx, template.shape)
    elif level > 0 and yx != (-1, -1):
        level -= 1
        image = pyr_image[level]
        template = pyr_template[level]

        # get the interested area to match the template
        yxs_ai, area_inter = get_area_interest(yx, image, image.shape, template.shape)
        # match the template
        yx_temp, image_match = normalized_cross_correlation(area_inter, template, 0.8)

        if not yx_temp:
            # not found
            return None
        elif level == 0:
            # found
            return yxs_ai[0] + yx_temp

        # get the real yx coordinate
        # for the next layer
        yx = yxs_ai[0] + yx_temp
        yx = (yx[0, 0], yx[0, 1])
        return template_matching_multiple_scales(pyr_image, pyr_template, yx=yx, level=level)
    return final_image

def get_area_interest(yx, image, i_shape, t_shape):
    t_height, t_width = t_shape
    i_height, i_width = i_shape
    h2, w2 = int(t_height / 2), int(t_width / 2)
    # weights to get the interested area
    w_match = np.array([[-h2, -w2], [+h2, +w2]], dtype=np.int64)
    # matching area founded
    yxs_am = np.array([yx, [yx[0] + t_height, yx[1] + t_width]], dtype=np.int64)
    # calculation of the indexes of the interested
    # area of the current layer
    yxs_ai = (yxs_am + w_match) * 2

    # correct index if out of index
    if yxs_ai[0, 0] < 0:
        yxs_ai[1, 0] += abs(yxs_ai[0, 0])
        yxs_ai[0, 0] = 0
    if yxs_ai[0, 1] < 0:
        yxs_ai[1, 1] += abs(yxs_ai[0, 0])
        yxs_ai[0, 1] = 0
    if yxs_ai[1, 0] > i_height:
        dif = yxs_ai[1, 0] - i_width
        yxs_ai[0, 0] -= dif
        yxs_ai[1, 0] -= dif
    if yxs_ai[1, 1] > i_width:
        dif = yxs_ai[1, 1] - i_width
        yxs_ai[0, 1] -= dif
        yxs_ai[1, 1] -= dif

    area_inter = image[yxs_ai[0, 0]: yxs_ai[1, 0], yxs_ai[0, 1]: yxs_ai[1, 1]]
    return yxs_ai, area_inter


def get_yxs_tuples(points, shape):
    """
    Get the first y,x from a list of points.
    :param points: [[y1,...,xn],[y1,...,xn]] array that
    contains other two array with a list of points.
    :return: return an array with only two values, y and x.
    """
    yxs = []
    yx_saved = (-1, -1)
    height, width = shape
    for y, x in points:
        if yx_saved == (-1, -1) or y > (yx_saved[0] + height) or x > (yx_saved[1] + width):
            yxs.append((y, x))
            yx_saved = (y, x)
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
