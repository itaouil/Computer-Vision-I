import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
import random
import time
import sys


def display_image(window_name, img):
    """
    Displays image with given window name.
    :param window_name: name of the window
    :param img: image object to display
    """
    cv.imshow(window_name, img)
    cv.waitKey(0)
    cv.destroyAllWindows()


def my_integral(img):
    # insert a border of 1 pixel
    img_integ = cv.copyMakeBorder(
        img, 1, 1, 1, 1, cv.BORDER_CONSTANT, value=0).astype(np.uint64)

    # computation of the integral image
    for i in range(img.shape[0] + 1):
        for j in range(img.shape[1] + 1):
            img_integ[i, j] = (
                (img_integ[i, j] + img_integ[i - 1, j]
                 + img_integ[i, j - 1] - img_integ[i-1, j-1]))

    # remove border of 1 pixel
    # at the bottom and right
    return img_integ[:-1, :-1]


def mean_4_image(img_mean, yx, w_shape):
    # decrease of one the dimension of the window
    w_shape = (w_shape[0] - 1, w_shape[1] - 1)
    sum = 0
    for y in range(yx[0], yx[0] + w_shape[0]):
        for x in range(yx[1], yx[1] + w_shape[1]):
            sum += img_mean[y, x]
    mean = int(sum) // np.size(img_mean)
    return mean


def mean_4_integral(img_mean, yx, w_shape):
    # decrease of one the dimension of the window
    w_shape = (w_shape[0] - 1, w_shape[1] - 1)
    a = img_mean[yx[0] + w_shape[0], yx[1] + w_shape[1]]
    b = img_mean[yx[0], yx[1] + w_shape[0]]
    c = img_mean[yx[0] + w_shape[0], yx[1]]
    d = img_mean[yx[0], yx[1]]
    sum = (a - b - c + d)
    mean = sum / np.size(img_mean)
    return mean.astype(np.uint8)


def calc_mean_exec_time(img, func_mean, YX, func_integral=None):
    start_time = time.time()
    means = []
    # if func_integral is None, img_mean is euqual to img
    # this because the mean gray value is computed by
    # summing up each pixel and not using the integral func
    img_mean = func_integral(img) if func_integral else img
    for yx in YX:
        means.append(func_mean(img_mean, yx, (square_l, square_l)))
    print("- run-time: %ss" % (time.time() - start_time))
    # we are not outputting the mean gray values because
    # it is not required
    # print(means)


def max_pwise_error(img1, img2):
    # computation of the absolute pixel wise difference
    errors = abs(img1.astype(np.int16) -
                 img2.astype(np.int16))
    return errors.max()


def gaussian_blur(img, k_size, sigma):
    if k_size == (0, 0):
        # get the kernel size extracted by the formula
        # at the link https://bit.ly/33xESq3
        k_size = int((sigma - 0.35) / 0.15)

    # computing the kernel
    kernel = np.zeros(k_size)
    for y in range(k_size[0]):
        for x in range(k_size[1]):
            a = (x - (k_size[1]-1)/2)**2
            b = (y - (k_size[0]-1)/2)**2
            num = -1 * (a + b)
            kernel[y, x] = np.exp(num/(2*sigma**2))

    # normalization
    kernel /= np.sum(kernel)
    return cv.filter2D(img, -1, kernel)


def gaussian_blur_w_sep(img, k_size, sigma):
    if k_size == (0, 0):
        # get the kernel size extracted by the formula
        # at the link https://bit.ly/33xESq3
        k_size = int((sigma - 0.35) / 0.15)

    # computing the kernel Y
    kernelY = np.zeros((k_size[0], 1))
    for y in range(k_size[0]):
        num = -1 * ((y - (k_size[0]-1)/2)**2)
        kernelY[y, 0] = np.exp(num/(2*sigma**2))

    # computing the kernel X
    kernelX = np.zeros(k_size[1])
    for x in range(k_size[1]):
        num = -1 * ((x - (k_size[1]-1)/2)**2)
        kernelX[x] = np.exp(num/(2*sigma**2))

    # normalization
    kernelY /= np.sum(kernelY[:, 0])
    kernelX /= np.sum(kernelX)
    # obtaining the final kernel
    kernel = kernelY * kernelX
    return cv.filter2D(img, -1, kernel)


def salt_n_pepper(img):
    img_sp_gaus = img.copy()
    # creation of the salt n pepper noise
    for y in range(img_sp_gaus.shape[0]):
        for x in range(img_sp_gaus.shape[1]):
            # access only the 30% of time to the pixel
            if random.uniform(0, 1) <= 0.30:
                # assign randomly 255 or 0
                img_sp_gaus[y, x] = 255 if random.randint(0, 2) else 0
    return img_sp_gaus


def distance_mean_gray_val(img1, img2):
    mean1 = (np.sum(img1.astype(np.int16)) /
             np.size(img1))
    mean2 = (np.sum(img2.astype(np.int16)) /
             np.size(img2))

    return abs(mean1 - mean2)


def filter_SVD(img, kernel):
    img_svd = img.copy()
    w, u, vt = cv.SVDecomp(kernel)
    # getting the highest singular value
    i_value = np.argmax(w)

    vt = vt[i_value, :].reshape((1, 3))
    u = u[:, i_value].reshape((3, 1)) * w[i_value, 0:1]
    # filtering the image w/ the obtained kernel
    img_svd = cv.sepFilter2D(img_svd, -1, vt, u)

    return img_svd


if __name__ == '__main__':
    np.seterr(over='ignore')
    img_path = sys.argv[1]

#    =========================================================================
#    ==================== Task 1 =================================
#    =========================================================================
    print('\nTask 1:')

    img = cv.imread(img_path, cv.IMREAD_GRAYSCALE)

    # ++++++++++++++++++++++++++++++
    # a
    # ++++

    # the function cv.integrale
    img_integ = my_integral(img)
    # normalization of the integral
    img_integ = ((img_integ - img_integ.min()) /
                 (img_integ.max() - img_integ.min()) * 255).astype(np.uint8)

    display_image('Task 1 - a', img_integ)

    # ++++++++++++++++++++++++++++++
    # b
    # ++++

    # Compute the mean grey value
    img_integ2 = cv.integral(img)
    img_integ3 = my_integral(img)
    # summing up each pixel value in the image
    mean1 = mean_4_image(img, (0, 0), img.shape)
    # computing an integral image using the function cv.integral
    mean2 = mean_4_integral(img_integ2, (0, 0), img_integ2.shape)
    # computing an integral image with your own function
    mean3 = mean_4_integral(img_integ3, (0, 0), img_integ3.shape)
    print('Mean grey value of the image (i): ', mean1)
    print('Mean grey value of the image (ii): ', mean2)
    print('Mean grey value of the image (iii): ', mean3)

    # ++++++++++++++++++++++++++++++
    # c
    # ++++

    square_l = 100
    # getting the 10 random points
    YX = [(random.randint(0, img_integ2.shape[0]-square_l),
           random.randint(0, img_integ2.shape[1]-square_l))
          for _ in range(10)]

    print('Mean gray value w/ 10 random squares (i)', end=' ')
    calc_mean_exec_time(img, mean_4_image, YX)
    print('Mean gray value w/ 10 random squares (ii)', end=' ')
    calc_mean_exec_time(img, mean_4_integral, YX, cv.integral)
    print('Mean gray value w/ 10 random squares (iii)', end=' ')
    calc_mean_exec_time(img, mean_4_integral, YX, my_integral)


#    =========================================================================
#    ==================== Task 2 =================================
#    =========================================================================
    print('\nTask 2:')

    img_eqz = cv.equalizeHist(img)

    display_image('Equalization', img_eqz)

    img_my_eqz = img.copy()
    histogram = np.zeros(256)

    # histogram creation
    for i in range(256):
        histogram[i] = np.count_nonzero(img_my_eqz == i)

    # Creation of the cumulative distribution function CDF
    cdf = np.array([np.sum(histogram[:(i+1)]) for i in range(256)])

    # normalization
    nr = np.round(((cdf - cdf.min()) / (cdf.max() - cdf.min())) * 255)

    for y in range(img.shape[0]):
        for x in range(img.shape[1]):
            img_my_eqz[y, x] = nr[img[y, x]]

    display_image('My equalization', img_my_eqz)
    error = max_pwise_error(img_eqz, img_my_eqz)
    print('Max pixel wise error (equalization): ', error)

#    =========================================================================
#    ==================== Task 4 =================================
#    =========================================================================

    print('\nTask 4:')

    sigma = (2 * (2**(1/2)))
    k_size = (3, 3)

    display_image('Gray image', img)

    img_gaus = cv.GaussianBlur(img, k_size, sigma)

    display_image('OpenCV gaussian', img_gaus)

    img_my_gaus = gaussian_blur(img, k_size, sigma)

    display_image('My gaussian', img_my_gaus)

    img_my_gaus_sep = gaussian_blur_w_sep(img, k_size, sigma)

    display_image('My gaussian w/ separability', img_my_gaus_sep)

    # computation maximum pixel wise error
    print('Maximum pixel error:')
    # OpenCV - MyGaussian
    error = max_pwise_error(img_gaus, img_my_gaus)
    print('OpenCV - MyGaussian = ', error)
    # OpenCV - MyGaussianSep
    error = max_pwise_error(img_gaus, img_my_gaus_sep)
    print('OpenCV - MyGaussianSep = ', error)
    # MyGaussian - MyGaussianSep
    error = max_pwise_error(img_my_gaus, img_my_gaus_sep)
    print('MyGaussian - MyGaussianSep = ', error)


#    =========================================================================
#    ==================== Task 5 =================================
#    =========================================================================

    print('\nTask 5:')

    sigma1 = (2)
    sigma2 = (2 * (2**(1/2)))
    k_size = (0, 0)
    img_my_gaus_1 = img.copy()
    img_my_gaus_1 = cv.GaussianBlur(
        img_my_gaus_1, k_size, sigma1)
    img_my_gaus_1 = cv.GaussianBlur(
        img_my_gaus_1, k_size, sigma1)

    display_image('My gaussian twice', img_my_gaus_1)

    img_my_gaus_2 = cv.GaussianBlur(
        img, k_size, sigma2)

    display_image('My gaussian once', img_my_gaus_2)

    # computation maximum pixel error
    error = max_pwise_error(img_my_gaus_1, img_my_gaus_2)
    print('Maximum pixel error:', error)

#    =========================================================================
#    ==================== Task 7 =================================
#    =========================================================================

    print('\nTask 7:')

    k_sizes = [7, 9]
    img_sp = salt_n_pepper(img)
    display_image('Salt n Pepper', img_sp)

    # Gaussian filtering
    gray_means = []
    for k_s in k_sizes:
        img_sp_copy = img_sp.copy()
        img_sp_gaus = cv.GaussianBlur(img_sp_copy, (k_s, k_s), 0)
        distance = distance_mean_gray_val(img, img_sp_gaus)
        gray_means.append((distance, k_s, img_sp_gaus))

    res = min(gray_means, key=lambda x: x[0])
    txt = 'SP gaussian (size: {}, mean: {:0.2f})'.format(
        res[1], res[0])
    print(txt)
    display_image(txt, res[2])

    # Median filtering
    gray_means = []
    for k_s in k_sizes:
        img_sp_copy = img_sp.copy()
        img_sp_median = cv.medianBlur(img_sp_copy, k_s)
        distance = distance_mean_gray_val(img, img_sp_median)
        gray_means.append((distance, k_s, img_sp_median))

    res = min(gray_means, key=lambda x: x[0])
    txt = 'SP median (size: {}, mean: {:0.2f})'.format(
        res[1], res[0])
    print(txt)
    display_image(txt, res[2])

    # Bilateral filtering
    gray_means = []
    for k_s in k_sizes:
        img_sp_copy = img_sp.copy()
        img_sp_bilateral = cv.bilateralFilter(
            img_sp_copy, k_s, 80, 80)
        distance = distance_mean_gray_val(img, img_sp_bilateral)
        gray_means.append((distance, k_s, img_sp_bilateral))

    res = min(gray_means, key=lambda x: x[0])
    txt = 'SP bilateral (size: {}, mean: {:0.2f})'.format(
        res[1], res[0])
    print(txt)
    display_image(txt, res[2])

#    =========================================================================
#    ==================== Task 8 =================================
#    =========================================================================

    print('\nTask 8:')

    kernel1 = np.matrix([
        [0.0113, 0.0838, 0.0113],
        [0.0838, 0.6193, 0.0838],
        [0.0113, 0.0838, 0.0113]])

    kernel2 = np.matrix([
        [-0.8984, 0.1472, 1.1410],
        [-1.9075, 0.1566, 2.1359],
        [-0.8659, 0.0573, 1.0337]])

    img_k1 = cv.filter2D(img, -1, kernel1)
    img_k1_svd = filter_SVD(img, kernel1)

    display_image('kernel1', img_k1)
    display_image('kernel1 w/ SVD', img_k1_svd)

    img_k2 = cv.filter2D(img, -1, kernel2)
    img_k2_svd = filter_SVD(img, kernel2)

    display_image('kernel2', img_k2)
    display_image('kernel2 w/ SVD', img_k2_svd)

    # computation of the pixel wise error
    error = max_pwise_error(img_k1, img_k1_svd)
    print('Pixel wise error w/ kernel1: ', error)
    error = max_pwise_error(img_k2, img_k2_svd)
    print('Pixel wise error w/ kernel2: ', error)
