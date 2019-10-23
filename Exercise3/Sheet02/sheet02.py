import cv2
import time
import sys
import numpy as np
from matplotlib import pyplot as plt

def get_gaussian_kernel(sigma, n = 0):
    """
        Compute gaussian kernel
        size given the size of the
        kernel and its standard
        deviation. If the size is omitted
        and hence is 0 it is estimated
        from the sigma.
    """
    # Our kernel and
    # its size
    kernel = None
    size = n

    # If n is known
    # populate kernel
    # with zeros
    if size != 0:
        kernel = np.zeros((size, size))
    # Otherwise estimate
    # it from the sigma
    else:
        size = int((sigma - 0.35) / 0.15)
        kernel = np.zeros((size, size))

    # Compute filter values
    # for the kernel
    for i in range(size):
        for j in range(size):
            # Adapt i and j to have
            # the correct position in
            # the kernel
            x = (i - (size - 1) // 2)
            y = (j - (size - 1) // 2)
            kernel[i][j] = np.exp(-(np.power(x, 2) + np.power(y, 2)) / (2 * np.power(sigma, 2)))

    # Normalize kernel so that
    # sum of filter values are 1
    kernel /= np.sum(kernel)

    return kernel

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
    mean_img1 = np.sum(img1)/np.size(img1)
    mean_img2 = np.sum(img2)/np.size(img2)

    return np.absolute(mean_img1 - mean_img2)

def get_convolution_using_fourier_transform(image, kernel):
    # Get image size
    h, w = image.shape

    # Get half kernel size
    hk_size = (kernel.shape[0]) // 2

    # Compute FFT of image
    # and shift fft matrix
    ftimage = np.fft.fft2(image)
    ftimage = np.fft.fftshift(ftimage)
    # print("FFT shift size: ", ftimage.shape)
    # magnitude_spectrum = 20 * np.log(np.abs(ftimage))
    # magnitude_spectrum = np.asarray(magnitude_spectrum, dtype=np.uint8)
    # display_image('Task1: Magn. Spect', magnitude_spectrum)

    # Pad kernel with 0s
    # around to be the same
    # size as the original image
    c_x, c_y = h//2, w//2
    kernel_pad = np.zeros((image.shape[0], image.shape[1]))
    kernel_pad[c_x - hk_size:c_x + hk_size+1, c_y - hk_size:c_y + hk_size+1] = kernel

    # Compute FFT of kernel
    # and shift fft matrix
    kernel_pad_fft = np.fft.fft2(kernel_pad)
    kernel_pad_fft = np.fft.fftshift(kernel_pad_fft)

    # Compute blurring in
    # spectrum domain
    blur_fft = ftimage * kernel_pad_fft

    # Compute inverse FFT
    # and inverse shift
    blur_fft = np.fft.ifftshift(blur_fft)
    blur_fft = np.fft.ifft2(blur_fft)

    # Return blurred image
    magnitude_spectrum = np.abs(blur_fft)
    return np.asarray(magnitude_spectrum, dtype=np.uint8)

def task1():
    # Read image
    image = cv2.imread("./data/einstein.jpeg", 0)
    display_image('Task1: Image', image)

    # Get gaussian kernel
    # kernel = cv2.getGaussianKernel(7, 1)
    kernel = get_gaussian_kernel(1, 7)

    # Convolute image gaussian kernel
    conv_result = cv2.filter2D(image, -1, kernel)
    display_image('Task1: Blur kernel', conv_result)

    # Convolute image using FFT
    fft_result = get_convolution_using_fourier_transform(image, kernel)
    display_image('Task1: FFT', fft_result)

    # compare results
    print("(Task1) - Mean absolute difference: ", mean_absolute_difference(conv_result, fft_result))

def sum_square_difference(image, template):
    return None

def normalized_cross_correlation(image, template):
    return None

def task2():
    image = cv2.imread("./data/lena.png", 0)
    template = cv2.imread("./data/eye.png", 0)

    result_ssd = sum_square_difference(image, template)
    result_ncc = normalized_cross_correlation(image, template)

    result_cv_sqdiff = None  # calculate using opencv
    result_cv_ncc = None  # calculate using opencv

    # draw rectangle around found location in all four results
    # show the results

def build_gaussian_pyramid_opencv(image, num_levels):
    return None

def build_gaussian_pyramid(image, num_levels, sigma):
    return None

def template_matching_multiple_scales(pyramid, template):
    return None

def task3():
    image = cv2.imread("./data/traffic.jpg", 0)
    template = cv2.imread("./data/template.jpg", 0)

    # cv_pyramid = build_gaussian_pyramid_opencv(image, 8)
    # mine_pyramid = build_gaussian_pyramid(image, 8)

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
