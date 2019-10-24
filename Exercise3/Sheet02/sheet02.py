import cv2
import time
import sys
import numpy as np
from matplotlib import pyplot as plt
# np.set_printoptions(precision=5)

def get_gaussian_derivative(kernel, derivative_k):
    # Get kernel shape
    kH, kW = kernel.shape[:2]
    dH, dW = derivative_k.shape[:2]

    # Looping variables
    # based on derivative kernel
    y_range = kH - 1 if derivative_k.shape == (2,1) else kH
    x_range = kW - 1 if derivative_k.shape == (1,2) else kW

    # Output
    output = np.zeros((kH, kW))

    # Loop over kernel
    for y in range(y_range):
        for x in range(x_range):
            # Extract ROI
            roi = kernel[y:y+dH, x:x+dW]

            # Perform derivation as convolution
            k = (roi.flatten()[::-1].reshape(dH,dW) * derivative_k).sum()

            # store the convolved value in the output (x,y)-
            # coordinate of the output image
            if derivative_k.shape == (1,2):
                output[y, x+1] = k
            else:
                output[y+1, x] = k

    return output

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

    return int(np.around(np.absolute(mean_img1 - mean_img2)))

def get_convolution_using_fourier_transform(image, kernel):
    # Get image size
    h, w = image.shape

    # Get half kernel size
    k_size = kernel.shape[0]
    hk_size = (kernel.shape[0]) // 2

    # Compute FFT of image
    # and shift fft matrix
    ftimage = np.fft.fft2(image)

    # Pad kernel with 0s
    # around to be the same
    # size as the original image
    c_x, c_y = h//2, w//2
    kernel_pad = np.zeros((image.shape[0], image.shape[1]))
    kernel_pad[0:k_size, 0:k_size] = kernel

    # Compute FFT of kernel
    # and shift fft matrix
    kernel_pad_fft = np.fft.fft2(kernel_pad)

    # Compute blurring in
    # spectrum domain
    blur_fft = ftimage * kernel_pad_fft

    # Compute inverse FFT
    # and inverse shift
    blur_fft = np.fft.ifft2(blur_fft)

    # Return blurred image
    return np.asarray(np.abs(blur_fft), dtype=np.uint8)

def task1():
    # Read image
    image = cv2.imread("./data/einstein.jpeg", 0)
    # display_image('Task1: Image', image)

    # Get gaussian kernel
    # kernel = cv2.getGaussianKernel(7, 1)
    kernel = get_gaussian_kernel(1, 7)

    # Convolute image gaussian kernel
    conv_result = cv2.filter2D(image, -1, kernel)
    # display_image('Task1: Blur kernel', conv_result)

    # Convolute image using FFT
    fft_result = get_convolution_using_fourier_transform(image, kernel)
    # display_image('Task1: FFT', fft_result)

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

def task4():
    # Read image in order to compute
    # the derivatives of it
    image = cv2.imread("./data/einstein.jpeg", 0)

    # Get gaussian kernel
    kernel = get_gaussian_kernel(0.6, 5)

    # Derivative kernel
    derivative_kernel = np.asarray([-1, 1]).reshape(1,2)

    # Get derivatives for x
    # and y of the kernel
    kernel_x = get_gaussian_derivative(kernel, derivative_kernel)
    kernel_y = get_gaussian_derivative(kernel, derivative_kernel.T)

    # Compute derivative for x and y direction of image
    edges_x = cv2.filter2D(image.copy(), -1, kernel_x)
    edges_y = cv2.filter2D(image.copy(), -1, kernel_y)

    # Compute magnitude and direction
    magnitude = (edges_y ** 2 + edges_x ** 2) ** 0.5
    direction = np.arctan2(edges_y, edges_x)

    # Normalize magnitude and direction
    magnitude = ((magnitude - magnitude.min()) / (magnitude.max() - magnitude.min())) * 255
    direction = ((direction - direction.min()) / (direction.max() - direction.min())) * 255

    display_image("Magnitude", magnitude.astype(np.uint8))
    display_image("Direction", direction.astype(np.uint8))

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
    # task5()
