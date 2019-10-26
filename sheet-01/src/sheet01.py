import sys
import time
import random
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

def display_image(window_name, img):
    """
        Displays image with given window name.
        :param window_name: name of the window
        :param img: image object to display
    """
    cv.imshow(window_name, img)
    cv.waitKey(0)
    cv.destroyAllWindows()

def read_image(img_path):
    """
        Reads an image as
        a grayscale image
        and returns it.
    """
    # Read image using
    # opencv imread function
    gray_img = cv.imread(img_path, cv.IMREAD_GRAYSCALE)

    return gray_img

def integral_image(gray_img):
    """
        A recursive implementation
        to compute the integral image
        of a grayscale image.
    """
    # Get gray image shape
    height, width = gray_img.shape[:2]

    # Define integral image
    integral = np.full((height, width), None)

    # Define index i, j
    # for the recursive
    # computation
    i, j = height, width

    # Define helper function
    # for recursive computation
    def helper(i, j):
        # Base case 1:
        # Negative values are
        # not considered and should
        # just return a 0
        if i < 0 or j < 0:
            return 0
        # Base case 2:
        # Cumulative sum for that
        # index is already computed
        # and so we just return it
        elif integral[i][j] != None:
            return integral[i][j]
        # Recursive case 3:
        # Cumulative sum not already
        # computed and so we perform
        # recursive computation
        else:
            sum = helper(i-1, j) + helper(i, j-1) - helper(i-1, j-1) + gray_img[i][j]
            integral[i][j] = sum
            return sum

    # Compute integral
    # through the recursive
    # helper function
    helper(i-1, j-1)

    # Return the converted
    # integral image in order
    # to be displayed
    return integral

def mean_gray_pixel_sum(gray_img, x0, y0, x1, y1):
    """
        Compute mean gray of
        a grayscale image by
        computing the cumulative
        sum using and then dividing
        by the total number of pixels
    """
    # Return mean gray
    return int(np.sum(gray_img[x0:x1+1, y0:y1+1])) // np.size(gray_img)

def mean_gray_integral(integral, x0, y0, x1, y1):
    """
        Compute mean gray of
        a grayscale image by
        computing the cumulative
        sum using the opencv integral
        function and then dividing by
        the number of pixels
    """
    # Height and width
    # of the patch
    height = x1 - x0 + 1
    width = y1 - y0 + 1

    # Retrieve four integral
    # point for cumulative sum
    # computation
    top_left = integral[x0, y0]
    top_right = integral[x0, y1]
    bottom_left = integral[x1, y0]
    bottom_right = integral[x1, y1]

    # Compute cumulative sum
    cumulative_sum = bottom_right + top_left - bottom_left - top_right

    # Return mean gray
    return cumulative_sum // (height * width)

def equalizeHistogram(gray_img):
    """
        Equalizes grayscale
        image histogram in order
        to linearize the CDF.
    """
    # Define histogram for
    # the image as a binned
    # container of 256 slots
    # (i.e. from 0 to 255).
    # Initially containing all
    # zeroes.
    histogram = np.zeros(256)

    # Flatten gray image to be 1D
    # numpy array
    flat_gray = gray_img.flatten()

    # Update histogram based
    # on the gray image intensities
    for p in flat_gray:
        histogram[p] += 1

    # Compute CDF for the histogram
    cdf = np.cumsum(histogram)

    # Normalize histogram
    # between 0 and 255 and
    # floor and ceil floating
    # value accordingly
    cdf = np.around((255 * (cdf - cdf.min())) / (cdf.max() - cdf.min()))

    # Map old values
    # into new domain
    cdf = cdf[flat_gray]

    # Reshape our equalized
    # histogram to original one
    equalized = np.reshape(cdf, gray_img.shape)

    return equalized.astype(np.uint8)

def absolute_pixel_error(img1, img2):
    """
        Returns maximum absolute
        pixel-wise error after casting
        the images to int16 in order to
        make negative values untouched.
    """
    return np.absolute(np.subtract(img1.astype(np.int16), img2.astype(np.int16))).max()

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

def get_sep2D_gaussian_kernel(sigma, n = 0):
    """
        Computes row and column
        kernels for a gaussian.
    """
    # Row and Column
    # kernels
    kernel_sep_filter = None

    # Kernel size
    size = n

    # If n is known
    # populate kernel
    # with zeros
    if size != 0:
        kernel_sep_filter = np.zeros(size)
    # Otherwise estimate
    # it from the sigma
    else:
        size = int((sigma - 0.35) / 0.15)
        kernel_sep_filter = np.zeros(size)

    # Compute kernel_rows
    for i in range(size):
        # Adapt i to have
        # the correct position in
        # the kernel
        x = (i - (size - 1) // 2)
        kernel_sep_filter[i] = np.exp(-(np.power(x, 2)) / (2 * np.power(sigma, 2)))

    # Normalize kernel so that
    # sum of filter values are 1
    kernel_sep_filter /= np.sum(kernel_sep_filter)

    return kernel_sep_filter

def add_salt_pepper_noise(img, p):
    """
        Adds noise with a
        probability p to the
        image passed
    """
    # White or black flag
    flag = True
    # Add noise
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if random.uniform(0, 1) <= p:
                img[i][j] = 255 if flag else 0
                flag = not flag

    return img

def mean_gray_value(img1, img2):
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

def decompose_kernel(kernel):
    """
        Decomposes kernel
        using SVD to obtain
        1D kernels.
    """
    # Decompose kernel
    w, u, vt = cv.SVDecomp(kernel)

    # Find index with
    # maximum value
    i = np.argmax(w)

    # Return approximated kernel
    return [w[i], u[:, i].reshape(3,1), vt[i, :].reshape(1,3)]

if __name__ == '__main__':
    img_path = sys.argv[1]

#    =========================================================================
#    ==================== Task 1 =================================
#    =========================================================================
    print('Task 1a:')

    # Read the image
    # into grayscale
    gray_img = read_image(img_path)

    # Width and height
    # of the gray_img
    height, width = gray_img.shape[:2]

    # Compute the grayscale's
    # image integral with the
    # custom function and normalize
    # it between 0 and 255
    custom_integral = integral_image(gray_img)
    custom_integral = 255 * (custom_integral - custom_integral.min()) / (custom_integral.max() - custom_integral.min())
    display_image('1 - a', custom_integral.astype(np.uint8))

    print('Task 1b:')
    opencv_integral = cv.integral(gray_img)[1:, 1:]
    custom_integral = integral_image(gray_img)
    print("Mean gray value (i) ", mean_gray_pixel_sum(gray_img, 0, 0, 299, 479))
    print("Mean gray value (ii) ", mean_gray_integral(opencv_integral, 0, 0, 299, 479))
    print("Mean gray value (iii) ", mean_gray_integral(custom_integral, 0, 0, 299, 479))

    print('Task 1c:')

    # Generate 10 random
    # rectangle's centre
    points = [[random.randint(49, height-51), random.randint(49, width-51)] for x in range(10)]

    # Mean method 1
    t0 = time.time()
    for point in points:
        # Define top left and
        # bottom right points
        # of the rectangle
        x0 = point[0] - 49
        y0 = point[1] - 49
        x1 = point[0] + 50
        y1 = point[1] + 50

        # Compute mean method 1 (loop)
        mean_gray_pixel_sum(gray_img, x0, y0, x1, y1)

    t1 = time.time()
    print("Running time (i): ", t1 - t0, " seconds")

    # Mean method 2
    t0 = time.time()
    opencv_integral = cv.integral(gray_img)[1:, 1:]
    for x in points:
        # Define top left and
        # bottom right points
        # of the rectangle
        x0 = point[0] - 49
        y0 = point[1] - 49
        x1 = point[0] + 50
        y1 = point[1] + 50

        # Compute mean method 2 (opencv integral image)
        mean_gray_integral(opencv_integral, x0, y0, x1, y1)

    t1 = time.time()
    print("Running time (ii): ", t1 - t0)

    # Mean method 3
    t0 = time.time()
    custom_integral = integral_image(gray_img)
    for x in points:
        # Define top left and
        # bottom right points
        # of the rectangle
        x0 = point[0] - 49
        y0 = point[1] - 49
        x1 = point[0] + 50
        y1 = point[1] + 50

        # Compute mean method integral
        mean_gray_integral(custom_integral, x0, y0, x1, y1)

    t1 = time.time()
    print("Running time (iii): ", t1 - t0)

#    =========================================================================
#    ==================== Task 2 =================================
#    =========================================================================
    print('Task 2:');

    # Read the image
    # as grayscale
    gray_img = read_image(img_path)

    # Equalize image's histogram
    # using built-in opencv function
    gray_img_equalized = cv.equalizeHist(gray_img)
    display_image('2 - a', gray_img_equalized)

    # Equalize image's histogram
    # using custom function
    gray_img_equalized_custom = equalizeHistogram(gray_img)
    display_image('2 - b', gray_img_equalized_custom)

    # Output maximum pixel wise error
    print("Maximum pixel-wise error: ", absolute_pixel_error(gray_img_equalized, gray_img_equalized_custom))

#    =========================================================================
#    ==================== Task 4 =================================
#    =========================================================================
    print('Task 4:');

    # Read the image
    # as grayscale
    gray_img = read_image(img_path)
    display_image('4', gray_img)

    # Gaussian Blur
    gray_gaussian_blur = cv.GaussianBlur(gray_img.copy(), (0,0), 2 * np.power(2, 0.5))
    display_image('4 - a', gray_gaussian_blur)

    # 2D Filter Blur
    gau_kernel = get_gaussian_kernel(2 * np.power(2, 0.5), 0)
    gray_filter_2d_blur = cv.filter2D(gray_img.copy(), -1, gau_kernel)
    display_image('4 - b', gray_filter_2d_blur)

    # 2D Sep Filter Blur
    sep_kernel = get_sep2D_gaussian_kernel(2 * np.power(2, 0.5), 0)
    gray_filter_sep_blur = cv.sepFilter2D(gray_img.copy(), -1, sep_kernel.reshape(16,1), sep_kernel.reshape(1,16))
    display_image('4 - c', gray_filter_sep_blur)

    # Gaussian blur - 2D Filter blur maximum pixel error
    print("Gaussian blur - 2D Filter blur pixel error: ", absolute_pixel_error(gray_gaussian_blur, gray_filter_2d_blur))

    # Gaussian blur - 2D Sep blur maximum pixel error
    print("Gaussian blur - 2D Sep blur pixel error: ", absolute_pixel_error(gray_gaussian_blur, gray_filter_sep_blur))

    # Gaussian blur - 2D Sep blur maximum pixel error
    print("2D Filter - 2D Sep Filter pixel error: ", absolute_pixel_error(gray_filter_2d_blur, gray_filter_sep_blur))

#    =========================================================================
#    ==================== Task 5 =================================
#    =========================================================================
    print('Task 5:');

    # Read the image
    # as grayscale
    gray_img = read_image(img_path)
    display_image('5 - Gray Image', gray_img)

    # Filter image with sigma = 2
    blur_sigma_2 = cv.GaussianBlur(gray_img, (0,0), 2)
    blur_sigma_2 = cv.GaussianBlur(blur_sigma_2, (0,0), 2)
    display_image('5 - a - Filtered image with sigma = 2', blur_sigma_2)

    # Filter image with sigma = 2 * 2^(1/2)
    blur_sigma_2_sqrt = cv.GaussianBlur(gray_img, (0,0), 2 * np.power(2, 0.5))
    display_image('5 - b - Filtered image with sigma = 2 * sqrt(2)', blur_sigma_2_sqrt)

    # Sigmas maxium pixel error
    print("Sigma = 2 and Sigma = 2 * sqrt(2) pixel error: ", absolute_pixel_error(blur_sigma_2_sqrt, blur_sigma_2))

#    =========================================================================
#    ==================== Task 7 =================================
#    =========================================================================
    print('Task 7:');

    # Read the image
    # as grayscale
    gray_img = read_image(img_path)
    display_image('7 - Gray Image', gray_img)

    # Noisy image
    gray_img_copy = gray_img.copy()
    noisy_img = add_salt_pepper_noise(gray_img_copy, 0.3)
    display_image('7 - Noisy Image', noisy_img)

    # Find which kernel size
    # yields the best resutl
    best_denoised_gaussian = None
    best_denoised_median = None
    best_denoised_bilateral = None

    min_gaussian = 300
    min_median = 300
    min_bilateral = 300

    size_g = 0
    size_m = 0
    size_b = 0

    for x in [1,3,5,7,9]:

        # Denoise the image
        # with gaussian filter
        denoise_gaussian = cv.GaussianBlur(noisy_img, (x,x), 0)
        if mean_gray_value(gray_img, denoise_gaussian) < min_gaussian:
            min_gaussian = mean_gray_value(gray_img, denoise_gaussian)
            best_denoised_gaussian = denoise_gaussian
            size_g = x

        # Denoise the image
        # with median filter
        denoise_median = cv.medianBlur(noisy_img, x)
        if mean_gray_value(gray_img, denoise_median) < min_median:
            min_median = mean_gray_value(gray_img, denoise_median)
            best_denoised_median = denoise_median
            size_m = x

        # Denoise the image
        # with bilateral filter
        denoise_bilateral = cv.bilateralFilter(noisy_img, x, 80, 80)
        if mean_gray_value(gray_img, denoise_bilateral) < min_bilateral:
            min_bilateral = mean_gray_value(gray_img, denoise_bilateral)
            best_denoised_bilateral = denoise_bilateral
            size_b = x

    display_image('7 - a - Denoise with Gauss', best_denoised_gaussian)
    print("Mean distance gauss: ", size_g, min_gaussian)

    display_image('7 - a - Denoise with Median', best_denoised_median)
    print("Mean distance median: ", size_m, min_median)

    display_image('7 - a - Denoise with Bilateral', best_denoised_bilateral)
    print("Mean distance bilateral: ", size_b, min_bilateral)

#    =========================================================================
#    ==================== Task 8 =================================
#    =========================================================================
    print('Task 8:');

    # Read the image
    # as grayscale
    gray_img = read_image(img_path)

    # Kernel 1
    kernel1 = np.array([[0.0113, 0.0838, 0.0113],
                        [0.0838, 0.6193, 0.0838],
                        [0.0113, 0.0838, 0.0113]])

    # Kernel 2
    kernel2 = np.array([[-0.8984, 0.1472, 1.1410],
                        [-1.9075, 0.1566, 2.1359],
                        [-0.8659, 0.0573, 1.0337]])

    # Filter image with kernel1
    filtered_kernel1 = cv.filter2D(gray_img, -1, kernel1)
    display_image('8 - a - 2D Filter with kernel 1', filtered_kernel1)

    # Filter image with kernel2
    filtered_kernel2 = cv.filter2D(gray_img, -1, kernel2)
    display_image('8 - a- 2D Filter with kernel 2', filtered_kernel2)

    # Perform SVD of kernel1
    w1, u1, vt1 = decompose_kernel(kernel1)
    approximated_u1 = w1 * u1
    approximated_v1 = vt1

    # Filter with 1D kernel1
    # filtered_1D_kernel1 = cv.filter2D(gray_img, -1, u1)
    # filtered_1D_kernel1 = cv.filter2D(filtered_1D_kernel1, -1, w1)
    # filtered_1D_kernel1 = cv.filter2D(filtered_1D_kernel1, -1, vt1)
    filtered_1D_kernel1 = cv.sepFilter2D(gray_img, -1, approximated_u1, approximated_v1)
    display_image('8 - b - 2D Filter with kernel1 1D kernels', filtered_1D_kernel1)
    print("Maximum pixel error kernel1 2D and 1D: ", absolute_pixel_error(filtered_1D_kernel1, filtered_kernel1))

    # Perform SVD of kernel1
    w2, u2, vt2 = decompose_kernel(kernel2)
    approximated_u2 = w2 * u2
    approximated_v2 = vt2

    # Filter with 1D kernel2
    filtered_1D_kernel2 = cv.sepFilter2D(gray_img, -1, approximated_u2, approximated_v2)
    # filtered_1D_kernel2 = cv.filter2D(gray_img, -1, u2)
    # filtered_1D_kernel2 = cv.filter2D(filtered_1D_kernel2, -1, w2)
    # filtered_1D_kernel2 = cv.filter2D(filtered_1D_kernel2, -1, vt2)
    display_image('8 - b - 2D Filter with kernel2 1D kernels', filtered_1D_kernel2)
    print("Maximum pixel error kernel2 2D and 1D: ", absolute_pixel_error(filtered_1D_kernel2, filtered_kernel2))
