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

def mean_gray_for_loop(gray_img, x0, y0, x1, y1):
    """
        Compute mean gray of
        a grayscale image by
        computing the cumulative
        sum using a for loop and
        then dividing by the total
        number of pixels
    """
    # Total sum
    cumulative_sum = 0

    # Height and width
    # of the patch
    height = x1 - x0 + 1
    width = y1 - y0 + 1

    # Compute cumulative sum
    for i in range(x0, x1+1):
        for j in range(y0, y1+1):
            cumulative_sum += gray_img[i][j]

    # Return mean gray
    return cumulative_sum / (height * width)

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
    return cumulative_sum / (height * width)

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
    flat_gray = np.asarray(gray_img).flatten()

    # Update histogram based
    # on the gray image intensities
    for p in flat_gray:
        histogram[p] += 1

    # Compute CDF for the histogram
    cdf = [histogram[0]]
    for x in range(1, len(histogram)):
        cdf.append(cdf[-1] + histogram[x])

    # Cast cumulative sum
    # to numpy array
    cdf = np.asarray(cdf)

    # Normalize histogram
    # between 0 and 255 and
    # cast it to uint8
    numerator = (cdf - cdf.min()) * 255
    denominator = (cdf.max() - cdf.min())
    cdf = numerator / denominator
    cdf = cdf.astype("uint8")
    cdf = cdf[flat_gray]

    # Reshape our equalized
    # histogram
    equalized = np.reshape(cdf, gray_img.shape)

    return equalized

def absolute_pixel_error(img1, img2):
    """
        Returns maximum absolute
        pixel-wise error.
    """
    return np.absolute(np.array(img1) - np.array(img2)).max()


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
    # custom function
    custom_integral = integral_image(gray_img).astype(np.uint8)
    display_image('1 - a - Integral image using custom function', custom_integral)

    # Compute the grayscale's
    # image integral with opencv
    # integral function
    opencv_integral = cv.integral(gray_img).astype(np.uint8)[1:, 1:]
    display_image('1 - a - Integral image using cv.integral', opencv_integral)

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
        mean_grays_for_loop = mean_gray_for_loop(gray_img, x0, y0, x1, y1)

    t1 = time.time()
    print("Mean method1 running time: ", t1 - t0, mean_grays_for_loop, "\n")

    # Mean method 2
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
        mean_gray_custom_integral = mean_gray_integral(custom_integral, x0, y0, x1, y1)

    t1 = time.time()
    print("Mean method2 running time: ", t1 - t0, mean_gray_custom_integral, "\n")

    # Mean method 3
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

        # Compute mean method 3 (opencv integral image)
        mean_gray_opencv_integral = mean_gray_integral(opencv_integral, x0, y0, x1, y1)

    t1 = time.time()
    print("Mean method3 running time: ", t1 - t0, mean_gray_opencv_integral, "\n")

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
    display_image('2 - a - OpenCV histogram equalization', gray_img_equalized)

    # Equalize image's histogram
    # using custom function
    gray_img_equalized_custom = equalizeHistogram(gray_img)
    display_image('2 - b - Custom histogram equalization', gray_img_equalized_custom)

    # Output maximum pixel wise error
    print("Are equal: ", np.array_equal(gray_img_equalized, gray_img_equalized_custom))
    print("Maximum pixel-wise error: ", absolute_pixel_error(gray_img_equalized, gray_img_equalized_custom))

#    =========================================================================
#    ==================== Task 4 =================================
#    =========================================================================
    print('Task 4:');

    # Read the image
    # as grayscale
    gray_img = read_image(img_path)
    display_image('4 - Gray Image', gray_img)

    # Gaussian blur
    gray_gaussian_blur = cv.GaussianBlur(gray_img, (0,0), 2 * (2 ** 1/2))
    display_image('4 - a - Gray Image (Gaussian Blur)', gray_gaussian_blur)


#    =========================================================================
#    ==================== Task 6 =================================
#    =========================================================================
    print('Task 6:');





#    =========================================================================
#    ==================== Task 7 =================================
#    =========================================================================
    print('Task 7:');





#    =========================================================================
#    ==================== Task 8 =================================
#    =========================================================================
    print('Task 8:');
