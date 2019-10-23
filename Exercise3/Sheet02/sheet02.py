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
    mean_img1 = np.sum(img1)/np.size(img1)
    mean_img2 = np.sum(img2)/np.size(img2)

    return np.absolute(mean_img1 - mean_img2)


def get_convolution_using_fourier_transform(image, kernel):
    # Compute FFT of image
    # and shift 0 frequency
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


def sum_square_difference(image, template):
    return None


def normalized_cross_correlation(image, template):
    # resizing template to square size
    # height, width = template.shape
    # if height > width:
    #     pad_h = (height - width) // 2
    #     template = template[pad_h: pad_h + width, 0: width]
    # else:
    #     pad_w = (width - height) // 2
    #     template = template[0: height, pad_w: pad_w + height]

    height, width = template.shape
    h2 = (height - 1) // 2
    w2 = (width - 1) // 2
    print(image.shape)
    # insert a border
    image = cv2.copyMakeBorder(
        image, h2, h2, w2, w2, cv2.BORDER_CONSTANT, value=0).astype(np.int64)
    img_norm = image.copy()

    # kernel mean
    k_sum = 0
    for y in range(height):
        for x in range(width):
            k_sum += template[y, x]
    k_mean = k_sum / (height*width)

    for y in range(h2, image.shape[0] - h2):
        print(y)
        for x in range(w2, image.shape[1] - w2):
            # image window mean
            i_sum = 0
            i_sum = np.sum(
                image[(y - h2): ((y + h2) + 1), (x - w2): ((x + w2) + 1)])
            i_mean = i_sum / (height*width)

            # get the NCC value
            num, denum, dG, dF = 0, 0, 0, 0
            for k in range(-h2, h2 + 1):
                for l in range(-w2, w2 + 1):
                    dg = template[k + h2, l + w2] - k_mean
                    df = image[y + k, x + l] - i_mean
                    num += dg * df
                    dG += pow(dg, 2)
                    dF += pow(df, 2)

            denum = pow((dG * dF), 0.5)
            val = num / denum
            img_norm[y, x] = val

    img_norm = (((img_norm - img_norm.min()) /
                 (img_norm.max() - img_norm.min())) * 255).astype(np.uint8)
    return img_norm.astype(np.uint8)


def task2():
    image = cv2.imread("./data/lena.png", 0)
    template = cv2.imread("./data/eye.png", 0)
    display_image('Image', image)
    result_ssd = sum_square_difference(image, template)
    result_ncc = normalized_cross_correlation(image, template)
    display_image('NCC', result_ncc)
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

    cv_pyramid = build_gaussian_pyramid_opencv(image, 8)
    mine_pyramid = build_gaussian_pyramid(image, 8)

    # compare and print mean absolute difference at each level
    result = template_matching_multiple_scales(pyramid, template)

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
