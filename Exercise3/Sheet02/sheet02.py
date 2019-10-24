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


def apply_border(image, template):
    height, width = template.shape
    h2 = (height - 1) // 2
    w2 = (width - 1) // 2

    image = cv2.copyMakeBorder(
        image, h2, h2, w2, w2, cv2.BORDER_CONSTANT, value=0).astype(np.float)
    return image, height, width, h2, w2


def remove_border(image, h2, w2):
    return image[h2: image.shape[0] - h2, w2: image.shape[1] - w2].astype(np.uint8)


def normalize(image):
    return (((image - image.min()) /
             (image.max() - image.min())) * 255).astype(np.uint8)


def threshold(image, foreground=255):
    perc_70 = int(255 * 0.7)
    return np.where(image >= perc_70, foreground, 0).astype(np.uint8)


def draw_rectangles(image, positions, h2, w2):
    point_saved = (-1, -1)
    for point in zip(*positions[::-1]):  # draw the rectangle around the matched template
        if point_saved == (-1, -1) or point[0] > (point_saved[0] + h2) or point[1] > (point_saved[1] + w2):
            cv2.rectangle(image, (point[0] - w2, point[1] - h2), (point[0] + w2, point[1] + h2), (0, 204, 153), 1)
            point_saved = point
    return image


def sum_square_difference(image, template):
    foreground = 255
    image, height, width, h2, w2 = apply_border(image, template)
    image_sum = image.copy()

    for y in range(h2, image.shape[0] - h2):
        print(y)
        for x in range(w2, image.shape[1] - w2):
            sum = 0
            for k in range(-h2, h2 + 1):
                for l in range(-w2, w2 + 1):
                    sum += (template[k, l] - image[y + k, x + l]) ** 2
            image_sum[y, x] = sum

    image_sum = normalize(image_sum)
    image_thres = threshold(image_sum, foreground)
    image_thres = remove_border(image_thres, h2, w2)
    positions = np.where(image_thres == foreground)
    image = draw_rectangles(image, positions, h2, w2)
    return image


def cross_correlation(image_w, template):
    dg = template - template.mean()
    df = image_w - image_w.mean()
    num = np.sum(dg * df)
    den = dg.std() * df.std()
    return num / den


def normalized_cross_correlation(image, template):
    foreground = 255
    image_copy = image.copy()
    image_copy, height, width, h2, w2 = apply_border(image_copy, template)
    image_norm = image_copy.copy()
    # template mean
    t_mean = np.sum(template[height: width]) / (height * width)

    # NCC computation
    for y in range(h2, image_copy.shape[0] - h2):
        print(y)
        for x in range(w2, image_copy.shape[1] - w2):
            # i_window = image_norm[(y - h2): ((y + h2) + 1), (x - w2): ((x + w2) + 1)]
            # image_norm[y, x] = cross_correlation(i_window, template)
            # image window mean
            i_mean = np.sum(
                image_copy[(y - h2): ((y + h2) + 1), (x - w2): ((x + w2) + 1)]) / (height * width)

            # get the NCC value
            num, denum, dG, dF = 0, 0, 0, 0
            for k in range(-h2, h2 + 1):
                for l in range(-w2, w2 + 1):
                    dg = template[k + h2, l + w2] - t_mean
                    df = image_copy[y + k, x + l] - i_mean
                    num += dg * df
                    dG += pow(dg, 2)
                    dF += pow(df, 2)

            denum = pow((dG * dF), 0.5)
            image_norm[y, x] = num / denum
    # -- (end) NCC computation

    image_norm = normalize(image_norm)
    display_image('NCC', image_norm)

    image_thres = threshold(image_norm, foreground)
    image_thres = remove_border(image_thres, h2, w2)
    positions = np.where(image_thres == foreground)
    image = draw_rectangles(image, positions, h2, w2)
    return image


def task2():
    image = cv2.imread("./data/lena.png", 0)
    template = cv2.imread("./data/eye.png", 0)

    result_ncc = normalized_cross_correlation(image, template)
    display_image('NCC', result_ncc)


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
