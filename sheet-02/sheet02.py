import cv2
import time
import numpy as np
from matplotlib import pyplot as plt


def normalize(image):
    # Normalize between 0 and 255
    return ((image - image.min()) / (image.max() - image.min())) * 255


def get_gaussian_derivative(kernel, derivative_k):
    # Get kernel shape
    kH, kW = kernel.shape[:2]
    dH, dW = derivative_k.shape[:2]

    # Looping variables
    # based on derivative kernel
    y_range = kH - 1 if derivative_k.shape == (2, 1) else kH
    x_range = kW - 1 if derivative_k.shape == (1, 2) else kW

    # Output
    output = np.zeros((kH, kW))

    # Loop over kernel
    for y in range(y_range):
        for x in range(x_range):
            # Extract ROI
            roi = kernel[y:y + dH, x:x + dW]

            # Perform derivation as convolution
            k = (roi.flatten()[::-1].reshape(dH, dW) * derivative_k).sum()

            # store the convolved value in the output (x,y)-
            # coordinate of the output image
            if derivative_k.shape == (1, 2):
                output[y, x + 1] = k
            else:
                output[y + 1, x] = k

    return output


def get_gaussian_kernel(sigma, n=0):
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

    #  Compute filter values
    # for the kernel
    for i in range(size):
        for j in range(size):
            # Adapt i and j to have
            #  the correct position in
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
        Mean of absolute difference
    """
    return np.mean(np.abs(img1.astype(np.float32) - img2.astype(np.float32)))


def get_convolution_using_fourier_transform(image, kernel):
    # Get image size
    h, w = image.shape

    # Get half kernel size
    k_size = kernel.shape[0]
    hk_size = (kernel.shape[0]) // 2

    # Compute FFT of image
    #  and shift fft matrix
    ftimage = np.fft.fft2(image)

    # Pad kernel with 0s
    # around to be the same
    # size as the original image
    c_x, c_y = h // 2, w // 2
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

    # Get gaussian kernel
    kernel = get_gaussian_kernel(1, 7)

    # Convolute image gaussian kernel
    conv_result = cv2.filter2D(image, -1, kernel)

    # Convolute image using FFT
    fft_result = get_convolution_using_fourier_transform(image, kernel)

    # compare results
    print("(Task1) - Mean absolute difference: ", mean_absolute_difference(conv_result, fft_result))


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


def task4():
    # Read image in order to compute
    # the derivatives of it
    image = cv2.imread("./data/einstein.jpeg", 0)

    # Get gaussian kernel
    kernel = get_gaussian_kernel(0.6, 5)

    # Derivative kernel
    derivative_kernel = np.asarray([-1, 1]).reshape(1, 2)

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


def l2_distance_transform_1D(f, positive_inf, negative_inf):
    # Edges size
    n = f.size
    # print("N: ", n)

    if np.allclose(f, np.repeat(positive_inf, n)):
        return f

    # Variables to be used
    v = np.zeros(n)
    z = np.zeros(n + 1)

    # Set variable
    k = 0
    v[0] = 0
    z[0] = negative_inf
    z[1] = positive_inf

    for q in range(1, n):
        s = ((f[q] + q ** 2) - (f[int(v[k])] + v[k] ** 2)) / (2 * q - 2 * v[k])
        while s <= z[k]:
            k = k - 1
            s = ((f[q] + q ** 2) - (f[int(v[k])] + v[k] ** 2)) / (2 * q - 2 * v[k])

        k = k + 1
        v[k] = q
        z[k] = s
        z[k + 1] = positive_inf

    k = 0
    df = np.zeros(n)
    for q in range(n):
        while z[k + 1] < q:
            k = k + 1
        df[q] = (q - v[k]) ** 2 + f[int(v[k])]

    return df


def l2_distance_transform_2D(edge_function, positive_inf, negative_inf):
    # Compute distance transform
    #  for each column of our function
    for x in range(edge_function.shape[1]):
        edge_function[:, x] = l2_distance_transform_1D(edge_function[:, x], positive_inf, negative_inf)

    # Compute distance transform
    # for each row of our function
    for y in range(edge_function.shape[0]):
        edge_function[y, :] = l2_distance_transform_1D(edge_function[y, :], positive_inf, negative_inf)

    return np.sqrt(edge_function)


def task5():
    # Positive inf and negative inf
    positive_inf = 1000000000.0
    negative_inf = -positive_inf

    # Read traffic image
    image = cv2.imread("./data/traffic.jpg", 0)

    # Get image shape
    n, m = image.shape[:2]

    # Compute edges after blurring image
    blurred = cv2.GaussianBlur(image.copy(), (0, 0), 1)
    edges = cv2.Canny(blurred, 100, 200)

    # Edges function
    edge_function = np.zeros(edges.shape)
    edge_function[edges == 255] = 0
    edge_function[edges != 255] = positive_inf

    # Custom distance transform
    dist_transfom = l2_distance_transform_2D(edge_function, positive_inf, negative_inf)

    # OpenCV distance transform
    dist_transfom_cv = cv2.distanceTransform(image, cv2.DIST_L2, 5)

    # Mean absolute error
    print("Task 5, mean absolute difference: ", mean_absolute_difference(dist_transfom_cv, dist_transfom))

    # Distance transforms
    display_image("Dist transform CV: ", dist_transfom_cv.astype(np.uint8))
    display_image("Dist transform: ", normalize(dist_transfom).astype(np.uint8))


if __name__ == "__main__":
    # task1()
    # task2()
    # task3()
    task4()
    task5()
