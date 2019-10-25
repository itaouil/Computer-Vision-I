import cv2
import time
import sys
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
        s = ((f[q] + q ** 2) - (f[int(v[k])] + v[k] ** 2)) / (2*q - 2*v[k])
        while s <= z[k]:
            k = k - 1
            s = ((f[q] + q ** 2) - (f[int(v[k])] + v[k] ** 2)) / (2*q - 2*v[k])

        k = k + 1
        v[k] = q
        z[k] = s
        z[k+1] = positive_inf

    k = 0
    df = np.zeros(n)
    for q in range(n):
        while z[k+1] < q:
            k = k + 1
        df[q] = (q - v[k]) ** 2 + f[int(v[k])]

    return df

# def l2_distance_transform_1D(f, positive_inf, negative_inf):
#     n = len(f)
#     if np.allclose(f,np.repeat(positive_inf,n)):
#         return f
#     k = 0
#     v = np.zeros(n,int)
#     z = np.zeros(n)
#     z[0] = negative_inf
#     z[1] = positive_inf
#     for q in range(1,n):
#         s = f[q] + q**2 - f[v[k]] - v[k]**2
#     #    print(s)
#         s = s/(2*(q - v[k]))
#     #    print(s)
#     #    print('q',q,'k',k,'f[q]',f[q],'v[k]',v[k],'f[v[k]]',f[v[k]],'s',s)
#         while s <= z[k]:
#             k = k - 1
#             s = f[q] + q**2 - f[v[k]] - v[k]**2
#             s = s/(2*(q - v[k]))
#         k = k + 1
#         v[k] = q
#         z[k] = s
#         z[k + 1] = positive_inf
#     k = 0
#     D_f = np.zeros(n)
#     for q in range(0,n):
#         while z[k+1] < q:
#             k = k + 1
#         D_f[q] = (q-v[k])**2 + f[v[k]]
#     return D_f

def l2_distance_transform_2D(edge_function, positive_inf, negative_inf):
    # Compute distance transform
    # for each column of our function
    for x in range(edge_function.shape[1]):
        edge_function[:, x] = l2_distance_transform_1D(edge_function[:, x], positive_inf, negative_inf)

    # Compute distance transform
    # for each row of our function
    for y in range(edge_function.shape[0]):
        edge_function[y, :] = l2_distance_transform_1D(edge_function[y, :], positive_inf, negative_inf)

    return np.sqrt(edge_function)

# def l2_distance_transform_2D(edge_function, positive_inf, negative_inf):
#     D_edge_function = np.zeros(edge_function.shape)
#     n, m = edge_function.shape
#     for i in range(n):
#         D_edge_function[i,:] = l2_distance_transform_1D(edge_function[i,:],positive_inf,negative_inf)
#     for j in range(m):
#         D_edge_function[:,j] = l2_distance_transform_1D(D_edge_function[:,j],positive_inf,negative_inf)
#     return np.sqrt(D_edge_function)

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
    # task4()
    task5()
