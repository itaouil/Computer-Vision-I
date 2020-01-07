import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
import random
import time
import sys


def integral_image(img):
    # get image size
    height, width = img.shape
    # create empty image
    integral = np.zeros((height + 1, width + 1), np.uint64)
    # integral computation
    for y in range(1, height + 1):
        row_sum = 0.0
        for x in range(1, width + 1):
            row_sum += img[y - 1, x - 1]
            integral[y, x] = row_sum + integral[y - 1, x]
    return integral[1:, 1:]


def sum_image(image):
    height, width = image.shape
    sum = 0
    for i in range(height):
        for j in range(width):
            sum += image[i, j]
    return sum


def equalize_hist_image(img):
    # get image size
    height, width = img.shape
    # create empty image
    equ = np.zeros((height, width), np.uint64)

    # Count the frequency of intensities
    hist, bins = np.histogram(img.flatten(), 256, [0, 256])

    # Normalize vector of new intensity values to sum up to a sum of 255
    hist = hist * (255. / np.sum(hist))

    # Compute integral histogram - representing the new intensity values
    for i_hist in range(1, len(hist)):
        hist[i_hist] += hist[i_hist - 1]

    # Fill the new image -> replace old intensity values with new intensities taken from the integral histogram
    for y in range(height):
        for x in range(width):
            equ[y, x] = int(hist[img[y, x]])

    return equ


def get_kernel(sigma):
    kernel_size = 2 * int(np.ceil(3.5 * sigma)) + 1
    # kernel_size += 1 if kernel_size % 2 == 0 else 0
    kernel = np.zeros((kernel_size, kernel_size))
    for i in range(-(kernel_size // 2), (kernel_size // 2) + 1):
        for j in range(-(kernel_size // 2), (kernel_size // 2) + 1):
            kernel[i + (kernel_size // 2), j + (kernel_size // 2)] = \
                np.exp(-0.5 * (i * i + j * j) / (sigma * sigma))
    return kernel / kernel.sum()


def get_kernel_1D(sigma):
    kernel_size = 2 * int(np.ceil(3.5 * sigma)) + 1
    # kernel_size += 1 if kernel_size % 2 == 0 else 0
    kernel = np.zeros(kernel_size)

    for i in range(-(kernel_size // 2), (kernel_size // 2)):
        kernel[i + (kernel_size // 2)] = np.exp(-0.5 * (i * i) / (sigma * sigma))

    return kernel / kernel.sum()


def add_salt_n_pepper_noise(img):
    # get image size
    height, width = img.shape
    for y in range(height):
        for x in range(width):
            randi = random.randint(0, 9)
            if (randi < 1):
                img[y, x] = 0
            elif (randi < 2):
                img[y, x] = 255
    return img
