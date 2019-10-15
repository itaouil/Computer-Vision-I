import cv2 as cv
import numpy as np
import random
import sys
from numpy.random import randint

def display_image(window_name, img):
    """
    Displays image with given window name.
    :param window_name: name of the window
    :param img: image object to display
    """
    cv.imshow(window_name, img)
    cv.waitKey(0)
    cv.destroyAllWindows()

if __name__ == '__main__':
    """
    Set image path
    """
    img_path = 'bonn.png'

    """
    2a: Read and display the image
    """
    img = cv.imread(img_path)
    display_image('2 - a - Original Image', img)

    """
    2b: Display the intensity image
    """
    # Convert read image into
    # a GRAY color space
    img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    display_image('2 - b - Intensity Image', img_gray)

    """
    2c: For loop to perform the operation
    """
    # Shallow copy of the image
    # as well as store image shape
    img_cpy = img.copy()
    img_rows = img_cpy.shape[0]
    img_cols = img_cpy.shape[1]

    # Iterate over BGR image
    for row in range(img_rows):
        for col in range(img_cols):
            # Access vector element
            vector3d = img_cpy[row][col]

            # Modify pixel values
            # in the image copy
            vector3d[0] = max(vector3d[0] - 0.5 * img_gray[row][col], 0)
            vector3d[1] = max(vector3d[1] - 0.5 * img_gray[row][col], 0)
            vector3d[2] = max(vector3d[2] - 0.5 * img_gray[row][col], 0)

    display_image('2 - c - Reduced Intensity Image', img_cpy)

    """
    2d: one-line statement to perfom the operation above
    """
    # Expand img_gray dimensions to be
    # a numpy tensor of three dimensions
    img_gray = np.expand_dims(img_gray, axis=2)

    # Perform intensity image computation
    np.add(img_cpy, img_gray * 0.5)

    # Filter below zero values from image
    np.place(img_cpy, img_cpy < 0, 0)

    display_image('2 - d - Reduced Intensity Image One-Liner', img_cpy)

    """
    2e: Extract the center patch and place randomly in the image
    """
    # Retrieve 16x16 central patch
    # using slicing
    x_center = int(img_rows/2)
    y_center = int(img_cols/2)
    img_patch = img[x_center-8:x_center+8, y_center-8:y_center+8]
    display_image('2 - e - Center Patch', img_patch)

    """
    Random location of the patch for placement
    """
    # Get two random coordinates
    # to fit image patch in the image
    # copy and for the latter to be always
    # fully visible in the canvas
    rand_coord = [random.randint(0,img_rows-16),
                  random.randint(0,img_cols-16)]

    # Place image patch in
    # the original image
    img_cpy[rand_coord[0]:rand_coord[0]+16, rand_coord[1]:rand_coord[1]+16] = img_patch

    display_image('2 - e - Center Patch Placed Random %d, %d' % (rand_coord[0], rand_coord[1]), img_cpy)

    """
    2f: Draw random rectangles and ellipses
    """
    # Variable top_x and bottom_x
    # initialisation
    top_x, bottom_x = 0, 0

    # Draw 10 rectangles
    for x in range(10):
        # Update top_x and bottom_x
        # to space out rectangles
        top_x += (bottom_x - top_x + 5)
        bottom_x += (bottom_x - top_x + 50)

        # Draw rectangle
        cv.rectangle(img_cpy, (top_x, 30), (bottom_x, 100), (0,255,0), cv.FILLED)

    # Ellipses centre initialisation
    ellipses_x, ellipses_y = -20, 200

    # Draw 10 ellipses
    for x in range(10):
        # Update ellipses' centre_x
        # to space out ellipses
        ellipses_x += 45

        # Draw rectangle
        cv.ellipse(img_cpy, (ellipses_x,ellipses_y), (20,40), 0, 0, 360, 255, -1)

    display_image('2 - f - Rectangles and Ellipses', img_cpy)

    # destroy all windows
    cv.destroyAllWindows()
