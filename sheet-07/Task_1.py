import numpy as np
import matplotlib
import cv2 as cv

"""
  Implement iterative closest point:

  1) Read landmarks.txt file with numpy
  2) Pre-compute distance transform for hand.jpg
  3) Compute ICP for pixel Xn
  4) Estimate Phin for pixel Xn (SVD)
  5) Compute next point using estimated Phi
"""

def display_image(window_name, img):
    """
        Displays image with given window name.
    """
    cv.imshow(window_name, img)
    cv.waitKey(0)
    cv.destroyAllWindows()
  
  
def read_landmarks(path):
  """
    Reads landmarks points
  """
  landmarks = np.loadtxt("hand_landmarks.txt", dtype=np.str, delimiter=',')


def distance_transform(img):
  """
    Compute distance transform.
  """
  return cv.distanceTransform(img, cv2.DIST_L2, 5)


def task_1():
  """
    Main.
  """
    img = cv.imread("./data/hand.jpg", 0)


task_1()
