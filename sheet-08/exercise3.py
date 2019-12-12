import cv2
import numpy as np
import matplotlib.pylab as plt
from sklearn.metrics.pairwise import euclidean_distances

def display_image(window_name, img):
    """
        Displays image with given window name.
        :param window_name: name of the window
        :param img: image object to display
    """
    cv2.imshow(window_name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def main():
    # Load the images
    img1 = cv2.imread("./data/exercise3/mountain1.png")
    img2 = cv2.imread("./data/exercise3/mountain2.png")

    # extract sift keypoints and descriptors
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    sift = cv2.xfeatures2d.SIFT_create()
    kps1, descs1 = sift.detectAndCompute(gray1, None)
    kps2, descs2 = sift.detectAndCompute(gray2, None)

    # your own implementation of matching
    distances = euclidean_distances(descs1, descs2)
    ind_dists = np.argsort(distances, axis=1)
    print(ind_dists)

    # display the matches

    pass


if __name__ == '__main__':
    main()
