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
    img1 = cv2.imread("/Users/dailand10/Desktop/Computer-Vision-I/sheet-08/data/exercise3/mountain1.png", 0)
    img2 = cv2.imread("/Users/dailand10/Desktop/Computer-Vision-I/sheet-08/data/exercise3/mountain2.png", 0)

    # Extract descriptor for both images
    sift = cv2.xfeatures2d.SIFT_create()
    (kps1, descs1) = sift.detectAndCompute(img1, None)
    (kps2, descs2) = sift.detectAndCompute(img2, None)

    # Display keypoints
    # img_1 = cv2.drawKeypoints(img1, kps1, img1)
    # img_2 = cv2.drawKeypoints(img2, kps2, img2)
    # cv2.imwrite('sift_keypoints1.jpg', img_1)
    # cv2.imwrite('sift_keypoints2.jpg', img_2)

    # Find matches and get sort idxs
    distances = euclidean_distances(descs1, descs2)
    idx_sort = np.argsort(distances, axis=1)

    # Create our matches list
    good_matches = []
    for y in range(distances.shape[0]):
        # Retrieve two best matches
        first_match = distances[y, idx_sort[y, 0]]
        second_match = distances[y, idx_sort[y, 1]]

        # Ratio value
        ratio = first_match / second_match

        # Check if ratio is above 0.4
        if ratio < 0.4:
            good_matches.append([cv2.DMatch(y, idx_sort[y, 0], first_match)])

    # Draw Matches
    matching_img = cv2.drawMatchesKnn(img1, kps1, img2, kps2, good_matches, None, flags=2)
    display_image("Matches", matching_img)


if __name__ == '__main__':
    main()
