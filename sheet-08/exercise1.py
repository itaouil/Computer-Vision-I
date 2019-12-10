import numpy as np
from sklearn.datasets import fetch_lfw_people
from sklearn.model_selection import train_test_split
import cv2
import sklearn
import random
import matplotlib.pylab as plt


def main():
    random.seed(0)
    np.random.seed(0)

    # Loading the LFW dataset
    lfw = fetch_lfw_people(min_faces_per_person=70, resize=0.4)
    n_samples, h, w = lfw.images.shape
    X = lfw.data
    n_pixels = X.shape[1]
    y = lfw.target  # y is the id of the person in the image

    # splitting the data into train and test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=0)

    # Compute the PCA
    # TODO

    # Visualize Eigen Faces
    # TODO

    # Compute reconstruction error
    # TODO

    # Perform face detection
    # TODO

    # Perform face recognition
    # TODO

if __name__ == '__main__':
    main()
