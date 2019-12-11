import os

import numpy as np
from sklearn.datasets import fetch_lfw_people
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
import cv2 as cv
import sklearn
import random
import matplotlib.pylab as plt


def plot_gallery(images, titles, h, w, n_row=2, n_col=5):
    """Helper function to plot a gallery of portraits"""
    images = images.reshape((images.shape[0], h, w))
    plt.figure(figsize=(1.8 * n_col, 2.4 * n_row))
    plt.subplots_adjust(bottom=0, left=.01, right=.99, top=.90, hspace=.35)
    for i in range(n_row * n_col):
        plt.subplot(n_row, n_col, i + 1)
        plt.imshow(images[i].reshape((h, w)), cmap=plt.cm.gray)
        plt.title(titles[i], size=12)
        plt.xticks(())
        plt.yticks(())
    plt.show()


def display_image(window_name, img):
    """
        Displays image with given window name.
        :param window_name: name of the window
        :param img: image object to display
    """
    cv.imshow(window_name, img)
    cv.waitKey(0)
    cv.destroyAllWindows()


def get_transformation(eigenfaces, x_input, x_mean):
    x_diff = x_input - x_mean
    coeffs = np.dot(x_diff, eigenfaces.T).reshape((1, eigenfaces.shape[0]))
    bundle = coeffs.T * eigenfaces

    return np.sum(bundle, axis=0) + x_mean


def plot_recontruction_err(img1, img2, error, shape, type_img='none'):
    fig, axs = plt.subplots(1, 2)
    axs[0].imshow(img1.reshape(shape))
    axs[1].imshow(img2.reshape(shape))
    axs[0].set_xlabel('Input image')
    axs[1].set_xlabel('Reconstructed image')
    fig.suptitle('Detected {}\nReconstruction error: {:3.2f}'.format(type_img, error))
    fig.show()


def face_detect(eigenfaces, x_input, x_mean, h, w, threshold=45):
    # Resizing of the image
    x_input = cv.resize(x_input, (w, h), interpolation=cv.INTER_AREA).flatten()

    # Get transformation
    x_coeff = get_transformation(eigenfaces, x_input, x_mean)

    # Compute error between the input image
    norm_error = np.linalg.norm(x_input) - np.linalg.norm(x_coeff)

    type_img = 'face' if norm_error < threshold else 'other'
    # Plot images w/ error
    plot_recontruction_err(x_input, x_coeff, norm_error, (h, w), type_img)


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
    n_components = 100
    pca = PCA(n_components=n_components, svd_solver='randomized',
              whiten=True)
    pca.fit(X_train)

    # Compute mean from training data
    x_mean = X_train.mean(axis=0)

    """
    Visualize Eigen Faces
    """
    eigenfaces = pca.components_
    eigenface_titles = ["eigenface {}".format(i + 1) for i in range(eigenfaces.shape[0])]
    # TODO: remove comment
    # plot_gallery(eigenfaces, eigenface_titles, h, w)

    """
    Find the reconstruction error
    """
    # A threshold for good reconstruction error
    # that permit to detect a face is around 45.

    x_input = cv.imread('./data/exercise1/detect/face/putin.jpg', cv.IMREAD_GRAYSCALE)
    face_detect(eigenfaces, x_input, x_mean, h, w)

    """
    Perform face detection
    """
    path_face = './data/exercise1/detect/face/'
    path_other = './data/exercise1/detect/other/'
    img_faces = os.listdir(path_face)
    img_other = os.listdir(path_other)

    for img_file in img_faces:
        x_input = cv.imread(path_face + img_file, cv.IMREAD_GRAYSCALE)
        face_detect(eigenfaces, x_input, x_mean, h, w)

    for img_file in img_other:
        x_input = cv.imread(path_other + img_file, cv.IMREAD_GRAYSCALE)
        face_detect(eigenfaces, x_input, x_mean, h, w)

    """
    Perform face recognition
    """


if __name__ == '__main__':
    main()
