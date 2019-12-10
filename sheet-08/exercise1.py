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


def get_coefficients(x_input, x_mean, eigenfaces):
    x_diff = x_input - x_mean
    coeffs = np.dot(x_diff, eigenfaces.T).reshape((1, eigenfaces.shape[0]))
    bundle = coeffs.T * eigenfaces
    return bundle


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

    # Visualize Eigen Faces
    eigenfaces = pca.components_
    eigenface_titles = ["eigenface {}".format(i + 1) for i in range(eigenfaces.shape[0])]
    # plot_gallery(eigenfaces, eigenface_titles, h, w)

    # Compute reconstruction error
    x_input = cv.imread('./data/exercise1/detect/face/obama.jpg', cv.IMREAD_GRAYSCALE)
    # x_input = cv.imread('./data/exercise1/detect/other/monkey.jpg', cv.IMREAD_GRAYSCALE)
    x_input = cv.resize(x_input, (37, 50), interpolation=cv.INTER_AREA).flatten()
    x_mean = X_train.mean(axis=0)

    K_bundle = get_coefficients(x_input, x_mean, eigenfaces)
    x_coeff = (np.sum(K_bundle, axis=0) + x_mean)
    # Compute error between the input image
    input_error = (x_input - x_coeff)
    print(input_error.mean())
    norm_error = np.sqrt(np.sum(np.abs(input_error ** 2)))

    # plot figure
    fig, axs = plt.subplots(1, 4)
    axs[0].imshow(x_input.reshape((h, w)))
    axs[1].imshow(x_mean.reshape((h, w)))
    axs[2].imshow(x_coeff.reshape((h, w)))
    axs[3].imshow(input_error.reshape((h, w)))
    # axs[0].set_xlabel('fig1')
    fig.suptitle('Error: {}'.format(norm_error))
    fig.show()
    # Perform face detection
    # TODO

    # Perform face recognition
    # TODO


if __name__ == '__main__':
    main()
