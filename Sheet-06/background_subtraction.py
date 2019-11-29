#!/usr/bin/python3.5

import numpy as np
import cv2 as cv

def norm(x, mean, variance):
    """
        Computes the probability
        for a multivariate distribution
        given a k-dimensional point.

        :param x: point vector
        :param mean: mean of Multi. Norm
        :param sigma: sigma of Multi. Norm
    """
    normalizer = 1 / np.sqrt(2 * np.pi * np.power(variance, 2))
    exponential = np.exp(-np.power(x - mean, 2) / (2 * np.power(variance, 2)))
    return normalizer * exponential

def read_image(filename):
    '''
        load the image and foreground/background parts
        image: the original image
        background/foreground: numpy array of size (n_pixels, 3) (3 for RGB values), i.e. the data you need to train the GMM
    '''
    image = cv.imread(filename) / 255.0
    height, width = image.shape[:2]
    bounding_box = np.zeros(image.shape)
    bounding_box[90:350, 110:250, :] = 1
    bb_width, bb_height = 140, 260
    background = image[bounding_box == 0].reshape((height * width - bb_width * bb_height, 3))
    foreground = image[bounding_box == 1].reshape((bb_width * bb_height, 3))

    return image, foreground, background


def display_image(window_name, img):
    """
        Displays image with given window name.
        :param window_name: name of the window
        :param img: image object to display
    """
    cv.imshow(window_name, img)
    cv.waitKey(0)
    cv.destroyAllWindows()


class GMM(object):

    def __init__(self):
        self.thetas = []

    def gaussian_scores(self, data):
        # TODO
        pass

    def fit_single_gaussian(self, data):
        """
            Fitting a multivariate
            gaussian to a 3 channel
            image (RGB). The function
            will return a vector of
            means and a vector of sigmas.

            :param data: 3D vector of values
            :return: mean (vector), sigma (vector)
        """
        # Mean vector
        mean = np.zeros((1, 3))

        # Covariance vector
        sigma = np.zeros((1, 3))

        # Compute R mean and variance
        mean[0] = np.mean(data[:, :, 0])
        sigma[0] = np.var(data[:, :, 0])

        # Compute G mean and variance
        mean[1] = np.mean(data[:, :, 1])
        sigma[1] = np.var(data[:, :, 1])

        # Compute B mean and variance
        mean[2] = np.mean(data[:, :, 2])
        sigma[2] = np.var(data[:, :, 2])

        # Add multivariate
        # thetas listto
        self.thetas.append([, mean, sigma])


    def estep(self, thetas, data):
        """
            Expectation step.

            :param data: image
            :return: returns rik
        """
        # Create 3D matrix containing
        # all the ri values for every
        # single component in our MoG
        R = np.zeros((data.shape[0], len(thetas), 3))

        # Normalizer
        normalizer = np.zeros((data.shape[0], 3))

        # Populate our matrix R
        # with just the probability
        # for each pixel point
        for x in range(data.shape[0]):
            for k, theta in enumerate(self.thetas):
                for i in range(3):
                    R[x, k, i] = theta[k][0] * norm(data[x, i], theta[1][i], theta[2][i])
                    normalizer[x, i] += R[x, k, i]

        # Normalize R
        R /= normalizer

        return R


    def mstep(self, data, R):
        """
            Maximization step.

            :param data: image
            :return: returns rik
        """
        # Compute lambda update
        lambda_ = np.sum(R, axis=0)
        lambda_ /= np.sum(R)

        # Compute mean update
        mean = R * data
        mean /= np.sum()

        # Compute sigma update


    def em_algorithm(self, data, n_iterations=10):
        # TODO
        pass

    def split(self, epsilon=0.1):
        """
            Split gaussian into two
            different components, with
            different mean and sigma
            vectors.

            :param data: 3D vector of values
            :return: mean (vector), sigma (vector)
        """
        # New MoG list
        new_thetas = []

        # Split given MoG
        for component in self.thetas:
            # Compute new lambdas
            new_lambda = component[0] / 2

            # Compute component1 mean
            mean1 = component[1] + (epsilon * component[2])

            # Compute component2 mean
            mean2 = component[1] - (epsilon * component[2])

            # Append the new components
            new_thetas.append([new_lambda, mean1, component[2]])
            new_thetas.append([new_lambda, mean2, component[2]])

        # Update components list
        self.thetas = new_thetas

    def probability(self, data):
        # TODO
        pass

    def sample(self):
        # TODO
        pass

    def train(self, data, n_splits):
        # TODO
        pass


if __name__ == '__main__':
    image, foreground, background = read_image('person.jpg')
    display_image('', image)
    display_image('', foreground)
    display_image('', background)
    '''
    TODO: compute p(x|w=background) for each image pixel and manipulate the image such that everything below the threshold is black, display the resulting image
    Hint: Slide 64
    '''
    gmm_background = GMM()
