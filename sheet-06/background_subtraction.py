#!/usr/bin/python3.5

import numpy as np
import cv2 as cv


def norm(x, mean, sigma, sigma_inv, sigma_det):
    """
        Computes the probability
        for a multivariate distribution
        given a k-dimensional point.

        :param x: point vector
        :param mean: mean of Multi. Norm
        :param sigma: sigma of Multi. Norm
    """
    cons = 1 / (((2 * np.pi) ** (sigma.shape[0] / 2)) * (sigma_det ** 0.5))
    exp = np.exp(- 0.5 * np.dot(np.dot((x - mean), sigma_inv), (x - mean).T))
    return cons * exp[0, 0]


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

    def __init__(self, image):
        self.thetas = []
        self.image = image

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

            :param data: 2D vector of pixels
            :return: mean (vector), sigma (vector)
        """
        # Initial lambdas
        lambda_ = 1

        # Mean vector
        mean = np.zeros((1, 3))

        # Covariance vector
        sigma = np.zeros((3, 3))

        # Compute R mean and variance
        mean[0, 0] = np.mean(data[:, 0])
        sigma[0, 0] = np.var(data[:, 0], dtype=np.float64)

        # Compute G mean and variance
        mean[0, 1] = np.mean(data[:, 1])
        sigma[1, 1] = np.var(data[:, 1], dtype=np.float64)

        # Compute B mean and variance
        mean[0, 2] = np.mean(data[:, 2])
        sigma[2, 2] = np.var(data[:, 2], dtype=np.float64)

        self.thetas.append([lambda_, mean, sigma, np.linalg.inv(sigma), np.linalg.det(sigma)])

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
        for theta in self.thetas:
            # Compute new lambdas
            lambda_ = theta[0] / 2

            # Compute theta1 mean
            mean1 = theta[1] + (epsilon * theta[2][0, 0])

            # Compute theta2 mean
            mean2 = theta[1] - (epsilon * theta[2][0, 0])

            # Append the new components
            new_thetas.append([lambda_, mean1, theta[2], theta[3], theta[4]])
            new_thetas.append([lambda_, mean2, theta[2], theta[3], theta[4]])

        # Update components list
        self.thetas = new_thetas

    def estep(self, data):
        """
            Expectation step.

            :param data: image
            :return: returns rik
        """
        # Number of Norm components
        k_length = len(self.thetas)

        # Create 3D matrix containing
        # all the ri values for every
        # single component in our MoG
        R = np.zeros((data.shape[0], k_length))

        # Normalizer
        normalizer = np.zeros((data.shape[0]))

        # Populate our matrix R
        # with just the probability
        # for each pixel point
        for x in range(data.shape[0]):
            for k in range(k_length):
                lambda_i = self.thetas[k][0]
                norm_i = norm(data[x, :], self.thetas[k][1], self.thetas[k][2],
                              self.thetas[k][3], self.thetas[k][4])
                R[x, k] = lambda_i * norm_i
                normalizer[x] += R[x, k]

        # Normalize R
        for x in range(R.shape[0]):
            R[x, :] /= normalizer[x]

        return R

    def mstep(self, data, R):
        """
            Maximization step.

            :param data: image
            :return: returns rik
        """
        # Number of Norm components
        k_length = len(self.thetas)

        # Compute lambda
        prob_sum = np.sum(R, axis=0)
        lambda_ = prob_sum / np.sum(prob_sum)

        # Mean matrix
        mean = np.zeros((k_length, 3))

        # Compute mean update
        for x in range(data.shape[0]):
            for k in range(k_length):
                mean[k] += R[x, k] * data[x]
        mean /= prob_sum.reshape(prob_sum.shape[0], 1)

        # Sigma matrix
        sigma = np.zeros((k_length, 3))
        # Compute sigma update
        for x in range(data.shape[0]):
            for k in range(k_length):
                for i in range(3):
                    sigma[k, i] += R[x, k] * (data[x, i] - mean[k, i]) ** 2

        # Nomalize sigma
        sigma /= prob_sum.reshape(prob_sum.shape[0], 1)
        sigma_diag = []
        for k in range(k_length):
            sigma_k = np.zeros((3, 3))
            sigma_k[0, 0] = sigma[k, 0]
            sigma_k[1, 1] = sigma[k, 1]
            sigma_k[2, 2] = sigma[k, 2]
            sigma_diag.append(sigma_k)

        # Update thetas (i.e. components)
        for k in range(k_length):
            self.thetas[k] = [lambda_[k], np.array([mean[k]]), sigma_diag[k],
                              np.linalg.inv(sigma_diag[k]), np.linalg.det(sigma_diag[k])]

    def probability(self, pixel):
        # get RGB probability
        prob = 0
        for k in range(len(self.thetas)):
            lambda_ = self.thetas[k][0]
            norm_ = norm(pixel, self.thetas[k][1], self.thetas[k][2],
                        self.thetas[k][3], self.thetas[k][4])
            prob += lambda_ * norm_

        # set threshold
        return self.threshold(prob)

    def sample(self):
        img_thres = np.zeros((self.image.shape[0], self.image.shape[1]))
        for y in range(self.image.shape[0]):
            for x in range(self.image.shape[1]):
                img_thres[y, x] = self.probability(self.image[y, x])
        return img_thres

    def threshold(self, prob, rho=0.87):
        return 0 if prob > rho else 255

    def train(self, data, n_splits):
        """
            Train a MoG after
            performing a MoG
            split using the EM
            algorithm.
        """
        # Fit Normal distribution
        self.fit_single_gaussian(data)

        # Split singlt Norm in MoG
        for _ in range(n_splits):
            self.split()

        # Perform EM training
        self.em_algorithm(data)

    def em_algorithm(self, data, n_iterations=10):
        """
            EM algorithm.
        """
        self.print_thetas()

        for i in range(n_iterations):
            print('Iteration: {}/{}'.format(i + 1, n_iterations))
            R = self.estep(data)
            self.mstep(data, R)
            self.print_thetas()

    def print_thetas(self):
        for k in range(len(self.thetas)):
            print('\nMixture: ', k + 1)
            print('Lambda:\t', self.thetas[k][0])
            print('Mean:\t', self.thetas[k][1])
            print('Var:\t', self.thetas[k][2])


if __name__ == '__main__':
    image, foreground, background = read_image('person.jpg')
    # display_image('', image)
    # display_image('', foreground)
    # display_image('', background)
    '''
    TODO: compute p(x|w=background) for each image pixel and manipulate the image such that everything below the threshold is black, display the resulting image
    Hint: Slide 64
    '''
    gmm_background = GMM(image)
    gmm_background.train(background, 3)
    img_thres = gmm_background.sample()
    display_image('', img_thres)
