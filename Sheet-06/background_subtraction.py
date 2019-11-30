#!/usr/bin/python3.5

import numpy as np
import cv2 as cv

def norm(x, mean, sigma):
    """
        Computes the probability
        for a multivariate distribution
        given a k-dimensional point.

        :param x: point vector
        :param mean: mean of Multi. Norm
        :param sigma: sigma of Multi. Norm
    """
    cons = np.power(2*np.pi, -3/2) * np.power(np.linalg.det(sigma), -0.5)
    exp = np.exp(-0.5 * (x - mean).T * np.linalg.inv(sigma) * (x - mean))
    return cons * exp

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
        sigma = np.zeros((3, 3))

        # Initial lambdas

        # Compute R mean and variance
        mean[0] = np.mean(axis=0)
        sigma[0,0] = np.var(axis=0)

        # Compute G mean and variance
        mean[1] = np.mean(axis=1)
        sigma[1,1] = np.var(axis=1)

        # Compute B mean and variance
        mean[2] = np.mean(axis=2)
        sigma[2,2] = np.var(axis=2)

        # Add multivariate 
        # thetas listto
        self.thetas.append([3 * [1/3], mean, sigma])


    def estep(self, data):
        """
            Expectation step.

            :param data: image
            :return: returns rik
        """
        # Number of Norm components
        k = len(self.thetas)

        # Create 3D matrix containing
        # all the ri values for every
        # single component in our MoG
        R = np.zeros((data.shape[0], k))

        # Normalizer
        normalizer = np.zeros((data.shape[0], k))

        # Populate our matrix R
        # with just the probability
        # for each pixel point
        for x in range(data.shape[0]):
            for k in range(k):
                R[x, k] = self.thetas[k][0] * norm(data[x, :], self.thetas[k][1], self.thetas[k][2])
                normalizer[x, k] += R[x, k]
        
        # Normalize R
        R /= normalizer

        return R


    def mstep(self, data, R):
        """
            Maximization step.

            :param data: image
            :return: returns rik
        """
        # Number of Norm components
        k = len(self.thetas)

        # Common probability sum
        prob_sum = np.zeros((k, 3))
        

        # Compute lambda update
        for x in range(data.shape[0]):
            for k in range(k):
                for i in range(3):
                    prob_sum[k, i] += R[x,k,i]
        
        # Nomalize lambda_
        lambda_ = prob_sum / np.sum(prob_sum, axis=0)
        

        # Mean matrix
        mean = np.zeros((k, 3))
        
        # Compute mean update
        for x in range(data.shape[0]):
            for k in range(k):
                for i in range(3):
                    mean[k, i] += R[x,k,i] * data[x, i]
        
        # Nomalize mean
        mean /= prob_sum

        
        # Sigma matrix
        sigma = np.zeros((k, 3))
        
        # Compute sigma update
        for x in range(data.shape[0]):
            for k in range(k):
                for i in range(3):
                    sigma[k, i] += R[x,k,i] * (data[x, :] - mean[k, :]) * (data[x, :] - mean[k, :]).T
        
        # Nomalize sigma
        sigma /= prob_sum

        
        # Update thetas (i.e. components)
        for k in range(k):
            self.thetas[k] = [lambda_[k], mean[k], sigma[k]]


    def em_algorithm(self, data, n_iterations=10):
        """
            EM algorithm.
        """
        for _ in range(n_iterations):
            R = self.estep(data)
            self.mstep(data, R)
            

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
            new_lambda = sum(theta[0]) / 2

            # Compute theta1 mean
            mean1 = theta[1] + (epsilon * theta[2])

            # Compute theta2 mean
            mean2 = theta[1] - (epsilon * theta[2])

            # Append the new components
            new_thetas.append([3 * [new_lambda/3], mean1, theta[2]])
            new_thetas.append([3 * [new_lambda/3], mean2, theta[2]])

        # Update components list
        self.thetas = new_thetas

    def probability(self, data):
        # TODO
        pass

    def sample(self):
        # TODO
        pass

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