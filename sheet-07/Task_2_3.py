import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt


class PCA(object):

    def __init__(self, file, img):
        self.S = self.read_shapes(file)
        self.img = img
        self.mean = None
        self.W = None
        self.eigvalues = None
        self.eigvects = None

    @staticmethod
    def read_shapes(file):
        return np.loadtxt(file, skiprows=1).astype(np.uint16)

    def compute_mean(self):
        self.mean = np.array([np.mean(self.S, axis=1)]).T

    def sub_mean(self):
        self.W = self.S - self.mean

    def decomposition(self):
        self.eigvalues, self.eigvects = np.linalg.eig(np.dot(self.W, self.W.T))
        # Sort eigenvalues ascendant
        idxs = np.argsort(self.eigvalues)[::-1]
        self.eigvalues = self.eigvalues[idxs].real
        self.eigvects = self.eigvects[:, idxs].real

    def compute_k(self, threshold=0.9):
        # Compute sum of eigenvalues
        sum_eigen_values = np.sum(self.eigvalues)

        # Find for which k we meet the threshold
        for k in range(len(self.eigvalues)):
            # Compute sum of eigen values
            # till k to compute which k fits
            # best the threshold needed
            sum_eigen_k = np.sum(self.eigvalues[:k])

            # Check if threshold is met
            quotient = sum_eigen_k / sum_eigen_values
            quotient = np.around(quotient, decimals=1)
            if quotient == threshold:
                return k

        return len(self.eigvalues)

    def fit(self, k, weights):
        # weights *= 100

        # Copy mean
        model = self.mean.copy()

        # Add sum of weighted
        # eigen vectors
        for x in range(k):
            eigvect = np.array([self.eigvects[:, x]]).T
            model += eigvect * weights[x]
        
        # Plot model
        self.plot_model(np.around(model), "Model with weights: {}, and k={}".format(weights, k))

    def plot_model(self, lands1d, text, fill='green', line='red', alpha=1, with_txt=False):
        """ plots the snake onto a sub-plot
        :param lands: point locations ( [ (x0, y0), (x1, y1), ... (xn, yn)]
        :param text: text attached to the plot
        :param fill: point color
        :param line: line color
        :param alpha: [0 .. 1]
        :param with_txt: if True plot numbers as well
        :return:
        """
        # Stack horizontally x and y
        lands1d = lands1d.flatten()
        middle = lands1d.shape[0] // 2
        x_points = np.array([lands1d[:middle]]).T
        y_points = np.array([lands1d[middle:]]).T
        lands2d = np.hstack((x_points, y_points))

        # Stacks mean horizontally x and y
        mean = self.mean.flatten()
        middle = mean.shape[0] // 2
        x_points = np.array([mean[:middle]]).T
        y_points = np.array([mean[middle:]]).T
        mean2d = np.hstack((x_points, y_points))

        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111)

        ax.clear()
        # ax.imshow(self.img, cmap='gray')
        ax.set_title(text)

        V_plt = np.append(mean2d.reshape(-1), mean2d[0, :]).reshape((-1, 2))
        ax.plot(V_plt[:, 0], V_plt[:, 1], color='black', alpha=alpha)
        ax.scatter(mean2d[:, 0], mean2d[:, 1], color='black',
                   edgecolors='black',
                   linewidth=2, s=50, alpha=alpha)

        V_plt = np.append(lands2d.reshape(-1), lands2d[0, :]).reshape((-1, 2))
        ax.plot(V_plt[:, 0], V_plt[:, 1], color=line, alpha=alpha)
        ax.scatter(lands2d[:, 0], lands2d[:, 1], color=fill,
                   edgecolors='black',
                   linewidth=2, s=50, alpha=alpha)

        plt.pause(7)


def task_2():
    # Read txt file and the image
    # file = '/Users/dailand10/Desktop/Computer-Vision-I/sheet-07/data/hands_aligned_train.txt'
    # img = cv.imread("/Users/dailand10/Desktop/Computer-Vision-I/sheet-07/data/hand.jpg", 0)
    file = './data/hands_aligned_train.txt.new'
    img = cv.imread("./data/hand.jpg", 0)

    # PCA model
    model = PCA(file, img)
    model.compute_mean()
    model.sub_mean()
    model.decomposition()
    
    # Just used once to compute
    # which K conserves 90% of the
    # energy
    # k = model.compute_k()

    # Fit model with given weights
    model.fit(5, np.array([-0.4, -0.2, 0.0, 0.2, 0.4]))


def task_3():
    pass


if __name__ == '__main__':
    task_2()
