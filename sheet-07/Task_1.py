import numpy as np
from matplotlib import pyplot as plt
import cv2 as cv

"""
  Implement iterative closest point:

  1) Read landmarks.txt file with numpy
  2) Pre-compute distance transform for hand.jpg
  3) Compute ICP for pixel Xn
  4) Estimate Phi for pixel Xn (SVD)
  5) Compute next point using estimated Phi
"""


def read_landmarks(name):
    """
        Read data from file.
    """
    landmarks = []
    with open(name) as f:
        for line in f:
            line = line.strip()
            line = line[1:-1]
            landmarks.append(line.split(','))
        landmarks_final = np.array(landmarks).astype(np.uint)

    return landmarks_final


def plot_lands(img, lands, text, fill='green', line='red', alpha=1, with_txt=False):
    """ plots the snake onto a sub-plot
    :param V: point locations ( [ (x0, y0), (x1, y1), ... (xn, yn)]
    :param fill: point color
    :param line: line color
    :param alpha: [0 .. 1]
    :param with_txt: if True plot numbers as well
    :return:
    """
    lands2d = np.reshape(lands, (lands.shape[0] // 2, 2))
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111)

    ax.clear()
    ax.imshow(img, cmap='gray')
    ax.set_title(text)

    V_plt = np.append(lands2d.reshape(-1), lands2d[0, :]).reshape((-1, 2))
    ax.plot(V_plt[:, 0], V_plt[:, 1], color=line, alpha=alpha)
    ax.scatter(lands2d[:, 0], lands2d[:, 1], color=fill,
               edgecolors='black',
               linewidth=2, s=50, alpha=alpha)
    if with_txt:
        for i, (x, y) in enumerate(lands2d):
            ax.text(x, y, str(i))

    plt.pause(5)


class IterClosePoint(object):

    def __init__(self, img, landmarks, iterations=10):
        self.img = img
        self.landmarks = landmarks
        self.iterations = iterations
        self.Psi = None
        self.B = np.array([landmarks.flatten()]).T
        self.DT = self.distance_transform()
        self.A = self.lands2A()

    def distance_transform(self):
        """
          Compute distance transform.
        """
        img_canny = cv.Canny(self.img, 60, 70)
        img_canny = (255 - img_canny) // 255

        return cv.distanceTransform(img_canny, cv.DIST_C, 5)

    def get_distances(self, B_p):
        """
            Compute distances
        """
        D = np.zeros((B_p.shape[0], B_p.shape[1]))

        for x in range(0, B_p.shape[0], 2):
            # Access point
            point_x = int(B_p[x])
            point_y = int(B_p[x + 1])

            # Aggregate distance
            # to vector of distances
            distance = self.DT[point_y][point_x]
            D[x, 0] = distance
            D[x + 1, 0] = distance

        return D

    def get_gradients(self, B_p):
        """
            Compute gradients
        """
        G_x = np.ones((B_p.shape[0], B_p.shape[1]))
        G_y = np.ones((B_p.shape[0], B_p.shape[1]))

        for l in range(0, B_p.shape[0], 2):
            # Get point
            point_x = int(B_p[l, 0])
            point_y = int(B_p[l + 1, 0])

            # Compute derivative for
            # both x and y direction
            G_x[l, 0] = .5 * (self.DT[point_y, point_x + 1] - self.DT[point_y, point_x - 1])
            G_y[l + 1, 0] = .5 * (self.DT[point_y + 1, point_x] - self.DT[point_y - 1, point_x])

        return G_x, G_y

    def ICP(self, B_p):
        """
            Compute ICP B_p
        """
        # Vector of distances for each entry in B_p
        D = self.get_distances(B_p)

        # Vector of gradients for each entry in B_p
        G_x, G_y = self.get_gradients(B_p)

        # Compute Closest Points
        CP = B_p - ((D / (np.sqrt(G_x ** 2 + G_y ** 2) + .00001)) * G_x * G_y)

        return np.around(CP)

    def fit(self):
        # Visualize the initial points.
        plot_lands(self.img, self.B, 'Initial Shape')

        # Plot B
        for iter in range(self.iterations):
            print('Iteration: {}/{}'.format(iter + 1, self.iterations))

            # ICP
            self.B = self.ICP(self.B)

    def get_affine(self):
        D, U, V_t = cv.SVDecomp(self.A)
        B_p = np.dot(U.T, self.B)
        Y = B_p / D
        X = np.dot(V_t.T, Y)
        return X

    def estimate_transf(self, plot=True):
        # Transformation
        self.Psi = self.get_affine()

        if plot:
            # Visualize the landmark points
            # using the psi transformation.
            A_stack = self.lands2A()
            B_p = np.dot(A_stack, self.Psi)
            plot_lands(self.img, B_p, 'Shape after Psi transformation')

    def lands2A(self):
        length = self.landmarks.shape[0]
        A = np.zeros((2 * length, 6))
        for idx_land in range(length):
            x, y = self.landmarks[idx_land]
            idx_row = idx_land * 2
            A[idx_row, :] = [x, y, 0, 0, 1, 0]
            A[idx_row + 1, :] = [0, 0, x, y, 0, 1]
        return A

    def plot_lands(self, lands, text, fill='green', line='red', alpha=1, with_txt=False):
        """ plots the snake onto a sub-plot
        :param V: point locations ( [ (x0, y0), (x1, y1), ... (xn, yn)]
        :param fill: point color
        :param line: line color
        :param alpha: [0 .. 1]
        :param with_txt: if True plot numbers as well
        :return:
        """
        lands2d = np.reshape(lands, (lands.shape[0] // 2, 2))
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111)

        ax.clear()
        ax.imshow(self.img, cmap='gray')
        ax.set_title(text)

        V_plt = np.append(lands2d.reshape(-1), lands2d[0, :]).reshape((-1, 2))
        ax.plot(V_plt[:, 0], V_plt[:, 1], color=line, alpha=alpha)
        ax.scatter(lands2d[:, 0], lands2d[:, 1], color=fill,
                   edgecolors='black',
                   linewidth=2, s=50, alpha=alpha)
        if with_txt:
            for i, (x, y) in enumerate(lands2d):
                ax.text(x, y, str(i))

        plt.pause(5)


def task_1():
    """
      Main.
    """
    iterations = 1
    img = cv.imread("./data/hand.jpg", 0)
    landmarks = read_landmarks('./data/hand_landmarks.txt')

    model = IterClosePoint(img, landmarks, iterations)
    model.fit()
    model.estimate_transf(plot=True)


if __name__ == '__main__':
    task_1()
