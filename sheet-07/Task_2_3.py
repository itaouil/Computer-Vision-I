import numpy as np
# import matplotlib
import cv2
from matplotlib import pyplot as plt


class PCA(object):

    def __init__(self, file, img):
        self.S = self.read_shapes(file)
        self.mean = None
        self.W = None
        self.eigvalues = None
        self.eigvects = None
        self.img = img

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
        self.eigvalues = self.eigvalues[idxs]
        self.eigvects = self.eigvects[:, idxs]
    
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
            if quotient > threshold:
                return k
        
        return len(self.eigvalues)
    
    def fit(self, k, weights):
        # Models
        models = np.zeros((self.mean.shape[0], len(weights)))

        # Iterate over weights
        for idx, weight in enumerate(weights):
            sum_ = np.array([np.sum(self.eigvects[:, :k], axis=1)]).T
            rest = self.mean + sum_ * weight
            rest_f = rest.flatten()
            models[:, idx] = rest_f
        
        # Wrap models pixels
        models = np.around(models)
        
        # Plot models
        for k in range(models.shape[1]):
            # Stack horizontally x and y
            middle = models.shape[0] // 2 
            x_points = models[:middle, k]
            y_points = models[middle:, k]
            stacked = np.hstack((x_points, y_points))

            # Plot
            self.plot_model(stacked, "Model {},{}".format(k+1, models.shape[0]))
        
        return models
    
    def plot_model(self, lands, text, fill='green', line='red', alpha=1, with_txt=False):
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

        plt.pause(1)

def task_2():
    file = '/Users/dailand10/Desktop/Computer-Vision-I/sheet-07/data/hands_aligned_train.txt'
    img = cv2.imread("/Users/dailand10/Desktop/Computer-Vision-I/sheet-07/data/hand.jpg", 0)
    # file = './data/hands_aligned_train.txt'
    # img = cv.imread("./data/hand.jpg", 0)
    
    model = PCA(file, img)
    model.compute_mean()
    model.sub_mean()
    model.decomposition()
    k = model.compute_k()
    model.fit(k, [-0.4, -0.2, 0.0, 0.2, 0.4])



def task_3():
    pass


if __name__ == '__main__':
    task_2()
