import numpy as np
import matplotlib
import cv2


class PCA(object):

    def __init__(self, file):
        self.S = self.read_shapes(file)
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
        print(self.eigvalues, self.eigvects)


def task_2():
    # file = '/Users/dailand10/Desktop/Computer-Vision-I/sheet-07/data/hands_aligned_train.txt'
    file = './data/hands_aligned_train.txt'
    model = PCA(file)
    model.compute_mean()
    model.sub_mean()
    model.decomposition()


def task_3():
    pass


if __name__ == '__main__':
    task_2()
