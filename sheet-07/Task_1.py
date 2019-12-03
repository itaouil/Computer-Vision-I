import numpy as np
import matplotlib
import cv2 as cv

"""
  Implement iterative closest point:

  1) Read landmarks.txt file with numpy
  2) Pre-compute distance transform for hand.jpg
  3) Compute ICP for pixel Xn
  4) Estimate Phin for pixel Xn (SVD)
  5) Compute next point using estimated Phi
"""


def display_image(window_name, img):
    """
        Displays image with given window name.
    """
    cv.imshow(window_name, img)
    cv.waitKey(0)
    cv.destroyAllWindows()


def read_landmarks(name):
    """Read data from file.
    """
    landmarks = []
    with open(name) as f:
        for line in f:
            line = line.strip()
            line = line[1:-1]
            landmarks.append(line.split(','))
        landmarks_final = np.array(landmarks).astype(np.uint)

    return landmarks_final


def distance_transform(img):
    """
      Compute distance transform.
    """
    return cv.distanceTransform(img, cv2.DIST_L2, 5)


class IterClosePoint(object):

    def __init__(self, img, landmarks, iterations=10):
        self.img = img
        self.landmarks = landmarks
        self.iterations = iterations
        self.B = landmarks.flatten()

    def fit(self):
        A = self.lands2A()
        for iter in range(self.iterations):
            print('Iteration: {}/{}'.format(iter + 1, self.iterations))

            # step 3
            Psi = self.get_affine(A)

    def get_affine(self, A):
        D, U, V_t = cv.SVDecomp(A)
        B_p = np.dot(U.T, self.B)
        B_p = np.array([B_p])
        Y = B_p.T / D
        X = np.dot(V_t.T, Y)
        return X

    def lands2A(self):
        length = self.landmarks.shape[0]
        A = np.zeros((2 * length, 6 * length))
        for idx_land in range(length):
            x, y = self.landmarks[idx_land]
            idx_row = idx_land * 2
            idx_col = idx_land * 6
            A[idx_row, idx_col:idx_col + 6] = [x, y, 0, 0, 1, 0]
            A[idx_row + 1, idx_col:idx_col + 6] = [0, 0, x, y, 0, 1]
        return A


def task_1():
    """
      Main.
    """
    iterations = 5
    img = cv.imread("./data/hand.jpg", 0)
    landmarks = read_landmarks('./data/hand_landmarks.txt')

    model = IterClosePoint(img, landmarks, iterations)
    model.fit()


if __name__ == '__main__':
    task_1()
