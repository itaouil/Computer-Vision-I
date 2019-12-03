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

class IterClosePoint(object):

    def __init__(self, img, landmarks, iterations=10):
        self.img = img
        self.landmarks = landmarks
        self.iterations = iterations
        self.A = self.lands2A()
        self.B = landmarks.flatten()
        self.Psi = self.get_affine(self.A)
        self.DT = self.distance_transform()
    
    def distance_transform(self):
        """
          Compute distance transform.
        """
        return cv.distanceTransform(self.img, cv.DIST_L2, 5)

    def get_distances(self, B_p):
        """
            Compute distances
        """
        D = np.zeros((B_p.shape[0], B_p.shape[1]))

        for x in range(0, B_p.shape[0], 2):
            # Access point
            point_x = int(B_p[x])
            point_y = int(B_p[x+1])

            # Aggregate distance
            # to vector of distances
            distance = self.DT[point_y][point_x]
            D[x, 0] = distance
            D[x+1, 0] = distance
        
        return D

    def get_gradients(self, D):
        """
            Compute gradients
        """
        G = np.zeros((D.shape[0], D.shape[1]))

        # First and last value
        first_d = D[0, 0]
        last_d = D[D.shape[0]-1, 0]

        # Handle edge cases
        G[0,0] = 0.5 * (D[1,0] - last_d)
        G[D.shape[0]-1, 0] = 0.5 * (first_d - D[D.shape[0]-2, 0])

        for x in range(1, D.shape[0]-1):
            G[x, 0] = 0.5 * (D[x+1, 0] - D[x-1, 0])
        
        return D
    
    def ICP(self, B_p):
        """
            Compute ICP B_p
        """
        # Vector of distances
        # for each entry in B_p
        D = self.get_distances(B_p)

        # Vector of gradients
        # for each entry in B_p
        G = self.get_gradients(D)

        # Compute Closest Points
        CP = B_p - (D * G)

        return CP

    def fit(self):
        for iter in range(self.iterations):
            print('Iteration: {}/{}'.format(iter + 1, self.iterations))

            # Step1: Compute psi
            B_p = np.dot(self.A, self.Psi)

            # Step2: ICP
            tmp = self.ICP(B_p)

            # Step3: LSA
            self.Psi = self.get_affine(self.A)

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
    img = cv.imread("/Users/dailand10/Desktop/Computer-Vision-I/sheet-07/data/hand.jpg", 0)
    landmarks = read_landmarks('/Users/dailand10/Desktop/Computer-Vision-I/sheet-07/data/hand_landmarks.txt')

    model = IterClosePoint(img, landmarks, iterations)
    model.fit()


if __name__ == '__main__':
    task_1()
