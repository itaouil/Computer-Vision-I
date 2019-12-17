import numpy as np
import os
import time
import cv2 as cv
import matplotlib.pyplot as plt


def load_FLO_file(filename):
    assert os.path.isfile(filename), 'file does not exist: ' + filename   
    flo_file = open(filename,'rb')
    magic = np.fromfile(flo_file, np.float32, count=1)
    assert magic == 202021.25,  'Magic number incorrect. .flo file is invalid'
    w = np.fromfile(flo_file, np.int32, count=1)
    h = np.fromfile(flo_file, np.int32, count=1)
    data = np.fromfile(flo_file, np.float32, count=2*w[0]*h[0])
    flow = np.resize(data, (int(h[0]), int(w[0]), 2))
    flo_file.close()
    return flow

class OpticalFlow:
    def __init__(self):
        # Parameters for Lucas_Kanade_flow()
        self.EIGEN_THRESHOLD = 0.01 # use as threshold for determining if the optical flow is valid when performing Lucas-Kanade
        self.WINDOW_SIZE = [25, 25] # the number of points taken in the neighborhood of each pixel

        # Parameters for Horn_Schunck_flow()
        self.EPSILON= 0.002 # the stopping criterion for the difference when performing the Horn-Schuck algorithm
        self.MAX_ITERS = 1000 # maximum number of iterations allowed until convergence of the Horn-Schuck algorithm
        self.ALPHA = 1.0 # smoothness term

        # Parameter for flow_map_to_bgr()
        self.UNKNOWN_FLOW_THRESH = 1000

        self.prev = None
        self.next = None

        # Padding related vars
        self.borderType = cv.BORDER_CONSTANT
        self.padding = self.WINDOW_SIZE[0] // 2

    def next_frame(self, img):
        self.prev = self.next
        self.next = img

        if self.prev is None:
            return False

        frames = np.float32(np.array([self.prev, self.next]))
        frames /= 255.0

        #calculate image gradient
        self.Ix = cv.Sobel(frames[0], cv.CV_32F, 1, 0, 3)
        self.Iy = cv.Sobel(frames[0], cv.CV_32F, 0, 1, 3)
        self.It = frames[1]-frames[0]

        # Pad image values
        top = self.padding
        bottom = self.padding
        left = self.padding
        right = self.padding

        # Pad images and gradients
        self.prev = self.pad(self.prev, top, bottom, left, right)
        self.next = self.pad(self.next, top, bottom, left, right)
        self.Ix = self.pad(self.Ix, top, bottom, left, right)
        self.Iy = self.pad(self.Ix, top, bottom, left, right)
        self.It = self.pad(self.Ix, top, bottom, left, right)

        return True
    
    def pad(self, img, top, bottom, left, right):
        """
            Pad image.
        """
        return cv.copyMakeBorder(self.prev,
                                 top, 
                                 bottom, 
                                 left, 
                                 right, 
                                 self.borderType, 
                                 None, 
                                 0)

    def SVD(self, M, R):
        D, U, V_t = cv.SVDecomp(M)
        B_p = np.dot(U.T, R)
        Y = B_p / (D + 0.0001)
        X = np.dot(V_t.T, Y)
        return X

    #***********************************************************************************
    # function for converting flow map to to BGR image for visualisation
    # return bgr image
    def flow_map_to_bgr(self, flow):
        flow_bgr = None

        return flow_bgr

    #***********************************************************************************
    # implement Lucas-Kanade Optical Flow 
    # returns the Optical flow based on the Lucas-Kanade algorithm and visualisation result
    def Lucas_Kanade_flow(self):
        # Flow matrix
        flow = np.zeros((self.prev.shape[0], self.prev.shape[1], 2), dtype=np.float32)

        # Iterate over pixels
        for y in range(self.padding, self.prev.shape[0]):
            for x in range(self.padding, self.prev.shape[1]):
                # Matrix A
                A = np.zeros((self.WINDOW_SIZE[0], 2), dtype=np.float32)

                # Matrix B
                B = np.zeros((self.WINDOW_SIZE[0], 1), dtype=np.float32)

                # Populate A
                count = 0
                for i in range(-12, 13):
                    for j in range(-12, 13):
                        A[count, :] = [self.Ix[y + i, x + j], self.Iy[y + i, x + j]]
                        B[count, :] = self.It[y + i, x + j]
                
                # Compute moment matrix
                M = A.T @ A

                # Compute right hand side matrix
                R = -(A.T @ B)

                # Compute SVD to retrieve (u,v)
                motion = self.SVD(M, R)

                # Store motion in flow
                flow[y-self.padding, x-self.padding] = motion.flatten()
                

        flow_bgr = self.flow_map_to_bgr(flow)
        return flow, flow_bgr

    #***********************************************************************************
    # implement Horn-Schunck Optical Flow 
    # returns the Optical flow based on the Horn-Schunck algorithm and visualisation result
    def Horn_Schunck_flow(self):
        flow = None

        flow_bgr = self.flow_map_to_bgr(flow)
        return flow, flow_bgr

    #***********************************************************************************
    #calculate the angular error here
    # return average angular error and per point error map
    def calculate_angular_error(self, estimated_flow, groundtruth_flow):
        aae = None
        aae_per_point = None

        return aae, aae_per_point


if __name__ == "__main__":

    # path = "./"
    path = "/Users/dailand10/Desktop/Computer-Vision-I/sheet-09/"

    data_list = [
        path + 'data/frame_0001.png',
        path + 'data/frame_0002.png',
        path + 'data/frame_0007.png',
    ]

    gt_list = [
        path + 'data/frame_0001.flo',
        path + 'data/frame_0002.flo',
        path + 'data/frame_0007.flo',
    ]

    Op = OpticalFlow()
    
    for (i, (frame_filename, gt_filemane)) in enumerate(zip(data_list, gt_list)):
        groundtruth_flow = load_FLO_file(gt_filemane)
        img = cv.cvtColor(cv.imread(frame_filename), cv.COLOR_BGR2GRAY)
        if not Op.next_frame(img):
            continue

        flow_lucas_kanade, flow_lucas_kanade_bgr = Op.Lucas_Kanade_flow()
        aae_lucas_kanade, aae_lucas_kanade_per_point = Op.calculate_angular_error(flow_lucas_kanade, groundtruth_flow)
        print('Average Angular error for Luacas-Kanade Optical Flow: %.4f' %(aae_lucas_kanade))

        flow_horn_schunck, flow_horn_schunck_bgr = Op.Horn_Schunck_flow()
        aae_horn_schunk, aae_horn_schunk_per_point = Op.calculate_angular_error(flow_horn_schunck, groundtruth_flow)        
        print('Average Angular error for Horn-Schunck Optical Flow: %.4f' %(aae_horn_schunk))   

        flow_bgr_gt = Op.flow_map_to_bgr(groundtruth_flow)

        fig = plt.figure(figsize=(img.shape))

        # Display
        fig.add_subplot(2, 3, 1)
        plt.imshow(flow_bgr_gt)
        fig.add_subplot(2, 3, 2)
        plt.imshow(flow_lucas_kanade_bgr)
        fig.add_subplot(2, 3, 3)
        plt.imshow(aae_lucas_kanade_per_point)
        fig.add_subplot(2, 3, 4)
        plt.imshow(flow_bgr_gt)
        fig.add_subplot(2, 3, 5)
        plt.imshow(flow_horn_schunck_bgr)
        fig.add_subplot(2, 3, 6)
        plt.imshow(aae_horn_schunk_per_point)
        plt.show()

        print("*"*20)
