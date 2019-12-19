import numpy as np
import os
import time
import cv2 as cv
import matplotlib.pyplot as plt


def load_FLO_file(filename):
    assert os.path.isfile(filename), 'file does not exist: ' + filename
    flo_file = open(filename, 'rb')
    magic = np.fromfile(flo_file, np.float32, count=1)
    assert magic == 202021.25, 'Magic number incorrect. .flo file is invalid'
    w = np.fromfile(flo_file, np.int32, count=1)
    h = np.fromfile(flo_file, np.int32, count=1)
    data = np.fromfile(flo_file, np.float32, count=2 * w[0] * h[0])
    flow = np.resize(data, (int(h[0]), int(w[0]), 2))
    flo_file.close()
    return flow


def display_image(window_name, img):
    """
        Displays image with given window name.
        :param window_name: name of the window
        :param img: image object to display
    """
    cv.imshow(window_name, img)
    cv.waitKey(0)
    cv.destroyAllWindows()


class OpticalFlow:
    def __init__(self):
        # Parameters for Lucas_Kanade_flow()
        self.EIGEN_THRESHOLD = 0.01  # use as threshold for determining if the optical flow is valid when performing Lucas-Kanade
        self.WINDOW_SIZE = [25, 25]  # the number of points taken in the neighborhood of each pixel

        # Parameters for Horn_Schunck_flow()
        self.EPSILON = 0.002  # the stopping criterion for the difference when performing the Horn-Schuck algorithm
        self.MAX_ITERS = 1000  # maximum number of iterations allowed until convergence of the Horn-Schuck algorithm
        self.ALPHA = 1.0  # smoothness term

        # Parameter for flow_map_to_bgr()
        self.UNKNOWN_FLOW_THRESH = 1000

        self.prev = None
        self.next = None

        # Padding related vars
        self.borderType = cv.BORDER_CONSTANT

    def next_frame(self, img):
        self.prev = self.next
        self.next = img

        if self.prev is None:
            return False

        frames = np.float32(np.array([self.prev, self.next]))
        frames /= 255.0

        # calculate image gradient
        self.Ix = cv.Sobel(frames[0], cv.CV_32F, 1, 0, 3)
        self.Iy = cv.Sobel(frames[0], cv.CV_32F, 0, 1, 3)
        self.It = frames[1] - frames[0]

        return True

    def SVD(self, M, R):
        D, U, V_t = cv.SVDecomp(M)
        B_p = np.dot(U.T, R)
        if (D == 0).any():
            print(M)
            print(D)
        Y = B_p / D
        X = np.dot(V_t.T, Y)
        return X

    # ***********************************************************************************
    # function for converting flow map to to BGR image for visualisation
    # return bgr image
    def flow_map_to_bgr(self, flow):
        flow_hsv = np.zeros((flow.shape[0], flow.shape[1], 3), dtype=np.uint8)
        flow_hsv[:, :, 1] = 255

        mag, ang = cv.cartToPolar(flow[..., 0], flow[..., 1])
        flow_hsv[..., 0] = ang * 180 / np.pi / 2
        flow_hsv[..., 2] = cv.normalize(mag, None, 0, 255, cv.NORM_MINMAX)

        flow_bgr = cv.cvtColor(flow_hsv, cv.COLOR_HSV2BGR)

        return flow_bgr

    # ***********************************************************************************
    # implement Lucas-Kanade Optical Flow
    # returns the Optical flow based on the Lucas-Kanade algorithm and visualisation result
    def Lucas_Kanade_flow(self):
        kernel = np.ones((self.WINDOW_SIZE[0], self.WINDOW_SIZE[0]))

        # Flow matrix
        flow = np.zeros((self.prev.shape[0], self.prev.shape[1], 2), dtype=np.float32)

        I_xx = cv.filter2D(self.Ix ** 2, -1, kernel, self.borderType)
        I_yy = cv.filter2D(self.Iy ** 2, -1, kernel, self.borderType)
        I_yx = cv.filter2D(self.Ix * self.Iy, -1, kernel, self.borderType)
        I_tx = cv.filter2D(self.Ix * self.It, -1, kernel, self.borderType)
        I_ty = cv.filter2D(self.Iy * self.It, -1, kernel, self.borderType)

        # Iterate over pixels
        for y in range(self.prev.shape[0]):
            print(y)
            for x in range(self.prev.shape[1]):
                # Compute moment matrix
                M = np.array([
                    [I_xx[y, x], I_yx[y, x]],
                    [I_yx[y, x], I_yy[y, x]],
                ])

                # Compute right hand side matrix
                R = np.array([
                    [I_tx[y, x]],
                    [I_ty[y, x]]
                ])

                # Compute SVD to retrieve (u,v)
                motion = self.SVD(M, -R)

                # Store motion in flow
                flow[y, x] = motion.flatten()

        flow_bgr = self.flow_map_to_bgr(flow)
        return flow, flow_bgr

    # ***********************************************************************************
    # implement Horn-Schunck Optical Flow
    # returns the Optical flow based on the Horn-Schunck algorithm and visualisation result
    def Horn_Schunck_flow(self):
        flow = None
        mtx_u_t_1 = mtx_u_t = np.zeros((self.prev.shape[0], self.prev.shape[1]))
        mtx_v_t_1 = mtx_v_t = np.zeros((self.prev.shape[0], self.prev.shape[1]))
        kernel = np.array([
            [0, .25, 0],
            [.25, -1, .25],
            [0, .25, 0],
        ])
        n_iter = 0
        while self.l2_norm_error((mtx_u_t_1, mtx_u_t), (mtx_v_t_1, mtx_v_t)):
            mtx_u_t = mtx_u_t_1
            mtx_v_t = mtx_v_t_1
            up_mtx_u = mtx_u_t + cv.filter2D(mtx_u_t, -1, kernel, self.borderType)
            up_mtx_v = mtx_u_t + cv.filter2D(mtx_u_t, -1, kernel, self.borderType)
            mtx_u_t_1 = (up_mtx_u -
                         (self.Ix * (self.Ix * up_mtx_u + self.Iy * up_mtx_v + self.It)) /
                         (self.Ix ** 2 + self.Iy ** 2 + .0000001))
            mtx_v_t_1 = (up_mtx_v -
                         (self.Iy * (self.Ix * up_mtx_u + self.Iy * up_mtx_v + self.It)) /
                         (self.Ix ** 2 + self.Iy ** 2 + .0000001))
            n_iter += 1
            print('Iteration ({})'.format(n_iter))
        flow = np.dstack((mtx_u_t_1, mtx_v_t_1))
        flow_bgr = self.flow_map_to_bgr(flow)
        display_image('', flow_bgr)
        return flow, flow_bgr

    def l2_norm_error(self, mtx_u_s, mtx_v_s, threshold=0.0002):
        sum_u = np.abs(mtx_u_s[0] - mtx_u_s[1])
        sum_v = np.abs(mtx_v_s[0] - mtx_v_s[1])
        error = np.sum(sum_u - sum_v)
        print('Error: ', error)
        return error > threshold

    # ***********************************************************************************
    # calculate the angular error here
    # return average angular error and per point error map
    def calculate_angular_error(self, estimated_flow, groundtruth_flow):
        # Size of flow matrix
        size = estimated_flow.shape

        # AAE for each pixel
        aae_per_point = np.zeros((size[0], size[1]))

        # Compute per point
        for y in range(size[0]):
            for x in range(size[1]):
                # U and V error
                num_aae = (groundtruth_flow[y, x, 0] * estimated_flow[y, x, 0] +
                           groundtruth_flow[y, x, 1] * estimated_flow[y, x, 1] +
                           1)

                den_aae = np.sqrt((groundtruth_flow[y, x, 0] ** 2 + groundtruth_flow[y, x, 1] ** 2 + 1) *
                                  (estimated_flow[y, x, 0] ** 2 + estimated_flow[y, x, 1] ** 2 + 1))

                # Populate AAE per point
                aae_per_point[y, x] = np.arccos(num_aae / den_aae)

        # Compute AAE (average)
        num = aae_per_point.sum()
        den = aae_per_point.size
        aae = num / den

        return aae, aae_per_point


if __name__ == "__main__":

    path = "./"
    # path = "/Users/dailand10/Desktop/Computer-Vision-I/sheet-09/"

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
        groundtruth_flow_n = load_FLO_file(gt_list[0])
        img = cv.cvtColor(cv.imread(frame_filename), cv.COLOR_BGR2GRAY)
        if not Op.next_frame(img):
            continue

        # flow_lucas_kanade, flow_lucas_kanade_bgr = Op.Lucas_Kanade_flow()
        # aae_lucas_kanade, aae_lucas_kanade_per_point = Op.calculate_angular_error(flow_lucas_kanade, groundtruth_flow)
        # print('Average Angular error for Luacas-Kanade Optical Flow: %.4f' % (aae_lucas_kanade))

        flow_horn_schunck, flow_horn_schunck_bgr = Op.Horn_Schunck_flow()
        aae_horn_schunk, aae_horn_schunk_per_point = Op.calculate_angular_error(flow_horn_schunck, groundtruth_flow)
        print('Average Angular error for Horn-Schunck Optical Flow: %.4f' % (aae_horn_schunk))

        flow_bgr_gt = Op.flow_map_to_bgr(groundtruth_flow)

        fig = plt.figure(figsize=(img.shape))

        # Display
        fig.add_subplot(2, 3, 1)
        plt.imshow(flow_bgr_gt)
        fig.add_subplot(2, 3, 2)
        plt.imshow(flow_lucas_kanade_bgr)
        fig.add_subplot(2, 3, 3)
        plt.show()

        plt.imshow(aae_lucas_kanade_per_point)
        fig.add_subplot(2, 3, 4)
        plt.imshow(flow_bgr_gt)
        fig.add_subplot(2, 3, 5)
        plt.imshow(flow_horn_schunck_bgr)
        fig.add_subplot(2, 3, 6)
        plt.imshow(aae_horn_schunk_per_point)
        plt.show()

        print("*" * 20)
