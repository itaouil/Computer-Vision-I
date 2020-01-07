import cv2
import numpy as np
import os
import os.path as osp
import matplotlib.pyplot as plt

NUM_IMAGES = 14
NUM_Boards = NUM_IMAGES
# image_prefix = "images/"
image_prefix = "/Users/dailand10/Desktop/Computer-Vision-I/sheet-10/images/"
image_suffix = ".png"
images_files_list = [osp.join(image_prefix, f) for f in os.listdir(image_prefix)
                     if osp.isfile(osp.join(image_prefix, f)) and f.endswith(image_suffix)]
board_w = 10
board_h = 7
board_size = (board_w, board_h)
board_n = board_w * board_h
img_shape = (0, 0)
obj = []
for ptIdx in range(0, board_n):
    obj.append(np.array([[ptIdx / board_w, ptIdx % board_w, 0.0]], np.float32))
obj = np.vstack(obj)


def task1():
    """
        Detect and display
        corners for each image
    """
    global img_shape
    # Arrays to store object points and image points from all the images.
    objpoints = []  # 3d point in real world space
    imgpoints = []  # 2d points in image plane.

    # termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # Iterate over set of images
    for img_file in images_files_list:
        # Read image as grayscale
        img = cv2.imread(img_file)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_shape = gray.shape
        # Find the chessboard corners
        found, corners = cv2.findChessboardCorners(gray, (board_w, board_h), None)

        # Check if corners found
        # and display them
        if found:
            # Append 3D object points
            objpoints.append(obj)

            # Refine detected corner
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners2)

            # Draw and display the corners
            # img = cv2.drawChessboardCorners(img, (board_w, board_h), corners2, found)
            # cv2.imshow('img', img)
            # cv2.waitKey(500)

    return imgpoints, objpoints


def task2(imagePoints, objectPoints):
    ret, CM, D, rvecs, tvecs = cv2.calibrateCamera(objectPoints, imagePoints, img_shape[::-1], None, None)
    print('cameraMatrix:\n', CM, end='\n\n')
    print('distortionMatrix:\n', D, end='\n\n')
    print('rotation:\n', rvecs, end='\n\n')
    print('translation:\n', tvecs, end='\n\n')
    return CM, D, rvecs, tvecs


def task3(imagePoints, objectPoints, CM, D, rvecs, tvecs):
    """
        Compute reprojection error
    """
    # Reprojection error
    error = 0

    for c, img_file in enumerate(images_files_list):
        # Read image
        img = cv2.imread(img_file)

        # Compute 3D reprojection
        # for every single keypoint
        # of every image
        imagePointsProjection, _ = cv2.projectPoints(objectPoints[c], 
                                                    rvecs[0], 
                                                    tvecs[0], 
                                                    CM, 
                                                    D)

        # print(imagePoints[c][1][0][0])

        for k in range(board_w * board_h):
            # Squeeze arrays
            imagePointsSqueeze = np.squeeze(imagePoints)
            imagePointsProjSqueeze = np.squeeze(imagePointsProjection)

            print(imagePointsProjSqueeze[k])
            print(imagePointsProjSqueeze[k])

            # Draw ground-truth image points
            cv2.circle(img, 
                    (imagePointsSqueeze[k][0], imagePointsSqueeze[k][1]),
                    20,
                    (0, 255, 0),
                    1)
            
            # Draw reprojected image points
            cv2.circle(img, 
                    (imagePointsProjSqueeze[k][0], imagePointsProjSqueeze[k][1]),
                    20,
                    (255, 0, 0),
                    1)
        
        cv2.imshow("Task4", img)
        cv2.waitKey(10)

        # Compute reprojection error
        error += np.sum(np.abs(imagePoints - imagePointsProjection))
    
    # Compute final error
    error /= (NUM_IMAGES * board_w * board_h)
    print("Projection error: ", error)


def task4(CM, D):
    # implement your solution
    pass


def task5(CM, rvecs, tvecs):
    # implement your solution
    pass

def main():
    # Showing images
    # for img_file in images_files_list:
    #     print(img_file)
    #     img = cv2.imread(img_file)
    #     images.append(img)
    #     cv2.imshow("Task1", img)
    #     cv2.waitKey(10)

    imagePoints, objectPoints = task1()  # Calling Task 1

    CM, D, rvecs, tvecs = task2(imagePoints, objectPoints)  # Calling Task 2

    task3(imagePoints, objectPoints, CM, D, rvecs, tvecs)  # Calling Task 3

    task4(CM, D)  # Calling Task 4

    task5(CM, rvecs, tvecs)  # Calling Task 5

    print("FINISH!")


if __name__ == "__main__":
    main()
