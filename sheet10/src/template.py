import cv2
import numpy as np
import os
import os.path as osp
import matplotlib.pyplot as plt

NUM_IMAGES=14
NUM_Boards = NUM_IMAGES
# image_prefix = "../images/"
image_prefix = "/Users/dailand10/Desktop/Computer-Vision-I/sheet10/images/"
image_suffix = ".png"
images_files_list = [osp.join(image_prefix, f) for f in os.listdir(image_prefix)
                     if osp.isfile(osp.join(image_prefix, f)) and f.endswith(image_suffix)]
board_w = 10
board_h = 7
board_size = (board_w, board_h)
board_n = board_w * board_h
img_shape = (0,0)
obj = []
for ptIdx in range(0, board_n):
    obj.append(np.array([[ptIdx/board_w, ptIdx%board_w, 0.0]], np.float32))
obj = np.vstack(obj)

def task1():
    """
        Detect and display
        corners for each image
    """
    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d point in real world space
    imgpoints = [] # 2d points in image plane.

    # termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # Iterate over set of images
    for img_file in images_files_list:
        # Read image as grayscale
        img = cv2.imread(img_file)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Find the chessboard corners
        found, corners = cv2.findChessboardCorners(gray, (board_h, board_w), None)

        # Check if corners found
        # and display them
        if found:
            # Append 3D object points
            objpoints.append(obj)

            # Refine detected corner
            corners2 = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
            imgpoints.append(corners2)

            # Draw and display the corners
            # img = cv2.drawChessboardCorners(img, (board_h, board_w), corners2, found)
            # cv2.imshow('img', img)
            # cv2.waitKey(500)
    
    return imgpoints, objpoints

def task2(imagePoints, objectPoints):
    #implement your solution
    pass

def task3(imagePoints, objectPoints, CM, D, rvecs, tvecs):
    #implement your solution
    pass

def task4(CM, D):
    #implement your solution
    pass

def task5(CM, rvecs, tvecs):
    #implement your solution
    pass

def main():
    #Showing images
    # for img_file in images_files_list:
    #     print(img_file)
    #     img = cv2.imread(img_file)
    #     cv2.imshow("Task1", img)
    #     cv2.waitKey(10)
    
    imagePoints, objectPoints = task1() #Calling Task 1
    
    CM, D, rvecs, tvecs = task2(imagePoints, objectPoints) #Calling Task 2

    task3(imagePoints, objectPoints, CM, D, rvecs, tvecs)  # Calling Task 3

    task4(CM, D) # Calling Task 4

    task5(CM, rvecs, tvecs) # Calling Task 5
    
    print("FINISH!")

if __name__ == "__main__":
    main()