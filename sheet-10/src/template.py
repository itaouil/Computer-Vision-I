import cv2
import numpy as np
import os
import os.path as osp
import matplotlib.pyplot as plt

NUM_IMAGES=14
NUM_Boards = NUM_IMAGES
image_prefix = "../images/"
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
    obj.append(np.array([[ptIdx/board_w, ptIdx%board_w, 0.0]],np.float32))
obj = np.vstack(obj)

def task1():
    #implement your solution
    pass

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
    for img_file in images_files_list:
        print(img_file)
        img = cv2.imread(img_file)
        cv2.imshow("Task1", img)
        cv2.waitKey(10)
    
    imagePoints, objectPoints = task1() #Calling Task 1
    
    CM, D, rvecs, tvecs = task2(imagePoints, objectPoints) #Calling Task 2

    task3(imagePoints, objectPoints, CM, D, rvecs, tvecs)  # Calling Task 3

    task4(CM, D) # Calling Task 4

    task5(CM, rvecs, tvecs) # Calling Task 5
    
    print("FINISH!")

main()