import cv2
import numpy as np
import maxflow

def question_3(I,rho=0.7,pairwise_cost_same=0.005,pairwise_cost_diff=0.2):


    ### 1) Define Graph
    g = maxflow.Graph[float]()

    ### 2) Add pixels as nodes

    ### 3) Compute Unary cost

    ### 4) Add terminal edges

    ### 5) Add Node edges
    ### Vertical Edges

    ### Horizontal edges
    # (Keep in mind the stucture of neighbourhood and set the weights according to the pairwise potential)



    ### 6) Maxflow
    g.maxflow()

    cv2.imshow('Original Img', I), \
    cv2.imshow('Denoised Img', Denoised_I), cv2.waitKey(0), cv2.destroyAllWindows()
    return

def question_4(I,rho=0.6):

    labels = np.unique(I).tolist()

    Denoised_I = np.zeros_like(I)
    ### Use Alpha expansion binary image for each label

    ### 1) Define Graph

    ### 2) Add pixels as nodes

    ### 3) Compute Unary cost

    ### 4) Add terminal edges

    ### 5) Add Node edges
    ### Vertical Edges

    ### Horizontal edges
    # (Keep in mind the stucture of neighbourhood and set the weights according to the pairwise potential)

    ### 6) Maxflow


    cv2.imshow('Original Img', I), \
    cv2.imshow('Denoised Img', Denoised_I), cv2.waitKey(0), cv2.destroyAllWindows()

    return

def main():
    image_q3 = cv2.imread('./images/noise.png', cv2.IMREAD_GRAYSCALE)
    image_q4 = cv2.imread('./images/noise2.png', cv2.IMREAD_GRAYSCALE)

    ### Call solution for question 3
    question_3(image_q3, rho=0.7, pairwise_cost_same=0.005, pairwise_cost_diff=0.2)
    question_3(image_q3, rho=0.7, pairwise_cost_same=0.005, pairwise_cost_diff=0.35)
    question_3(image_q3, rho=0.7, pairwise_cost_same=0.005, pairwise_cost_diff=0.55)

    ### Call solution for question 4
    question_4(image_q4, rho=0.8)
    return

if __name__ == "__main__":
    main()



