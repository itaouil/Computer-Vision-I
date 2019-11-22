import cv2
import numpy as np
import maxflow

def question_3(I, rho=0.7, prw_same=0.005, prw_diff=0.2):
    ### 1) Define Graph
    g = maxflow.Graph[float]()

    ### 2) Add pixels as nodes
    nodeids = g.add_grid_nodes(I.shape)

    ### 3) Compute Unary cost
    u_source = np.where(I == 0, -np.log(rho), -np.log(1-rho))
    u_sink = np.where(I == 0, -np.log(1-rho), -np.log(rho))

    ### 4) Add terminal edges
    # g.add_grid_tedges(nodeids, -np.log(rho), -np.log(1-rho))
    g.add_grid_tedges(nodeids, u_source, u_sink)

    ### 5) Add Node edges
    ### Vertical Edges
    structure = np.array([[0, 0, 0],
                          [0, 0, 0],
                          [0, 1, 0]])

    v_weights = np.zeros((I.shape[0], I.shape[1]))
    for col in range(I.shape[1]):
        for row in range(I.shape[0]-1):
            v_weights[row, col] = prw_same if I[row][col] == I[row+1][col] else prw_diff
    print("Vertical weights: ", v_weights)

    g.add_grid_edges(nodeids, weights=v_weights, structure=structure, symmetric=True)

    ### Horizontal edges
    structure = np.array([[0, 0, 0],
                          [0, 0, 1],
                          [0, 0, 0]])

    h_weights = np.zeros((I.shape[0], I.shape[1]))
    for row in range(I.shape[0]):
        for col in range(I.shape[1]-1):
            h_weights[row, col] = prw_same if I[row][col] == I[row][col+1] else prw_diff
    print("horizontal weights: ", h_weights)

    g.add_grid_edges(nodeids, weights=h_weights, structure=structure, symmetric=True)

    ### 6) Maxflow
    g.maxflow()

    # Denoise image
    sgm = g.get_grid_segments(nodeids)
    Denoised_I = np.int_(np.logical_not(sgm)).astype(np.uint8) * 255
    print("Segmented: ", Denoised_I)

    cv2.imshow('Original Img', I), \
    cv2.imshow('Denoised Img', Denoised_I), cv2.waitKey(0), cv2.destroyAllWindows()

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

def main():
    image_q3 = cv2.imread('./images/noise.png', cv2.IMREAD_GRAYSCALE)
    image_q4 = cv2.imread('./images/noise2.png', cv2.IMREAD_GRAYSCALE)

    ### Call solution for question 3
    question_3(image_q3, rho=0.7, prw_same=0.005, prw_diff=0.2)
    question_3(image_q3, rho=0.7, prw_same=0.005, prw_diff=0.35)
    question_3(image_q3, rho=0.7, prw_same=0.005, prw_diff=0.55)

    ### Call solution for question 4
    # question_4(image_q4, rho=0.8)

if __name__ == "__main__":
    main()
