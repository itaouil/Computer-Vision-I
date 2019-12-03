import cv2
import numpy as np
import maxflow
from maxflow.fastmin import aexpansion_grid


def question_3(I, rho=0.7, prw_same=0.005, prw_diff=0.2):
    ### 1) Define Graph
    g = maxflow.Graph[float]()

    ### 2) Add pixels as nodes
    nodeids = g.add_grid_nodes(I.shape)

    ### 3) Compute Unary cost
    u_source = np.where(I == 0, -np.log(rho), -np.log(1 - rho))
    u_sink = np.where(I == 0, -np.log(1 - rho), -np.log(rho))

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
        for row in range(I.shape[0] - 1):
            v_weights[row, col] = prw_same if I[row][col] == I[row + 1][col] else prw_diff

    g.add_grid_edges(nodeids, weights=v_weights, structure=structure, symmetric=True)

    ### Horizontal edges
    structure = np.array([[0, 0, 0],
                          [0, 0, 1],
                          [0, 0, 0]])

    h_weights = np.zeros((I.shape[0], I.shape[1]))
    for row in range(I.shape[0]):
        for col in range(I.shape[1] - 1):
            h_weights[row, col] = prw_same if I[row][col] == I[row][col + 1] else prw_diff

    g.add_grid_edges(nodeids, weights=h_weights, structure=structure, symmetric=True)

    ### 6) Maxflow
    g.maxflow()

    # Denoise image
    sgm = g.get_grid_segments(nodeids)
    Denoised_I = np.int_(np.logical_not(sgm)).astype(np.uint8) * 255

    cv2.imshow('Original Img', I), \
    cv2.imshow('Denoised Img', Denoised_I), cv2.waitKey(0), cv2.destroyAllWindows()


def question_4(I, rho=0.8):
    labels = np.unique(I)

    # 1) Compute Unary cost
    # define unary costs
    u_same = - np.log(rho)
    u_diff = - np.log((1 - rho) / 2)

    # assign unary costs
    D = np.zeros(I.shape + labels.shape)
    for y in range(I.shape[0]):
        for x in range(I.shape[1]):
            for lbl in range(labels.shape[0]):
                if I[y, x] == labels[lbl]:
                    D[y, x, lbl] = u_same
                else:
                    D[y, x, lbl] = u_diff

    # 2) Compute Pairwise cost metric
    # define pairwise cost
    def pairwise_cost(i, j):
        def kronecker_delta(i, j):
            return 1 if i == j else 0

        return 1 - kronecker_delta(i, j)

    # assign pairwise costs
    V = np.zeros((labels.shape[0], labels.shape[0]))
    for i in range(labels.shape[0]):
        for j in range(labels.shape[0]):
            V[i, j] = pairwise_cost(i, j)

    # 3) Alpha expansion
    alpha_exp = aexpansion_grid(D, V).astype(np.uint8)

    # 4) Assign values to labels
    for lbl in range(labels.shape[0]):
        alpha_exp[alpha_exp == lbl] = labels[lbl]
    Denoised_I = alpha_exp

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
    question_4(image_q4, rho=0.8)


if __name__ == "__main__":
    main()
