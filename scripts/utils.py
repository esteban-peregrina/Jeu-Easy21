import numpy as np

def calculate_MSE(Q, Q_star):
    return np.sum((Q - Q_star)**2) / np.prod(Q.shape) # shape return the size of each dimension of the matrix, so we get the total number of element in the matrix by multiplying them together (prod)
