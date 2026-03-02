import numpy as np

def calculate_MSE(Q, Q_star):
    return np.sum((Q - Q_star)**2) / np.prod(Q.shape) # shape return the size of each dimension of the matrix, so we get the total number of element in the matrix by multiplying them together (prod)

def extract_Q_matrix_from_linear_agent(agent):
    """
    Reconstructs a 10x21x2 Q-matrix from a LinearAgent by evaluating 
    all possible state-action pairs.
    """
    Q_matrix = np.zeros((10, 21, 2))
    
    # Loop through all possible states and actions
    for d in range(1, 11):      # Dealer: 1 to 10
        for p in range(1, 22):  # Player: 1 to 21
            state = {'dealer': d, 'player': p}
            
            # Action 'hit' is index 0
            Q_matrix[d-1, p-1, 0] = agent.get_Q(state, 'hit')
            
            # Action 'stick' is index 1
            Q_matrix[d-1, p-1, 1] = agent.get_Q(state, 'stick')
            
    return Q_matrix