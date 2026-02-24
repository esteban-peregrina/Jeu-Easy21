import numpy as np

def feature_vector(state, action):
    """
    Compute feature vector for a given state-action pair, using coarse coding (one-hot encoding of groups of states and actions)
    
    :param state: state giving player sum and dealer first card
    :param action: string for the player's action
    """
    dealer_groups = [(1, 4), (4, 7), (7, 10)] # Groups of dealer first card
    player_groups = [(1, 6), (4, 9), (7, 12), (10, 15), (13, 18), (16, 21)] # Groups of player sum
    action_index = 0 if action == 'hit' else 1

    # One-hot encoding of groups of states and actions (3 groups for dealer card, 6 groups for player sum, 2 groups for action)
    features = np.zeros((3, 6, 2))
    for i, (d_min, d_max) in enumerate(dealer_groups):
        for j, (p_min, p_max) in enumerate(player_groups):
            for k in range(2):
                if d_min <= state['dealer'] <= d_max and p_min <= state['player'] <= p_max and k == action_index:
                    features[i, j, k] = 1
    
    return features.flatten() # Flatten the 3D feature matrix into a 1D vector

