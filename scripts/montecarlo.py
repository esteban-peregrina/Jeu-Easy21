import numpy as np
import random

from environment import init_game, step

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class Easy21Agent:
    def __init__(self, N0 = 100):
        # Size : 10 (dealer initial is just one draw) x 21 (player sum) x 2 (possible actions)
        self.Q = np.zeros((10, 21, 2))  # Quality matrix, store action score for each state
        self.N = np.zeros((10, 21, 2)) # Count how much time an action was done from a state

        self.N0 = N0
        
        self.actions = ['hit', 'stick']

    def get_action(self, state):
        """
        Compute agent action using epsilon-greedy strategy
        
        :param state: state giving player sum and dealer first card
        """
        dealer_card_index = state['dealer'] - 1 # Initial dealer card, drawn at game's start
        player_sum_index = state['player'] - 1

        # Choose to explore or exploit
        N_s = np.sum(self.N[dealer_card_index, player_sum_index, :]) # Compute how much time this exact state was encountered, no matter the action that followed
        epsilon = self.N0 / (self.N0 + N_s) # The more we explore, the more unlikely it is to keep exploring
        if (random.random() < epsilon): 
            return random.choice(self.actions) # Explore
        else:
            index = np.argmax(self.Q[dealer_card_index, player_sum_index, :]) # Look for the best action to do in this state (from experience)
            return self.actions[index] # Exploit best action 

    def learn(self, episodes = 500000):
        """
        Implement Monte Carlo epsilon-greedy control
        
        :param episodes: Number of game the agent will play to train itself
        """
        for _ in range(episodes):
            history = [] # Each episode is a full game made of a sequence of states and actions
            state = init_game()
            terminal = False
            # Play the game using the learnt quality matrix
            while not terminal:
                action = self.get_action(state)
                history.append((state.copy(), action))
                state, reward, terminal = step(state, action)
            # Rewind the episode, and tweak the quality matrix to be more representative of the quality of the actions in each given state
            for state, action in history:
                dealer_card_index = state['dealer'] - 1 # We could also search for the corresponding index of this dealer_card but it way less efficient
                player_sum_index = state['player'] - 1
                action_index = 0 if action == 'hit' else 1
                self.N[dealer_card_index, player_sum_index, action_index] += 1 # Mark this scenario as being visited one more time
                
                # Update quality matrix
                alpha = 1.0 / self.N[dealer_card_index, player_sum_index, action_index] 
                error = reward - self.Q[dealer_card_index, player_sum_index, action_index] # Was the reward actually better than excpected, or worse ?
                self.Q[dealer_card_index, player_sum_index, action_index] += alpha * error # Correct the action's score for this state, with less and less weight as experience increase

def plot_value_function(agent, title="Optimal Value Function"):
    """
    Plot value function as a 3D surface, where z-axis indicates how good (positive) or bad (negative) a given state (x,y) is for the agent.
    
    :param agent: Easy21Agent to take data from
    :param title: Title for the plot window
    """
    V = np.max(agent.Q, axis=2) # Look for the best actions to do for each state

    # Creating grid
    dealer_card_indexs = np.arange(1, 11) # From 1 to 10
    player_sum_indexs = np.arange(1, 22)   # From 1 to 21
    X, Y = np.meshgrid(dealer_card_indexs, player_sum_indexs)

    # Display plot
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot surface
    surf = ax.plot_surface(X, Y, V.T, cmap='viridis', edgecolor='none')

    # Set legends
    ax.set_xlabel('Dealer first card')
    ax.set_ylabel('Player sum')
    ax.set_zlabel('Optimal value V*(s)')
    ax.set_title(title)
    fig.colorbar(surf, shrink=0.5, aspect=5)

    plt.show()

def plot_optimal_policy(agent, title="Optimal Policy"):
    """
    Plot optimal policy as a 2D heatmap, where color indicate best action to do for a given stat (x,y) 
    
    :param agent: Easy21Agent to take data from
    :param title: Title for the plot window
    """
    # Get best action index
    policy = np.argmax(agent.Q, axis=2)

    # Optimal policy heatmap
    plt.imshow(policy.T, origin='lower', extent=[1, 10, 1, 21], aspect='auto', cmap='coolwarm')
    plt.colorbar(label='0: Hit (Blue) | 1: Stick (Red)')
    plt.xlabel('Dealer first card')
    plt.ylabel('Player sum')
    plt.title('Optimal policy')
    plt.show()

if __name__ == '__main__':
    agent = Easy21Agent()
    agent.learn() 
    plot_value_function(agent)
    plot_optimal_policy(agent)

    