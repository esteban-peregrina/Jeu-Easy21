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

        self.Qstar = np.zeros((10, 21, 2)) # Optimal quality matrix, store optimal action score for each state

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

    def mc_learn(self, episodes = 500000):
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
                error = reward - self.Q[dealer_card_index, player_sum_index, action_index] # Was the LAST reward actually better than excpected, or worse ?
                self.Q[dealer_card_index, player_sum_index, action_index] += alpha * error # Correct the action's score for this state, with less and less weight as experience increase

        self.Qstar = np.copy(self.Q) # After learning, we can consider that the learnt quality matrix is the optimal one

    def sarsa_learn(self, lambd4, episodes = 1000):
        """
        Implement SARSA epsilon-greey controle
        
        :param episodes: Number of game the agent will play to train itself
        """
        for _ in range(episodes):
            current_state = init_game() # Get the t=0 state
            terminal = False
            current_action = self.get_action(current_state) # Predict the t=0 action

            # Eligibility traces matrix, same size (10, 21, 2) as quality matrix
            # Stores how much each state-action pair is relevant (eligible) for the current state
            E = np.zeros_like(self.Q) 
            # Play the game  using the learnt quality matrix and update it at each step
            while not terminal:
                # Current instant t
                d_idx = current_state['dealer'] - 1 # Dealer first card index, drawn at game's start
                p_idx = current_state['player'] - 1 # Player sum index
                a_idx = 0 if current_action == 'hit' else 1 # Action index

                self.N[d_idx, p_idx, a_idx] += 1 # Mark this scenario as being visited one more time
                E[d_idx, p_idx, a_idx] += 1 # Mark this scenario as being visited one more time
                
                # Next instant t+1
                next_state, next_reward, terminal = step(current_state, current_action) # Executing the t action lead to the t+1 reward and the t+1 state

                # Compute t time difference error (between t+1 and t !)
                if not terminal:
                    next_action = self.get_action(next_state) # Predict the t+1 action
                    nd_idx = next_state['dealer'] - 1
                    np_idx = next_state['player'] - 1
                    na_idx = 0 if next_action == 'hit' else 1
                    
                    delta = next_reward + self.Q[nd_idx, np_idx, na_idx] - self.Q[d_idx, p_idx, a_idx] # t time-difference error
                else:
                    delta = next_reward - self.Q[d_idx, p_idx, a_idx] 
                    next_action = None # End of the game
                
                # Update quality matrix on the go
                alpha = 1.0 / self.N[d_idx, p_idx, a_idx]
                self.Q += alpha * delta * E # Correct the action's score for all state, with less and less weight as experience increase (alpha), and with less and less weights as we go futher from the current state (E and so lambda)
                
                # Reduce trace eligibility for all state-action pair
                E *= lambd4

                # Iterating : t becomes t+1
                if not terminal:
                    current_state = next_state
                    current_action = next_action

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
    agent.sarsa_learn(0.1) 
    plot_value_function(agent)
    plot_optimal_policy(agent)

    