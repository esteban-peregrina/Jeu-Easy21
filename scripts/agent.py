import numpy as np
import random

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from environment import init_game, step
from plot import plot_value_function, plot_optimal_policy

GAMMA = 1.0 # No discounting, as the game is episodic and we want to maximize the final reward of the game, not the intermediate rewards

class Easy21Agent:
    def __init__(self, N0 = 100):
        # Size : 10 (dealer initial is just one draw) x 21 (player sum) x 2 (possible actions)
        self.Q = np.zeros((10, 21, 2))  # Quality matrix, store action score for each state
        self.N = np.zeros((10, 21, 2)) # Count how much time an action was done from a state

        self.N0 = N0
        
        self.actions = ['hit', 'stick']

        self.Qstar = np.zeros((10, 21, 2)) # Optimal quality matrix, store optimal action score for each state
        self.MSE = [] # Store MSE between learnt quality matrix and optimal quality matrix at each episodes

    def get_action(self, state, epsilon=None):
        """
        Compute agent action using epsilon-greedy strategy
        
        :param state: state giving player sum and dealer first card
        """
        dealer_card_index = state['dealer'] - 1 # Initial dealer card, drawn at game's start
        player_sum_index = state['player'] - 1

        # Choose to explore or exploit
        N_s = np.sum(self.N[dealer_card_index, player_sum_index, :]) # Compute how much time this exact state was encountered, no matter the action that followed
        if epsilon is None:
            epsilon = self.N0 / (self.N0 + N_s) # The more we explore, the more unlikely it is to keep exploring
        if (random.random() < epsilon): 
            return random.choice(self.actions) # Explore
        else:
            index = np.argmax(self.Q[dealer_card_index, player_sum_index, :]) # Look for the best action to do in this state (from experience)
            return self.actions[index] # Exploit best action 

    def mc_learn(self, alpha=None, episodes = 500000):
        """
        Implement Monte Carlo epsilon-greedy control
        
        :param episodes: Number of game the agent will play to train itself
        """
        self.Q = np.zeros((10, 21, 2))  # Quality matrix, store action score for each state
        self.N = np.zeros((10, 21, 2)) # Count how much time an action was done from a state
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
                if alpha is None:
                    alpha = 1.0 / self.N[dealer_card_index, player_sum_index, action_index] # Learning rate, with less and less weight as experience increase
                error = reward - self.Q[dealer_card_index, player_sum_index, action_index] # Was the LAST reward actually better than excpected, or worse ?
                self.Q[dealer_card_index, player_sum_index, action_index] += alpha * error # Correct the action's score for this state

        self.Qstar = np.copy(self.Q) # After learning, we can consider that the learnt quality matrix is the optimal one

    def sarsa_learn(self, l4mbda, alpha=None, episodes = 1000):
        """
        Implement SARSA epsilon-greey control
        
        :param episodes: Number of game the agent will play to train itself
        """
        self.Q = np.zeros((10, 21, 2))  # Quality matrix, store action score for each state
        self.N = np.zeros((10, 21, 2)) # Count how much time an action was done from a state
        self.MSE = []
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
                    
                    delta = next_reward + GAMMA * self.Q[nd_idx, np_idx, na_idx] - self.Q[d_idx, p_idx, a_idx] # t time-difference error
                else:
                    delta = next_reward - self.Q[d_idx, p_idx, a_idx] 
                    next_action = None # End of the game
                
                # Update quality matrix on the go
                if alpha is None:
                    alpha = 1.0 / self.N[d_idx, p_idx, a_idx] # Learning rate, with less and less weight as experience increase
                self.Q += alpha * delta * E # Correct the action's score for all state, with less and less weights as we go futher from the current state (E and so lambda)
                
                # Reduce trace eligibility for all state-action pair
                E *= l4mbda

                # Iterating : t becomes t+1
                if not terminal:
                    current_state = next_state
                    current_action = next_action
            self.MSE.append(calculate_mse(self.Q, self.Qstar)) # After each episode, we can compute the MSE between the learnt quality matrix and the optimal quality matrix, to see if we are actually learning something

def calculate_mse(Q, Q_star):
    return np.sum((Q - Q_star)**2) / np.prod(Q.shape) # shape return the size of each dimension of the matrix, so we get the total number of element in the matrix by multiplying them together (prod)

if __name__ == '__main__':
    agent = Easy21Agent()
    agent.mc_learn(1000000)
    agent.sarsa_learn(0.1) 
    plot_value_function(agent)
    plot_optimal_policy(agent)

    # Compute MSE between learnt quality matrix and optimal quality matrix
    print("Computing MSE...")
    lambdas = np.linspace(0, 1, 11) # [0.0, 0.1, ..., 1.0]
    MSE = []
    for l in lambdas:
        agent.sarsa_learn(l4mbda=l);
        MSE.append(calculate_mse(agent.Q, agent.Qstar))

    #  MSE vs Lambda
    plt.figure()
    plt.plot(lambdas, MSE, marker='o')
    plt.xlabel(r'$\lambda$')
    plt.ylabel('MSE')
    plt.title(r'MSE in function of $\lambda$ after 1000 episodes')
    plt.grid(True)
    plt.savefig('./../records/MSE.png')
    plt.close()
    print("MSE saved in records/")
    # MSE is best for lambda = 0.4
    
    # MSE vs episodes for lambda = 0 and lambda = 1
    print("Running Learning Curves...")
    agent.sarsa_learn(l4mbda=0)
    lc_0 = agent.MSE
    agent.sarsa_learn(l4mbda=1)
    lc_1 = agent.MSE

    plt.figure()
    plt.plot(lc_0, label='Lambda = 0')
    plt.plot(lc_1, label='Lambda = 1')
    plt.xlabel('Episodes')
    plt.ylabel('MSE')
    plt.title('Learning Curves (MSE vs Episodes)')
    plt.grid(True)
    plt.legend()
    plt.savefig('./../records/learning_curves.png')
    plt.close()
    print("Learning Curves saved in records/")

    # Discussion : current method are tabular (mapping each state-action pair to a score), 
    # so we can not generalize to unseen states, 
    # and we need to visit each state-action pair a lot of time to have a good estimation of their score. 
    # We could use function approximation (for example with a neural network) to be able to generalize to unseen states, 
    # and to learn faster by sharing knowledge between similar states.