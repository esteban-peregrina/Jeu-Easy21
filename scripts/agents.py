"""
Created by Esteban Peregrina on 2026-02-17.

This script implements Easy21 agents, 
including a tabular agent that learns a quality matrix,
and a linear agent that learns a parameter vector for a linear model of the quality function.
"""

import numpy as np
import random

from environment import init_game, step
from utils import calculate_MSE, extract_Q_matrix_from_linear_agent

class Easy21Agent:
    def __init__(self, N0=100):
        self.actions = ['hit', 'stick']
        self.N0 = N0

        self.MSE = [] # Store MSE at each episode for learning curve plotting 
    def get_epsilon(self, state_count):
        return self.N0 / (self.N0 + state_count) # state_count = N(s)
    
    def select_action(self, state, epsilon):
        if random.random() < epsilon:
            return random.choice(self.actions)
        return self.get_best_action(state) # Greedy logic is agent-specific

    def get_best_action(self, state):
        raise NotImplementedError # Method must be implemented in inherited classes

class TabularAgent(Easy21Agent):
    def __init__(self, N0=100):
        super().__init__(N0)
        # dealer: 1-10 (idx 0-9), player: 1-21 (idx 0-20), actions: 2 (idx 0-1)
        self.Q = np.zeros((10, 21, 2)) # Quality matrix, store action score for each state (dealer initial, player sum) pair
        self.N = np.zeros((10, 21, 2)) # Count how much time each action was done from a state
    def _get_indices(self, state, action=None): # For internal use only
        """
        Get state and action indexes
        """
        d_idx = state['dealer'] - 1 # Initial dealer card index, drawn at game's start
        p_idx = state['player'] - 1 # Current player sum index
        if action is None:
            return d_idx, p_idx
        a_idx = 0 if action == 'hit' else 1 
        return d_idx, p_idx, a_idx

    def get_best_action(self, state):
        d_idx, p_idx = self._get_indices(state)
        # We look for the action to do in this state with the best score (from experience) and keep its index (argmax)
        best_action_idx = np.argmax(self.Q[d_idx, p_idx, :])
        return self.actions[best_action_idx] # We retrieve the corresponding action with its index
    
    def mc_learn(self, episodes=500000):
        """
        Implement Monte Carlo control with epsilon-greedy exploration
        
        :param episodes: Number of game the agent will play to train itself
        """
        # Clear Q and N
        self.Q = np.zeros((10, 21, 2)) 
        self.N = np.zeros((10, 21, 2)) 
        for _ in range(episodes):
            state = init_game()
            history = [] # Each episode is a full game made of a sequence of states and actions
            terminal = False

            # Play the game using the quality matrix
            while not terminal:
                d_idx, p_idx = self._get_indices(state)
                
                # Get action
                N_s = np.sum(self.N[d_idx, p_idx, :]) # N_s = N(s), compute how much time this exact state was encountered, no matter the action that followed
                epsilon = self.get_epsilon(N_s)
                action = self.select_action(state, epsilon)

                history.append((state.copy(), action))
                state, reward, terminal = step(state, action)
            # Rewind the episode, and tweak the quality matrix to be more representative of the quality of the actions in each given state
            for state, action in history:
                d_idx, p_idx, a_idx = self._get_indices(state, action)
                self.N[d_idx, p_idx, a_idx] += 1 # Mark this scenario as being visited one more time
                
                # Update quality matrix
                alpha = 1.0 / self.N[d_idx, p_idx, a_idx] # Learning rate, weight less and less as experience increase
                # Gain G = reward R because there is no discounting (gamma = 1)
                error = reward - self.Q[d_idx, p_idx, a_idx] # Was the LAST reward actually better than excpected, or worse ?
                self.Q[d_idx, p_idx, a_idx] += alpha * error # Correct the action's score for this state

    def sarsa_learn(self, lmbda, episodes=1000, Q_star=None):
        """
        Implement SARSA(lambda) algorithm with epsilon-greedy exploration
        
        :param episodes: Number of game the agent will play to train itself
        """
        # Clear Q and N
        self.Q = np.zeros((10, 21, 2)) 
        self.N = np.zeros((10, 21, 2))
        
        self.MSE = [] # Clear MSE history
        for _ in range(episodes):
            state = init_game() # Get the t=0 state
            d_idx, p_idx = self._get_indices(state)

            # Get t=0 action
            N_s = np.sum(self.N[d_idx, p_idx, :]) #  N_s = N(s), compute how much time this exact state was encountered, no matter the action that followed
            epsilon = self.get_epsilon(N_s)
            action = self.select_action(state, epsilon) 

            # Eligibility traces matrix E(s,a), same size (10, 21, 2) as quality matrix
            # Stores how much each state-action pair is relevant (eligible) for the current state
            E = np.zeros_like(self.Q)

            terminal = False
            
            # Play the game  using the learnt quality matrix and update it at each step
            while not terminal:
                # Current instant t
                d_idx, p_idx, a_idx = self._get_indices(state, action)
                self.N[d_idx, p_idx, a_idx] += 1 # Mark this scenario as being visited one more time
                E[d_idx, p_idx, a_idx] += 1 # Accumulating trace (could be a different type of operation)
                
                # Next instant t+1
                next_state, reward, terminal = step(state, action) # Executing the t action lead to the t+1 reward and the t+1 state

                if terminal:
                    delta = reward - self.Q[d_idx, p_idx, a_idx]
                else:
                    nd_idx, np_idx = self._get_indices(next_state)
                    next_N_s = np.sum(self.N[nd_idx, np_idx, :])
                    next_epsilon = self.get_epsilon(next_N_s)
                    next_action = self.select_action(next_state, next_epsilon) 
                    na_idx = 0 if next_action == 'hit' else 1

                    # t time-difference error with no discounting (gamma = 1) delta = R + gamma * Q(s', a') - Q(s, a)
                    delta = reward + self.Q[nd_idx, np_idx, na_idx] - self.Q[d_idx, p_idx, a_idx]  # Using the next action score for the update (SARSA), not the best action score (Q-learning)

                
                # Update quality matrix on the go
                alpha = 1.0 / self.N[d_idx, p_idx, a_idx] # Learning rate, weight less and less as experience increase
                self.Q += alpha * delta * E # Correct the action's score for all state, weight less and less as we go futher from the current state (accumulating-type trace E and so lambda)
                
                # On-policy : behavior policy and target policy are the same (both epsilon-greedy) 
                # so we use the next action for both acting AND evaluating actions score (and so updating the quality matrix Q, in other word learning)

                # Reduce trace eligibility for all state-action pair
                E *= lmbda # Again, no discounting

                # Iterating : t becomes t+1
                if not terminal:
                    state, action = next_state, next_action
            if Q_star is not None:
                self.MSE.append(calculate_MSE(self.Q, Q_star)) # Store MSE at the end of each episode for learning curve plotting

    def q_learning_learn(self, episodes=10000, Q_star=None):
        """
        Implement Q-learning algorithm with epsilon-greedy exploration
        
        :param episodes: Number of game the agent will play to train itself
        """
        # Clear Q and N
        self.Q = np.zeros((10, 21, 2)) 
        self.N = np.zeros((10, 21, 2)) 

        self.MSE = [] # Clear MSE history
        for _ in range(episodes):
            state = init_game() # Get the t=0 state
            terminal = False
            
            # Play the game  using the learnt quality matrix and update it at each step
            while not terminal:
                d_idx, p_idx = self._get_indices(state)
                N_s = np.sum(self.N[d_idx, p_idx, :]) # N_s = N(s), compute how much time this exact state was encountered, no matter the action that followed
                epsilon = self.get_epsilon(N_s)
                action = self.select_action(state, epsilon) # Behavior policy is epsilon-greedy
                a_idx = 0 if action == 'hit' else 1 
                self.N[d_idx, p_idx, a_idx] += 1 # Mark this scenario as being visited one more time

                next_state, reward, terminal = step(state, action) # Executing the t action lead to the t+1 reward and the t+1 state
                
                # Target calculation 
                # We use a purely greedy TARGET policy (taking the max Q-value of the next state) 
                # to evaluate the action score (and so update the quality matrix Q, in other word learn), 
                # even though our BEHAVIOR policy (how we actually act in the next step)
                # will still be epsilon-greedy and might randomly explore.
                target = reward + (0 if terminal else np.max(self.Q[self._get_indices(next_state)])) # Target policy is greedy
                
                # Off-policy : separate behavior policy (how it acts) and target policy (how it evaluate actions)

                alpha = 1.0 / self.N[d_idx, p_idx, a_idx] # Learning rate, weight less and less as experience increase
                self.Q[d_idx, p_idx, a_idx] += alpha * (target - self.Q[d_idx, p_idx, a_idx]) # Correct the action's score for this state

                state = next_state
            
            if Q_star is not None:
                self.MSE.append(calculate_MSE(self.Q, Q_star)) # Store MSE at the end of each episode for learning curve plotting


class LinearAgent(Easy21Agent):
    def __init__(self, N0=100):
        super().__init__(N0)
        self.theta = np.zeros(36)  # Parameter vector theta of the linear model (36 features)
        self.alpha = 0.01
        self.epsilon_linear = 0.05 
    def feature_vector(self, state, action):
        """
        Compute feature vector for a given state-action pair, using coarse coding (binary feature encoding of groups of states and actions)
        
        :param state: state giving player sum and dealer first card
        :param action: string for the player's action
        """
        dealer_groups = [(1, 4), (4, 7), (7, 10)] # Groups of dealer first card
        player_groups = [(1, 6), (4, 9), (7, 12), (10, 15), (13, 18), (16, 21)] # Groups of player sum
        action_index = 0 if action == 'hit' else 1

        # Binary feature encoding (coarse coding) using overlapping groups of states and actions
        phi = np.zeros((3, 6, 2)) # features
        for i, (d_min, d_max) in enumerate(dealer_groups):
            for j, (p_min, p_max) in enumerate(player_groups):
                for k in range(2):
                    if d_min <= state['dealer'] <= d_max and p_min <= state['player'] <= p_max and k == action_index:
                        phi[i, j, k] = 1
        
        return phi.flatten() # Flatten the 3D feature matrix into a 1D vector
    
    def get_Q(self, state, action):
        """
        Compute Q(s, a) = phi(s, a)^T * theta()
        """
        phi = self.feature_vector(state, action)
        return np.dot(phi, self.theta)
    
    def get_best_action(self, state):
        """
        Exploit.
        """
        Q_values = [self.get_Q(state, a) for a in self.actions] # Compute action score for each action, e.g. [-0.8, 0.5]
        return self.actions[np.argmax(Q_values)] # We retrieve the corresponding action with its index
    
    def sarsa_learn_linear(self, lmbda, episodes=1000, Q_star=None):
        """
        Implement SARSA(lambda) algorithm with epsilon-greedy exploration
        """
        self.theta = np.zeros(36) # Reset parameter vector
        self.MSE = [] # Store MSE for each episode for learning curve plotting
        for _ in range(episodes):
            state = init_game() # Get the t=0 state

            # Get t=0 action
            action = self.select_action(state, self.epsilon_linear)

            # Eligibility traces vector e(s,a), same size (36) as parameter vector
            # Stores how much each state-action pair is relevant (eligible) for the current state
            e = np.zeros_like(self.theta)

            terminal = False
            
            # Play the game  using the learnt quality matrix and update it at each step
            while not terminal:
                # Current instant t
                phi = self.feature_vector(state, action)
                Q_current = np.dot(phi, self.theta)

                # Next instant t+1
                next_state, reward, terminal = step(state, action) # Executing the t action lead to the t+1 reward and the t+1 state

                if terminal:
                    delta = reward - Q_current
                else:
                    next_action = self.select_action(next_state, self.epsilon_linear) # Constant epsilon
                    Q_next = self.get_Q(next_state, next_action)

                    # t time-difference error with no discounting (gamma = 1) delta = R + gamma * Q(s', a') - Q(s, a)
                    delta = reward + Q_next - Q_current

                
                # Decay traces for past states and accumulate the trace for the current state-action pair
                e = lmbda * e + phi # Again, no discounting

                # Updating theta
                self.theta += self.alpha * delta * e

                # Iterating : t becomes t+1
                if not terminal:
                    state, action = next_state, next_action
            
            if Q_star is not None:
                    Q_current_matrix = extract_Q_matrix_from_linear_agent(self)
                    self.MSE.append(calculate_MSE(Q_current_matrix, Q_star))
