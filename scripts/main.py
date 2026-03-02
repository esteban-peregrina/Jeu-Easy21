"""
Created by Esteban Peregrina on 2026-02-24.

This script is the main entry point of the project. It creates agents, makes them learn, and plot their value function and optimal policy.
"""

import numpy as np
import matplotlib.pyplot as plt

from agents import TabularAgent, LinearAgent
from plot import plot_value_function, plot_optimal_policy
from utils import calculate_MSE, extract_Q_matrix_from_linear_agent


if __name__ == '__main__':
    ########## TABULAR AGENTS ##########
    MC_agent = TabularAgent()
    MC_agent.mc_learn(1000000)
    plot_value_function(MC_agent, file_name="value_function_MC.png", title="Optimal Value Function (Monte Carlo)")
    plot_optimal_policy(MC_agent, file_name="optimal_policy_MC.png", title="Optimal Policy (Monte Carlo)")
    
    Qstar = MC_agent.Q.copy() # We consider the quality matrix learnt by Monte Carlo as the optimal quality matrix, since it is learnt after a lot of episodes

    SARSA_agent = TabularAgent()
    SARSA_agent.sarsa_learn(0.3)
    plot_value_function(SARSA_agent, file_name="value_function_SARSA.png", title="Optimal Value Function (SARSA)")
    plot_optimal_policy(SARSA_agent, file_name="optimal_policy_SARSA.png", title="Optimal Policy (SARSA)")

    # --- SARSA evaluation ---
    # Compute MSE between SARSA learnt quality matrix and optimal quality matrix
    print("Computing MSE...")
    lambdas = np.linspace(0, 1, 11) # [0.0, 0.1, ..., 1.0]
    MSE_SARSA = []
    for l in lambdas:
        print(f"Training Tabular SARSA with lambda = {l:.1f}...")
        SARSA_agent.sarsa_learn(lmbda=l);
        MSE_SARSA.append(calculate_MSE(SARSA_agent.Q, Qstar))

    #  Plot MSE with respect to Lambda
    plt.figure()
    plt.plot(lambdas, MSE_SARSA, marker='o')
    plt.xlabel(r'$\lambda$')
    plt.ylabel('MSE')
    plt.title(r'MSE in function of $\lambda$ after 1000 episodes')
    plt.grid(True)
    plt.savefig('./../records/MSE_SARSA.png')
    plt.close()
    print("Monte Carlo-SARSA MSE saved in records/")
    
    # Plot learning curves for lambda = 0 and lambda = 1 (with MSE on y-axis and episodes on x-axis)
    print("Running Learning Curves...")
    SARSA_agent.sarsa_learn(0, Q_star=Qstar) # Giving Q_star to the learning function to store MSE at each episode
    lc_0 = SARSA_agent.MSE
    SARSA_agent.sarsa_learn(1, Q_star=Qstar)
    lc_1 = SARSA_agent.MSE

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

    # --- Q-Learning Evaluation ---
    Qlearning_agent = TabularAgent()
    
    # Plot learning curves for lambda = 0 and lambda = 1 (with MSE on y-axis and episodes on x-axis)
    print("Running Learning Curves...")
    SARSA_agent.sarsa_learn(0, 10000, Q_star=Qstar)
    lc_0 = SARSA_agent.MSE
    SARSA_agent.sarsa_learn(1, 10000, Q_star=Qstar)
    lc_1 = SARSA_agent.MSE
    Qlearning_agent.q_learning_learn(Q_star=Qstar)
    lc_2 = Qlearning_agent.MSE

    plt.figure()
    plt.plot(lc_0, label='SARSA - Lambda = 0')
    plt.plot(lc_1, label='SARSA - Lambda = 1')
    plt.plot(lc_2, label='Q-Learning')
    plt.xlabel('Episodes')
    plt.ylabel('MSE')
    plt.title('Learning Curves (MSE vs Episodes)')
    plt.grid(True)
    plt.legend()
    plt.savefig('./../records/learning_curves_SARSAvsQ-Learning.png')
    plt.close()
    print("Learning Curves saved in records/")

    # Discussion
    # On-policy uses the same policy (π) for interaction and learning. 
    # Off-policy uses a behavior policy (μ) for interaction 
    # to learn about a different target policy (π), requiring a correction step 
    # (here for the Q-Learning its in some way the use of max()).

    ########## LINEAR AGENTS ##########
    SARSA_linear_agent = LinearAgent()

    # --- Linear SARSA evaluation ---
    # Compute MSE between linear SARSA learnt quality matrix and optimal quality matrix
    print("Computing MSE...")
    MSE_linear = []
    for l in lambdas:
        print(f"Training Linear SARSA with lambda = {l:.1f}...")
        SARSA_linear_agent.sarsa_learn_linear(lmbda=l)

        Q_linear = extract_Q_matrix_from_linear_agent(SARSA_linear_agent)

        MSE_linear.append(calculate_MSE(Q_linear, Qstar))

    # Plot MSE with respect to Lambda
    plt.figure()
    plt.plot(lambdas, MSE_linear, marker='o')
    plt.xlabel(r'$\lambda$')
    plt.ylabel('MSE')
    plt.title(r'MSE in function of $\lambda$ after 1000 episodes')
    plt.grid(True)
    plt.savefig('./../records/MSE_SARSA_linear.png')
    plt.close()
    print("Monte Carlo-Linear SARSA saved in records/")

    # Plot learning curves for lambda = 0 and lambda = 1 (with MSE on y-axis and episodes on x-axis)
    print("Running Learning Curves...")
    SARSA_linear_agent.sarsa_learn_linear(0, Q_star=Qstar) # Giving Q_star to the learning function to store MSE at each episode
    lc_0 = SARSA_linear_agent.MSE
    SARSA_linear_agent.sarsa_learn_linear(1, Q_star=Qstar)
    lc_1 = SARSA_linear_agent.MSE

    plt.figure()
    plt.plot(lc_0, label='Lambda = 0')
    plt.plot(lc_1, label='Lambda = 1')
    plt.xlabel('Episodes')
    plt.ylabel('MSE')
    plt.title('Learning Curves (MSE vs Episodes)')
    plt.grid(True)
    plt.legend()
    plt.savefig('./../records/learning_curves_linear.png')
    plt.close()
    print("Learning Curves saved in records/")
