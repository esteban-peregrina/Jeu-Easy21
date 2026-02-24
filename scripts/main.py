"""
Created by Esteban Peregrina on 2026-02-24.

This script is the main entry point of the project. It creates agents, makes them learn, and plot their value function and optimal policy.
"""

import numpy as np
import matplotlib.pyplot as plt

from agents import TabularAgent, LinearAgent
from plot import plot_value_function, plot_optimal_policy
from utils import calculate_MSE


if __name__ == '__main__':
    MC_agent = TabularAgent()
    MC_agent.mc_learn(1000000)
    plot_value_function(MC_agent, file_name="value_function_MC.png", title="Optimal Value Function (Monte Carlo)")
    plot_optimal_policy(MC_agent, file_name="optimal_policy_MC.png", title="Optimal Policy (Monte Carlo)")
    
    Qstar = MC_agent.Q.copy() # We consider the quality matrix learnt by Monte Carlo as the optimal quality matrix, since it is learnt after a lot of episodes

    SARSA_agent = TabularAgent()
    SARSA_agent.sarsa_learn(0.3)
    plot_value_function(SARSA_agent, file_name="value_function_SARSA.png", title="Optimal Value Function (SARSA)")
    plot_optimal_policy(SARSA_agent, file_name="optimal_policy_SARSA.png", title="Optimal Policy (SARSA)")

    # Compute MSE between learnt quality matrix and optimal quality matrix
    print("Computing MSE...")
    lambdas = np.linspace(0, 1, 11) # [0.0, 0.1, ..., 1.0]
    MSE = []
    for l in lambdas:
        SARSA_agent.sarsa_learn(lmbda=l);
        MSE.append(calculate_MSE(SARSA_agent.Q, Qstar))

    #  Plot MSE with respect to Lambda
    plt.figure()
    plt.plot(lambdas, MSE, marker='o')
    plt.xlabel(r'$\lambda$')
    plt.ylabel('MSE')
    plt.title(r'MSE in function of $\lambda$ after 1000 episodes')
    plt.grid(True)
    plt.savefig('./../records/MSE.png')
    plt.close()
    print("MSE saved in records/")
    
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