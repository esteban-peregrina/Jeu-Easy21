"""
Created by Esteban Peregrina on 2026-02-24.

This script implements functions to plot the value function and optimal policy of the Easy21 agent, using matplotlib library.
"""

import numpy as np

import matplotlib.pyplot as plt

def plot_value_function(agent, file_name, title="Optimal Value Function"):
    """
    Plot value function as a 3D surface, where z-axis indicates how good (positive) or bad (negative) a given state (x,y) is for the agent.
    
    :param agent: Easy21Agent to take data from
    :param file_name: Name of the file to save the plot
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

    plt.savefig('./../records/' + file_name)
    plt.show()
    print("Value function saved in records/")


def plot_optimal_policy(agent, file_name, title="Optimal Policy"):
    """
    Plot optimal policy as a 2D heatmap, where color indicate best action to do for a given stat (x,y) 
    
    :param agent: Easy21Agent to take data from
    :param file_name: Name of the file to save the plot
    :param title: Title for the plot window
    """
    # Get best action index
    policy = np.argmax(agent.Q, axis=2)

    # Optimal policy heatmap
    plt.figure()
    plt.imshow(policy.T, origin='lower', extent=[1, 10, 1, 21], aspect='auto', cmap='coolwarm')
    plt.colorbar(label='0: Hit (Blue) | 1: Stick (Red)')
    plt.xlabel('Dealer first card')
    plt.ylabel('Player sum')
    plt.title('Optimal policy')
    plt.savefig('./../records/' + file_name)
    plt.close()
    print("Optimal policy saved in records/")