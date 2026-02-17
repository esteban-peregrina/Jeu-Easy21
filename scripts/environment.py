import random

def step(state, action):
    """
    Compute next game state and reward from action and current state.
    
    :param state: dictionnary for the game state
    :param action: string for the player's action
    """
    next_state = state.copy()

    if (action == 'hit'):
        next_state['player']+= draw_card()
        if next_state['player'] > 21 or next_state['player'] < 1: # player eliminates itself
            return next_state, -1, True
        
        return next_state, 0, False
    else: # action == 'stick'
        while (next_state['dealer'] < 17): 
            next_state['dealer']+= draw_card()
            
            if next_state['dealer'] > 21 or next_state['dealer'] < 1: # dealer eliminates itself
                return next_state, 1, True
        
        # computing winner
        if next_state['dealer'] > next_state['player']: 
            return next_state, -1, True
        elif next_state['dealer'] < next_state['player']:
            return next_state, 1, True
        else: # next_state['dealer'] == next_state['player']
            return next_state, 0, True

def draw_card():
    """
    Return signed value from drawing a card.
    """
    value = random.randint(1,10)
    if (random.random() > 2.0/3.0): # rouge
        return -value
    else: # noire
        return value
    
def init_game():
    """
    Return initial game state from drawing two black cards.
    """
    return {'dealer': random.randint(1,10), 'player': random.randint(1,10)}

if __name__ == '__main__':
    state = init_game()
    terminal = False
    print(state)
    while not terminal:
        print("Enter action:")
        action = input()
        state, reward, terminal = step(state, action)
        print(state)
    print("End of the game, reward is", reward)
