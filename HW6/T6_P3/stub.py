# Imports.
import numpy as np
import numpy.random as npr
import pygame as pg

# uncomment this for animation
from SwingyMonkey import SwingyMonkey

# uncomment this for no animation
# from SwingyMonkeyNoAnimation import SwingyMonkey


X_BINSIZE = 200
Y_BINSIZE = 100
X_SCREEN = 1400
Y_SCREEN = 900


class Learner(object):
    """
    This agent jumps randomly.
    """

    def __init__(self):
        self.last_state = None
        self.last_action = None
        self.last_reward = None
        self.is_first_state = True

        # We initialize our Q-value grid that has an entry for each action and state.
        # (action, rel_x, rel_y)
        self.Q = np.zeros((2, X_SCREEN // X_BINSIZE, Y_SCREEN // Y_BINSIZE))

    def reset(self):
        self.last_state = None
        self.last_action = None
        self.last_reward = None
        self.is_first_state = True

    def discretize_state(self, state):
        """
        Discretize the position space to produce binned features.
        rel_x = the binned relative horizontal distance between the monkey and the tree
        rel_y = the binned relative vertical distance between the monkey and the tree
        """

        rel_x = int((state["tree"]["dist"]) // X_BINSIZE)
        rel_y = int((state["tree"]["top"] - state["monkey"]["top"]) // Y_BINSIZE)
        return (rel_x, rel_y)

    def action_callback(self, state):
        """
        Implement this function to learn things and take actions.
        Return 0 if you don't want to jump and 1 if you do.
        """

        # Do nothing if first state
        if self.is_first_state:
            self.is_first_state = False
            self.last_action = 0
            self.last_state = state
            return self.last_action

        # Define variables
        alpha = 0.1
        epsilon = 0.001
        gamma = 0.9

        # 1. Discretize 'state' to get your transformed 'current state' features.
        last_x, last_y = self.discretize_state(self.last_state)
        curr_x, curr_y = self.discretize_state(state)

        # 2. Perform the Q-Learning update using 'current state' and the 'last state'.
        self.Q[self.last_action, last_x, last_y] += alpha * \
            (self.last_reward + gamma * np.amax(self.Q[:, curr_x, curr_y]) \
                - self.Q[self.last_action, last_x, last_y])

        # 3. Choose the next action using an epsilon-greedy policy.
        new_action = None
        if npr.rand() < epsilon:
            if npr.rand() < 0.3:
                new_action = 1
            else:
                new_action = 0
        else:
            new_action = np.argmax(self.Q[:, curr_x, curr_y])

        self.last_action = new_action
        self.last_state = state

        return self.last_action

    def reward_callback(self, reward):
        """This gets called so you can see what reward you get."""

        self.last_reward = reward


def run_games(learner, hist, iters=100, t_len=100):
    """
    Driver function to simulate learning by having the agent play a sequence of games.
    """
    for ii in range(iters):
        # Make a new monkey object.
        swing = SwingyMonkey(sound=False,  # Don't play sounds.
                             text="Epoch %d" % (ii),  # Display the epoch on screen.
                             tick_length=t_len,  # Make game ticks super fast.
                             action_callback=learner.action_callback,
                             reward_callback=learner.reward_callback)

        # Loop until you hit something.
        while swing.game_loop():
            pass

        # Save score history.
        hist.append(swing.score)

        # Reset the state of the learner.
        learner.reset()
    pg.quit()
    return


if __name__ == '__main__':
    # Select agent.
    agent = Learner()

    # Empty list to save history.
    hist = []

    # Run games. You can update t_len to be smaller to run it faster.
    run_games(agent, hist, 100, 100)
    print(hist)

    # Save history. 
    np.save('hist', np.array(hist))
