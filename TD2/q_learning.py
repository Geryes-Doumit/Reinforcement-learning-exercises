import random
import gym
import numpy as np
from IPython.display import clear_output
import matplotlib.pyplot as plt
import time


def update_q_table(Q, s, a, r, sprime, alpha, gamma):
    Q[s][a] = Q[s][a] + alpha * (r + gamma * np.max(Q[sprime]) - Q[s][a])
    return Q
    """
    This function should update the Q function for a given pair of action-state
    following the q-learning algorithm, it takes as input the Q function, the pair action-state,
    the reward, the next state sprime, alpha the learning rate and gamma the discount factor.
    Return the same input Q but updated for the pair s and a.
    """


def epsilon_greedy(Q, s, epsilone):
    if (random.randint(0, 100) < epsilone*100):
        return env.action_space.sample()
    else:
        return np.argmax(Q[s])
    """
    This function implements the epsilon greedy algorithm.
    Takes as input the Q function for all states, a state s, and epsilon.
    It should return the action to take following the epsilon greedy algorithm.
    """


if __name__ == "__main__":
    env = gym.make("Taxi-v3", render_mode="rgb_array")

    env.reset()
    env.render()

    Q = np.zeros([env.observation_space.n, env.action_space.n])

    alpha = 0.9 # choose your own

    gamma = 0.5 # choose your own

    epsilon = 0.01 # choose your own

    n_epochs = 1000 # choose your own
    max_itr_per_epoch = 10000 # choose your own
    rewards = []

    for e in range(n_epochs):
        r = 0

        S, _ = env.reset()

        for _ in range(max_itr_per_epoch):
            A = epsilon_greedy(Q=Q, s=S, epsilone=epsilon)

            Sprime, R, done, _, info = env.step(A)

            r += R

            Q = update_q_table(
                Q=Q, s=S, a=A, r=R, sprime=Sprime, alpha=alpha, gamma=gamma
            )

            # Update state and put a stoping criteria
            r += R
            S = Sprime
            
            if done:
                break

        print("episode #", e, " : r = ", r)

        rewards.append(r)

    print(f"Average of all {n_epochs} rewards = ", np.mean(rewards))
    print("Average of last 100 rewards = ", np.mean(rewards[-100:]))

    print("Training finished.\n")
    
    # One more episode to see the agent in action
    env = gym.make("Taxi-v3", render_mode="human")
    env.reset()
    env.render()
    for _ in range(100):
        A = epsilon_greedy(Q=Q, s=S, epsilone=epsilon)
        S, R, done, _, info = env.step(A)
        if done:
            break

    # plot the rewards in function of epochs

    plt.plot(rewards)
    plt.xlabel('Epochs')
    plt.ylabel('Reward')
    plt.title('Rewards vs Epochs')
    plt.show()
    """
    
    Evaluate the q-learning algorihtm
    
    """
    

    env.close()
