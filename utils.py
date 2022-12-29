import numpy as np
import math
import keyboard

"""
 * @author Guillaume Gagné-Labelle
 * @student# 20174375
 * @date Dec 23, 2022
 * @project CartPole Problem - Final Project - PHY3075 - UdeM
"""

def physics_policy(observation):
    x, x_dot, theta, theta_dot = observation[:]
    if abs(theta) < 0.03:
        return int(theta_dot > 0)
    else:
        return int(theta > 0)


def human_policy():
    if keyboard.is_pressed('d') or keyboard.is_pressed("right"):
        action = 1
    elif keyboard.is_pressed('a') or keyboard.is_pressed("left"):
        action = 0
    else:
        action = -1
    return action


def Q_policy(Q_table, state):
    return np.argmax(Q_table[state])

def discretizer(observation, est):
    x, x_dot, theta, theta_dot = observation[:]
    return tuple(map(int, est.transform([[x, x_dot, theta, theta_dot]])[0]))


def new_Q_value(Q_table, reward, state_new, discount_factor=0.995):
    return reward + discount_factor * np.max(Q_table[state_new])


def learning_rate(n, min_rate=0.01):
    return max(min_rate, min(1., 1. - math.log10((n + 1) / 75)))


def exploration_rate(n, min_rate=0.1):
    return max(min_rate, min(1., 1.0 - math.log10((n + 1) / 250)))


def episode(args, env, Q_table, est, episode, testing=False, max_reward = 1000):
    sum_of_reward = 0
    if args.policy == "Q": current_state, terminated = discretizer(env.reset(), est), False
    else: current_state, terminated = env.reset(), False

    while not terminated and sum_of_reward < max_reward:

        if args.policy == "Q":
            if np.random.random() >= exploration_rate(episode) or testing: action = Q_policy(Q_table, current_state)
            else: action = np.random.randint(2)
        elif args.policy == "physics":
            action = physics_policy(current_state)
        elif args.policy == "human":
            action = human_policy()
        elif args.policy == "none":
            action = -1
        elif args.policy == "constant0":
            action = 0
        elif args.policy == "constant1":
            action = 1
        else: raise Exception

        observation, reward, terminated = env.step(action)
        # if reward == 1: _, reward, _, _ = env.energy()
        new_state = discretizer(observation, est)

        sum_of_reward += reward

        if args.policy == "Q":
            if not args.test:
                lr = learning_rate(episode)
                learnt_value = new_Q_value(Q_table, reward, new_state)
                old_value = Q_table[current_state][action]
                Q_table[current_state][action] = (1 - lr) * old_value + lr * learnt_value
            current_state = new_state
        else:
            current_state = observation
    return Q_table, sum_of_reward
