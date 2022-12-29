from sklearn.preprocessing import KBinsDiscretizer
import argparse
from cartpole import CartPoleEnv
import matplotlib.pyplot as plt
from utils import *

"""
 * @author Guillaume Gagné-Labelle
 * @student# 20174375
 * @date Dec 23, 2022
 * @project CartPole Problem - Final Project - PHY3075 - UdeM
"""

args_form = argparse.ArgumentParser(allow_abbrev=False)
args_form.add_argument('--policy', type=str, choices=["none","constant0", "constant1", "Q"], default="none")
args_form.add_argument('--render_mode', type=str, choices=["rgb_array","human","quick_human"], default="rgb_array")
args_form.add_argument("--delete_limits", action="store_true", default=False)
args = args_form.parse_args()

Q_table = np.load("Q_advanced.npy")
R, S, perf, x = [], [], [], []
episode = 0

for i in range(750):
    env = CartPoleEnv(render_mode=args.render_mode, delete_limits=args.delete_limits)
    np.random.seed(i)
    observation, terminated = env.reset(), False
    sum_of_reward = 0

    e_tot, e_pot, e_cin, e_rot = [], [], [], []
    time = []

    for t in range(1000):
        if terminated: break

        if args.policy == "none":
            action = -1
        elif args.policy == "constant0":
            action = 0
        elif args.policy == "constant1":
            action = 1
        elif args.policy == "Q":
            n_bins = (4, 2, 32, 16)
            est = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy='uniform')
            est.fit([env.low_state, env.high_state])
            action = Q_policy(Q_table, discretizer(observation, est))
        else: raise Exception

        observation, reward, terminated = env.step(action)
        sum_of_reward += reward
        E, U, K, T = env.energy()

        e_tot.append(E)
        e_pot.append(U)
        e_cin.append(K)
        e_rot.append(T)
        time.append(t*0.02)
    perf.append(sum_of_reward)
    x.append(episode)
    episode+=1

    plt.plot(time, e_tot, label="Totale")
    plt.plot(time, e_pot, label="Potentielle")
    plt.plot(time, e_cin, label="Cinétique")
    plt.plot(time, e_rot, label="Rotationnelle")
    plt.title("Maximisation indirecte de l'énergie potentielle")
    plt.xlabel("Temps [s]")
    plt.ylabel("Énergie [J]")
    plt.legend()
    plt.grid()
    plt.show()