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
args_form.add_argument('--policy', type=str, choices=["physics","human","Q"], default="Q")
args_form.add_argument('--render_mode', type=str, choices=["rgb_array","human","quick_human"], default="rgb_array")
args_form.add_argument('--test', action="store_true", default=False)
args_form.add_argument("--seeds", type=int, nargs="+", default=[0,1,2,3,4,5,6,7,8,9])
args = args_form.parse_args()

for seed in args.seeds:

    env = CartPoleEnv(render_mode=args.render_mode)
    np.random.seed(seed)
    observation = env.reset()
    n_bins = (4, 2, 32, 16)
    est = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy='uniform')
    est.fit([env.low_state, env.high_state])
    Q_table = np.zeros(n_bins + (2,))
    if args.test:
        Q_table = np.load("Q_advanced.npy")
    y_train, y_test, y_train_mean, y_test_mean, y_train_std, y_test_std = [], [], [], [], [], []
    x = []

    print("---------------------- SEED %d BEGINNING -------------------" % seed)

    n_episodes = 751
    for e in range(n_episodes):
        info_msg = "Episode: %d" % e
        if not args.test:
            Q_table, train_reward = episode(args=args, env=env, Q_table=Q_table, episode=e, est=est, testing=False)
            y_train.append(train_reward)
            if e % 25 == 0:
                y_train_mean.append(np.array(y_train).mean())
                y_train_std.append(np.array(y_train).std())
                info_msg += " | Avg training time: %.2f" % (np.array(y_train).mean() * 0.02)

            if e == int(n_episodes / 4):
                np.save("Q_intermediate", Q_table)
            if e == n_episodes - 1:
                np.save("Q_advanced", Q_table)
        else:
            y_train =[0]

        _, test_reward = episode(args=args, env=env, Q_table=Q_table, episode=e, est=est, testing=True)
        y_test.append(test_reward)

        if e % 25 == 0:
            y_test_mean.append(np.array(y_test).mean())
            y_test_std.append(np.array(y_test_std).std())
            info_msg += " | Avg testing time: %.2f" % (np.array(y_test).mean() * 0.02)
            print(info_msg)
            y_train, y_test = [], []
            x.append(e)


    if not args.test:
        y_train_mean = np.array(y_train_mean)
        y_train_std = np.array(y_train_std)
        plt.plot(x, y_train_mean, label="Entraînement")
        plt.fill_between(x, y_train_mean-y_train_std, y_train_mean+y_train_std, alpha=0.3)

    y_test_mean = np.array(y_test_mean)
    y_test_std = np.array(y_test_std)
    plt.plot(x, y_test_mean, label="Évaluation")
    plt.fill_between(x, y_test_mean-y_test_std, y_test_mean+y_test_std, alpha=0.3)

    plt.title("Performance d'un agent")
    plt.xlabel("Épisode")
    plt.ylabel("Temps")
    plt.legend()
    plt.grid()
    plt.show()
    print("----------------------- SEED %d END ------------------------" % seed)
