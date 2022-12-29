import numpy as np
import matplotlib.pyplot as plt

"""
 * @author Guillaume Gagné-Labelle
 * @student# 20174375
 * @date Dec 23, 2022
 * @project CartPole Problem - Final Project - PHY3075 - UdeM
"""

train, test, physics, sapiens, post = [], [], [], [], []
for i in range(10):
    train_tmp = np.load("./rl/y_train_mean%d.npy" % i)
    test_tmp = np.load("./rl/y_test_mean%d.npy" % i)
    phys_tmp = np.load("./physics/y_physics_mean%d.npy" % i)

    train.append(train_tmp)
    test.append(test_tmp)
    physics.append(phys_tmp)

for i in range(751):
    post.append(np.load("./y_Q_mean%d.npy"%i))
train = np.array(train)
test = np.array(test)
physics = np.array(physics)
post = np.array(post)
print(post.shape)

phys = physics*0.02
print("DQN:",test.mean()*0.02,test.std()*0.02)
print("Pysique", phys.mean(), phys.std())

sapiens = np.load("./human/y_homoSapiens.npy")
print("Sapiens", (sapiens*0.02).mean(), (sapiens*0.02).std())
beginning = sapiens[-751:-750]
end = sapiens[-750:]
end_mean = end.reshape(end.shape[0]//25, 25).mean(1)
end_std = end.reshape(end.shape[0]//25, 25).std(1)
sapiens_mean = np.concatenate((beginning,end_mean))
sapiens_std = np.concatenate((beginning,end_std))

post = post[-1]
print("Post", (post*0.02).mean(), (post*0.02).std())
beginning = [post[0]]
end = post[1:]
end_mean = end.reshape(end.shape[0]//25, 25).mean(1)
end_std = end.reshape(end.shape[0]//25, 25).std(1)
post_mean = np.concatenate((beginning,end_mean))
post_std = np.concatenate((beginning,end_std))
post_std = np.clip(post_std, 0, 250)

x = np.linspace(0, 750, train.shape[1] ,endpoint=True)

plt.plot(x, physics.mean(0) * 0.02, label="Physique")
plt.fill_between(x, (physics.mean(0)-physics.std(0)) * 0.02, (physics.mean(0)+physics.std(0))*0.02, alpha=0.3)
plt.plot(x, sapiens_mean*0.02, label="Homo Sapiens")
plt.fill_between(x, (sapiens_mean-sapiens_std)*0.02, (sapiens_mean+sapiens_std)*0.02, alpha=0.3)
plt.plot(x, train.mean(0) * 0.02, label="Entraînement")
plt.fill_between(x, (train.mean(0)-train.std(0)) * 0.02, (train.mean(0)+train.std(0)) * 0.02, alpha=0.3)
plt.plot(x, test.mean(0) * 0.02, label="Évaluation")
plt.fill_between(x, (test.mean(0)-test.std(0)) * 0.02,( test.mean(0)+test.std(0)) * 0.02, alpha=0.3)
plt.plot(x, post_mean*0.02, label="Post-entraîn.")
plt.fill_between(x, (post_mean-post_std)*0.02, (post_mean+post_std)*0.02, alpha=0.3)

plt.title("Performance de différents agents pour maintenir le bâton en équilibre")
plt.xlabel("Épisode")
plt.ylabel("Temps [s]")
plt.legend(loc="center left")
plt.grid()
plt.show()

