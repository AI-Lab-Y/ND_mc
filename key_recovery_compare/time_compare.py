import matplotlib.pyplot as plt
import numpy as np
import os

def load_data(dir):
    gohr_time=np.array([])
    mc_time=np.array([])
    for j in range(10):
        a = np.load(dir+"/gohr/res"+str(j)+"run_time.npy")
        gohr_time = np.append(gohr_time,a)
        a = np.load(dir+"/mc/res"+str(j)+"run_time.npy")
        mc_time = np.append(mc_time,a)
    return gohr_time,mc_time

def plot_time(pathname):
    gohr_time=np.array([])
    mc_time=np.array([])
    for i in os.listdir(pathname):
        if os.path.isdir(os.path.join(pathname,i)):
            dir = os.path.join(pathname,i)
            gohr_time,mc_time = np.append(gohr_time,load_data(dir)[0]), np.append(mc_time,load_data(dir)[1])
    x = np.arange(gohr_time.size)
    plt.figure(1)
    type1 = plt.scatter(x,gohr_time,c='red',s=2)
    plt.savefig(pathname+"/gohr_time.jpg")
    type2 = plt.scatter(x,mc_time,c='c',s=2)
    plt.legend((type1,type2),("gohr","mc"),loc=2,frameon=True)
    plt.savefig(pathname+"/all_time.jpg")
    plt.figure(2)
    type2 = plt.scatter(x,mc_time,c='blue',s=2)
    plt.savefig(pathname+"/mc_time.jpg")
    print("gohr's average time:",np.mean(gohr_time))
    print("mc's average time:",np.mean(mc_time))

# plot_time('./result/threshold05 10 res6_10')
# plot_time('./result/threshold05 10 res6_1')
plot_time('./result/threshold10 10 res6_10')