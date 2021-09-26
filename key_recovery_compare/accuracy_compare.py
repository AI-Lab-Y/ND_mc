import numpy as np
import os

def testdis(x):
    x = int(x)
    count = 0
    while x > 0:
        count += x % 1
        x  >>= 1
    return (count<=2)

def load_data(dir):
    gohr1=np.array([]); gohr2=np.array([])
    mc1=np.array([]); mc2=np.array([])
    for j in range(10):
        a = np.load(dir+"/gohr/res"+str(j)+"run_sols1.npy")
        gohr1 = np.append(gohr1,a)
        a = np.load(dir+"/gohr/res"+str(j)+"run_sols2.npy")
        gohr2 = np.append(gohr2,a)
        a = np.load(dir+"/mc/res"+str(j)+"run_sols1.npy")
        mc1 = np.append(mc1,a)
        a = np.load(dir+"/mc/res"+str(j)+"run_sols2.npy")
        mc2 = np.append(mc2,a)
    return gohr1,gohr2,mc1,mc2

def count_accuracy(pathname):
    gohr_total = 0; mc_total = 0
    gohr_tt = np.array([]); mc_tt = np.array([])
    for i in os.listdir(pathname):
        if os.path.isdir(os.path.join(pathname,i)):
            dir = os.path.join(pathname,i)
            gohr1,gohr2,mc1,mc2 = load_data(dir)
            gohr_k = 0; mc_k = 0
            for l,sl in zip(gohr1,gohr2):
                if l == 0 and testdis(sl):
                    gohr_k += 1
            for l,sl in zip(mc1,mc2):
                if l == 0 and testdis(sl):
                    mc_k += 1
            with open(dir+"/gohr/gohr_acc.txt","w") as f:
                print("the accuracy is ",gohr_k/10,"%",file = f)
            with open(dir+"/mc/mc_acc.txt","w") as f:
                print("the accuracy is ",mc_k/10,"%",file = f)
            gohr_total += gohr_k; mc_total += mc_k
            gohr_tt = np.append(gohr_tt,gohr_k); mc_tt = np.append(mc_tt,mc_k)
    with open(pathname+"/gohr_acc.txt","w") as f:
        print("the accuracy is ",gohr_total/(10*np.size(gohr_tt)),"%",file = f)
    with open(pathname+"/mc_acc.txt","w") as f:
        print("the accuracy is ",mc_total/(10*np.size(mc_tt)),"%",file = f)
    np.save(open(pathname+'/gohracc.npy','wb'),gohr_tt)
    np.save(open(pathname+'/mcacc.npy','wb'),mc_tt)

# count_accuracy('./result/threshold10 10 res6_1')
# count_accuracy('./result/threshold05 10 res6_10')
# count_accuracy('./result/threshold05 10 res6_1')
count_accuracy('./result/threshold10 10 res6_10')