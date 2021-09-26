import speck_jk_rk as sp_jr
import speck as sp
import numpy as np
from keras.models import load_model


def view_jk_rk_accuracy(nr=5, diff=(0x40, 0), jk_net='./', rk_net='./', group_size=2):
    jk_nd = load_model(jk_net)
    rk_nd = load_model(rk_net)
    X_j, X, Y = sp_jr.make_train_data(n=10**6, nr=nr, group_size=group_size, diff=diff)

    l1, acc1 = jk_nd.evaluate(X, Y, batch_size=10000, verbose=0)
    l2, acc2 = jk_nd.evaluate(X_j, Y, batch_size=10000, verbose=0)
    l3, acc3 = rk_nd.evaluate(X, Y, batch_size=10000, verbose=0)
    print('the accuracy of jk_nd over testing set 1 is ', acc1)
    print('the accuracy of jk_nd over testing set 2 is ', acc2)
    print('the accuracy of rk_nd is ', acc3)


def view_nd_accuracy(nr=5, diff=(0x40, 0), net='./', group_size=2):
    nd = load_model(net)
    X, Y = sp.make_dataset_with_group_size(n=10**6, nr=nr, diff=diff, group_size=group_size)
    loss, acc = nd.evaluate(X, Y, batch_size=10000, verbose=0)
    print('acc is ', acc)


# diff=(0x40, 0)
# for nr in range(5, 7):
#     for group_size in [2, 4, 8, 16]:
#         print('nr is ', nr, ' group_size is ', group_size)
#         jk_net='./saved_model/new_model/jk_{}_{}_mc_distinguisher.h5'.format(nr, group_size)
#         rk_net='./saved_model/new_model/rk_{}_{}_mc_distinguisher.h5'.format(nr, group_size)
#         view_jk_rk_accuracy(nr=nr, diff=diff, jk_net=jk_net, rk_net=rk_net, group_size=group_size)


for nr in range(5, 8):
    net = './saved_model/{}_distinguisher.h5'.format(nr)
    view_nd_accuracy(nr=nr, diff=(0x40, 0), net=net, group_size=1)
