import numpy as np
import speck as sp
from os import urandom
from keras.models import load_model


def make_target_diff_samples(n=10**7, nr=7, diff_type=1, diff=(0x40, 0), joint_key=1, group_size=2):
    p0l = np.frombuffer(urandom(2 * n), dtype=np.uint16)
    p0r = np.frombuffer(urandom(2 * n), dtype=np.uint16)
    if diff_type == 1:
        p1l, p1r = p0l ^ diff[0], p0r ^ diff[1]
    else:
        p1l = np.frombuffer(urandom(2 * n), dtype=np.uint16)
        p1r = np.frombuffer(urandom(2 * n), dtype=np.uint16)

    if joint_key == 1:
        num = n // group_size
        raw_keys = np.frombuffer(urandom(8 * num), dtype=np.uint16).reshape(4, -1)
        keys = np.repeat(raw_keys, group_size, axis=1)
    else:
        keys = np.frombuffer(urandom(8 * n), dtype=np.uint16).reshape(4, -1)
    ks = sp.expand_key(keys, nr)
    c0l, c0r = sp.encrypt((p0l, p0r), ks)
    c1l, c1r = sp.encrypt((p1l, p1r), ks)
    X = sp.convert_to_binary([c0l, c0r, c1l, c1r]).reshape(n // group_size, -1)

    return X


def make_test_dataset(n=10**7, nr=6, diff=(0x40, 0), group_size=2):
    assert n % group_size == 0
    num = n // 2
    X_p = make_target_diff_samples(n=num, nr=nr, diff_type=1, diff=diff, joint_key=1, group_size=group_size)
    X_n = make_target_diff_samples(n=num, nr=nr, diff_type=1, diff=diff, joint_key=0, group_size=group_size)
    X = np.concatenate((X_p, X_n), axis=0)
    Y_p = [1 for _ in range(num // group_size)]
    Y_n = [0 for _ in range(num // group_size)]
    Y = np.concatenate((Y_p, Y_n))

    return X, Y


def test_the_joint_key_distingisher(n=10**7, net='./n', nr=6, diff=(0x40, 0), group_size=2):
    nd = load_model(net)
    X, Y = make_test_dataset(n=n, nr=nr, diff=diff, group_size=group_size)
    loss, acc = nd.evaluate(X, Y, batch_size=10000, verbose=0)
    print('acc is ', acc)


diff = (0x40, 0)
for nr in [5, 6, 7]:
    for group_size in [2, 4, 8, 16]:
        net = './saved_model/new_model/jk_{}_{}_mc_distinguisher.h5'.format(nr, group_size)
        test_the_joint_key_distingisher(n=10 ** 7, net=net, nr=nr, diff=diff, group_size=group_size)
