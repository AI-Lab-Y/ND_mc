import numpy as np
import random
from keras.models import load_model
from os import urandom

import chaskey as chk

Block_size = 128


def generate_positive_data(n=10**6, nr=7, diff=(0x8400, 0x0400, 0, 0)):
    t0a, t0b, t0c, t0d, t1a, t1b, t1c, t1d = chk.make_target_diff_dataset(n, nr, 0, head=0,
                                                                          group_size=1, diff=diff, type=1)
    x = chk.convert_to_binary([t0a, t0b, t0c, t0d, t1a, t1b, t1c, t1d])
    return x


def generate_negative_data(n=10**6, nr=7, diff=(0x8400, 0x0400, 0, 0)):
    t0a, t0b, t0c, t0d, t1a, t1b, t1c, t1d = chk.make_target_diff_dataset(n, nr, 0, head=0,
                                                                          group_size=1, diff=diff, type=0)
    x = chk.convert_to_binary([t0a, t0b, t0c, t0d, t1a, t1b, t1c, t1d])
    return x


def select_false_negative_data(n=10**7, nr=7, net='./', diff=(0x8400, 0x0400, 0, 0)):
    nd = load_model(net)
    fn_x = np.zeros((n, Block_size * 2), dtype=np.uint8)
    st = 0
    while 1:
        x = generate_positive_data(10 ** 7, nr=nr, diff=diff)
        z = nd.predict(x, batch_size=10000)
        z = np.squeeze(z)
        num = np.sum(z < 0.5)
        if st + num <= n:
            fn_x[st:st+num] = x[z < 0.5]
            st = st + num
        else:
            tp = x[z < 0.5]
            fn_x[st:] = tp[:n-st]
            break
    return fn_x


def select_false_positive_data(n=10**7, nr=7, net='./'):
    nd = load_model(net)
    fp_x = np.zeros((n, Block_size * 2), dtype=np.uint8)
    st = 0
    while 1:
        x = generate_negative_data(10 ** 7, nr=nr)
        z = nd.predict(x, batch_size=10000)
        z = np.squeeze(z)
        num = np.sum(z > 0.5)
        if st + num <= n:
            fp_x[st:st + num] = x[z > 0.5]
            st = st + num
        else:
            tp = x[z > 0.5]
            fp_x[st:] = tp[:n - st]
            break
    return fp_x


def detect_fn_mc_acc(fn_x, net='./', group_size=2):
    n = fn_x.shape[0]
    assert n % group_size == 0

    nd = load_model(net)
    new_x = fn_x.reshape(n // group_size, -1)
    z = nd.predict(new_x, batch_size=10000)
    if np.sum(z >= 0.5) != 0:
        print('successfully detect false negative', np.sum(z > 0.5)/(n //group_size))
    else:
        print('current experiments failed')


def detect_fp_mc_acc(fp_x, net='./', group_size=2):
    n = fp_x.shape[0]
    assert n % group_size == 0

    nd = load_model(net)
    new_x = fp_x.reshape(n // group_size, -1)
    z = nd.predict(new_x, batch_size=10000)
    if np.sum(z < 0.5) != 0:
        print('successfully detect false positive', np.sum(z < 0.5)/(n //group_size))
    else:
        print('current experiments failed')


# type=0, crypto;  type=1, my model
def test_distinguisher_acc(nr=6, net='./', diff=(0x8400, 0x0400, 0, 0), group_size=2):
    nd = load_model(net)
    X, Y = chk.make_chaskey_dataset(n=10**6, x=nr, y=0, head=0, group_size=group_size, diff=diff)
    loss, acc = nd.evaluate(x=X, y=Y, batch_size=10000)
    print('loss is ', loss, '  acc is ', acc)


nr = 3
net = './saved_model/{}_distinguisher.h5'.format(nr)
net_2_mc = './saved_model/mc/{}_2_mc_distinguisher.h5'.format(nr)
net_4_mc = './saved_model/mc/{}_4_mc_distinguisher.h5'.format(nr)
net_8_mc = './saved_model/mc/{}_8_mc_distinguisher.h5'.format(nr)
net_16_mc = './saved_model/mc/{}_16_mc_distinguisher.h5'.format(nr)

print('cur rounds is ', nr)

fn_x = select_false_negative_data(n=10**7, nr=nr, net=net, diff=(0x8400, 0x0400, 0, 0))
detect_fn_mc_acc(fn_x, net=net_2_mc, group_size=2)
detect_fn_mc_acc(fn_x, net=net_4_mc, group_size=4)
detect_fn_mc_acc(fn_x, net=net_8_mc, group_size=8)
detect_fn_mc_acc(fn_x, net=net_16_mc, group_size=16)

fp_x = select_false_positive_data(n=10**7, nr=nr, net=net)
detect_fp_mc_acc(fp_x, net=net_2_mc, group_size=2)
detect_fp_mc_acc(fp_x, net=net_4_mc, group_size=4)
detect_fp_mc_acc(fp_x, net=net_8_mc, group_size=8)
detect_fp_mc_acc(fp_x, net=net_16_mc, group_size=16)

