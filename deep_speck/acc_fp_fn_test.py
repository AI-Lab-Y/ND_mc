import speck as sp
import numpy as np
import random
from keras.models import load_model
from os import urandom

WORD_SIZE = sp.WORD_SIZE()


# binarize a given ciphertext sample
# ciphertext is given as a sequence of arrays
# each array entry contains one word of ciphertext for all ciphertexts given
def convert_to_binary(l):
  n = len(l)
  k = WORD_SIZE * n
  X = np.zeros((k, len(l[0])), dtype=np.uint8)

  for i in range(k):
    index = i // WORD_SIZE
    offset = WORD_SIZE - 1 - i % WORD_SIZE
    X[i] = (l[index] >> offset) & 1
  X = X.transpose()
  return(X)


def generate_positive_data(n=10**6, nr=7, diff=(0x0040, 0)):
    keys = np.frombuffer(urandom(8 * n), dtype=np.uint16).reshape(4, -1)
    ks = sp.expand_key(keys, nr)
    plain0l = np.frombuffer(urandom(2 * n), dtype=np.uint16)
    plain0r = np.frombuffer(urandom(2 * n), dtype=np.uint16)
    plain1l = plain0l ^ diff[0]
    plain1r = plain0r ^ diff[1]
    ct0l, ct0r = sp.encrypt((plain0l, plain0r), ks)
    ct1l, ct1r = sp.encrypt((plain1l, plain1r), ks)
    x = convert_to_binary([ct0l, ct0r, ct1l, ct1r])
    return x


def generate_negative_data(n=10**6, nr=7):
    keys = np.frombuffer(urandom(8 * n), dtype=np.uint16).reshape(4, -1)
    ks = sp.expand_key(keys, nr)
    plain0l = np.frombuffer(urandom(2 * n), dtype=np.uint16)
    plain0r = np.frombuffer(urandom(2 * n), dtype=np.uint16)
    plain1l = np.frombuffer(urandom(2 * n), dtype=np.uint16)
    plain1r = np.frombuffer(urandom(2 * n), dtype=np.uint16)
    ct0l, ct0r = sp.encrypt((plain0l, plain0r), ks)
    ct1l, ct1r = sp.encrypt((plain1l, plain1r), ks)
    x = convert_to_binary([ct0l, ct0r, ct1l, ct1r])
    return x


def select_false_negative_data(n=10**7, nr=7, net='./', diff=(0x0040, 0)):
    nd = load_model(net)
    fn_x = np.zeros((n, WORD_SIZE * 4), dtype=np.uint8)
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
    fp_x = np.zeros((n, WORD_SIZE * 4), dtype=np.uint8)
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


def test_distinguisher_acc(nr=6, net='./', diff=(0x0040, 0), group_size=2):
    nd = load_model(net)
    x, y = sp.make_dataset_with_group_size(n=10 ** 7, nr=nr, diff=diff, group_size=group_size)
    loss, acc = nd.evaluate(x=x, y=y, batch_size=10000)
    print('loss is ', loss, '  acc is ', acc)


nr = 5
net = './saved_model/{}_distinguisher.h5'.format(nr)
net_2_mc = './saved_model/mc/{}_2_mc_distinguisher.h5'.format(nr)
net_4_mc = './saved_model/mc/{}_4_mc_distinguisher.h5'.format(nr)
net_8_mc = './saved_model/mc/{}_8_mc_distinguisher.h5'.format(nr)
net_16_mc = './saved_model/mc/{}_16_mc_distinguisher.h5'.format(nr)

print('cur rounds is ', nr)

fn_x = select_false_negative_data(n=10**7, nr=nr, net=net, diff=(0x0040, 0))
detect_fn_mc_acc(fn_x, net=net_2_mc, group_size=2)
detect_fn_mc_acc(fn_x, net=net_4_mc, group_size=4)
detect_fn_mc_acc(fn_x, net=net_8_mc, group_size=8)
detect_fn_mc_acc(fn_x, net=net_16_mc, group_size=16)

fp_x = select_false_positive_data(n=10**7, nr=nr, net=net)
detect_fp_mc_acc(fp_x, net=net_2_mc, group_size=2)
detect_fp_mc_acc(fp_x, net=net_4_mc, group_size=4)
detect_fp_mc_acc(fp_x, net=net_8_mc, group_size=8)
detect_fp_mc_acc(fp_x, net=net_16_mc, group_size=16)