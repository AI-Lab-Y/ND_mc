import numpy as np
import random
from keras.models import load_model
from os import urandom
import SHA256 as sha


block_size = 256


def select_false_negative_data(n=10**7, nr=7, net='./', diff=[(135, 0x80)]):
    nd = load_model(net)
    fn_x = np.zeros((n, block_size * 2), dtype=np.uint8)
    st = 0
    while 1:
        x = sha.make_target_diff_samples(n=10**6, Nr=nr, diff_type=1, diff=diff)
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
    fp_x = np.zeros((n, block_size * 2), dtype=np.uint8)
    st = 0
    while 1:
        x = sha.make_target_diff_samples(n=10 ** 6, Nr=nr, diff_type=0)
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
        print('successfully detect false negative', np.sum(z >= 0.5)/(n // group_size))
    else:
        print('current experiments failed')


def detect_fp_mc_acc(fp_x, net='./', group_size=2):
    n = fp_x.shape[0]
    assert n % group_size == 0

    nd = load_model(net)
    new_x = fp_x.reshape(n // group_size, -1)
    z = nd.predict(new_x, batch_size=10000)
    if np.sum(z < 0.5) != 0:
        print('successfully detect false positive', np.sum(z < 0.5)/(n // group_size))
    else:
        print('current experiments failed')


def test_distinguisher_acc(n=10**7, nr=3, net='./', diff=[(135, 0x80)], group_size=2):
    nd = load_model(net)
    x, y = sha.make_dataset(n=n, Nr=nr, diff=diff, group_size=group_size)
    loss, acc = nd.evaluate(x=x, y=y, batch_size=10000, verbose=0)
    print('loss is ', loss, '  acc is ', acc)


cur_nr = 3
net = './saved_model/' + str(cur_nr) + '_distinguisher.h5'
net_2_mc = './saved_model/mc/' + str(cur_nr) + '_2_mc_distinguisher.h5'
net_4_mc = './saved_model/mc/' + str(cur_nr) + '_4_mc_distinguisher.h5'
net_8_mc = './saved_model/mc/' + str(cur_nr) + '_8_mc_distinguisher.h5'
net_16_mc = './saved_model/mc/' + str(cur_nr) + '_16_mc_distinguisher.h5'

print('cur rounds is ', cur_nr)

fn_x = select_false_negative_data(n=10**7, nr=cur_nr, net=net, diff=[(135, 0x80)])
detect_fn_mc_acc(fn_x, net=net_2_mc, group_size=2)
detect_fn_mc_acc(fn_x, net=net_4_mc, group_size=4)
detect_fn_mc_acc(fn_x, net=net_8_mc, group_size=8)
detect_fn_mc_acc(fn_x, net=net_16_mc, group_size=16)

fp_x = select_false_positive_data(n=10**7, nr=cur_nr, net=net)
detect_fp_mc_acc(fp_x, net=net_2_mc, group_size=2)
detect_fp_mc_acc(fp_x, net=net_4_mc, group_size=4)
detect_fp_mc_acc(fp_x, net=net_8_mc, group_size=8)
detect_fp_mc_acc(fp_x, net=net_16_mc, group_size=16)

# n = 10**6
# test_distinguisher_acc(n=n, nr=cur_nr, net=net, diff=[(135, 0x80)], group_size=1)
# test_distinguisher_acc(n=n, nr=cur_nr, net=net_2_mc, diff=[(135, 0x80)], group_size=2)
# test_distinguisher_acc(n=n, nr=cur_nr, net=net_4_mc, diff=[(135, 0x80)], group_size=4)
# test_distinguisher_acc(n=n, nr=cur_nr, net=net_8_mc, diff=[(135, 0x80)], group_size=8)
# test_distinguisher_acc(n=n, nr=cur_nr, net=net_16_mc, diff=[(135, 0x80)], group_size=16)