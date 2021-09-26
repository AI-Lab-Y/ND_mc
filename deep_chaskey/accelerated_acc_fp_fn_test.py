import numpy as np
import random
from keras.models import load_model
from os import urandom

import chaskey as chk

Block_size = 128


def resample_and_combine(x, sam_num, sam_len, idx_range):
  index = [i for i in range(idx_range)]
  assert sam_len <= idx_range
  id = np.array([random.sample(index, sam_len) for i in range(sam_num)], dtype=np.uint8).reshape((-1, 1))
  new_x = x[id].reshape((sam_num, sam_len * Block_size * 2))

  return new_x


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


def select_false_negative_data(nr=7, net='./', diff=(0x8400, 0x0400, 0, 0)):
    nd = load_model(net)
    x = generate_positive_data(10**7, nr=nr, diff=diff)
    z = nd.predict(x, batch_size=10000)
    z = np.squeeze(z)
    fn_x = x[z < 0.5]
    return fn_x


def select_false_positive_data(nr=7, net='./', diff=(0x8400, 0x0400, 0, 0)):
    nd = load_model(net)
    x = generate_negative_data(10**7, nr=nr)
    z = nd.predict(x, batch_size=10000)
    z = np.squeeze(z)
    fp_x = x[z > 0.5]
    return fp_x


def detect_fn_mc_acc(fn_x, net='./', group_size=2):
    nd = load_model(net)
    id_range = fn_x.shape[0]
    new_x = resample_and_combine(fn_x, sam_num=10000, sam_len=group_size, idx_range=id_range)
    z = nd.predict(new_x, batch_size=10000)
    if np.sum(z >= 0.5) != 0:
        print('successfully detect false negative', np.sum(z > 0.5)/10000)
    else:
        print('current experiments failed')


def detect_fp_mc_acc(fp_x, net='./', group_size=2):
    nd = load_model(net)
    id_range = fp_x.shape[0]
    new_x = resample_and_combine(fp_x, sam_num=10000, sam_len=group_size, idx_range=id_range)
    z = nd.predict(new_x, batch_size=10000)
    if np.sum(z < 0.5) != 0:
        print('successfully detect false positive', np.sum(z < 0.5)/10000)
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
net_2_mc = './saved_model/{}_2_mc_distinguisher.h5'.format(nr)
net_4_mc = './saved_model/{}_4_mc_distinguisher.h5'.format(nr)
net_8_mc = './saved_model/{}_8_mc_distinguisher.h5'.format(nr)
net_16_mc = './saved_model/{}_16_mc_distinguisher.h5'.format(nr)

fn_x = select_false_negative_data(nr=nr, net=net, diff=(0x8400, 0x0400, 0, 0))
print('cur rounds is ', nr)
detect_fn_mc_acc(fn_x, net=net_2_mc, group_size=2)
detect_fn_mc_acc(fn_x, net=net_4_mc, group_size=4)
detect_fn_mc_acc(fn_x, net=net_8_mc, group_size=8)
detect_fn_mc_acc(fn_x, net=net_16_mc, group_size=16)

fp_x = select_false_positive_data(nr=nr, net=net)
detect_fp_mc_acc(fp_x, net=net_2_mc, group_size=2)
detect_fp_mc_acc(fp_x, net=net_4_mc, group_size=4)
detect_fp_mc_acc(fp_x, net=net_8_mc, group_size=8)
detect_fp_mc_acc(fp_x, net=net_16_mc, group_size=16)
