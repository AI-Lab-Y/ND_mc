import numpy as np
import random
from keras.models import load_model
from os import urandom

from tensorflow.python.ops.control_flow_ops import group
import SHA256 as sha


def resample_and_combine(x, group_size=2, block_size=256):
  sam_num = x.shape[0] // group_size
  x_len = sam_num * group_size
  group_x = x[0:x_len].reshape((sam_num, group_size * block_size * 2))

  return group_x


cur_nr = 3
net = load_model('./saved_model/' + str(cur_nr) + '_distinguisher.h5')
net_2_mc = load_model('./saved_model/mc/' + str(cur_nr) + '_2_mc_distinguisher.h5')
net_4_mc = load_model('./saved_model/mc/' + str(cur_nr) + '_4_mc_distinguisher.h5')
net_8_mc = load_model('./saved_model/mc/' + str(cur_nr) + '_8_mc_distinguisher.h5')
net_16_mc = load_model('./saved_model/mc/' + str(cur_nr) + '_16_mc_distinguisher.h5')


def select_false_negative_data(n=10**6, nr=3, net=net, diff=[(135, 0x80)]):
    x = sha.make_target_diff_samples(n, nr, 1, diff)
    z = net.predict(x, batch_size=10000)
    z = np.squeeze(z)
    fn_x = x[z < 0.5]
    return fn_x


def select_false_positive_data(n=10**6, nr=3, net=net):
    x = sha.make_target_diff_samples(n, nr, 0)
    z = net.predict(x, batch_size=10000)
    z = np.squeeze(z)
    fp_x = x[z >= 0.5]
    return fp_x


def detect_fn_mc_acc(fn_x, net=net_2_mc, group_size=2):
    new_x = resample_and_combine(fn_x, group_size=group_size, block_size=256)
    new_x_len = new_x.shape[0]
    z = net.predict(new_x, batch_size=10000)
    if np.sum(z >= 0.5) != 0:
        print('successfully detect false negative', np.sum(z >= 0.5)/new_x_len)
    else:
        print('current experiments failed')


def detect_fp_mc_acc(fp_x, net=net_2_mc, group_size=2):
    new_x = resample_and_combine(fp_x, group_size=group_size, block_size=256)
    new_x_len = new_x.shape[0]
    z = net.predict(new_x, batch_size=10000)
    if np.sum(z < 0.5) != 0:
        print('successfully detect false positive', np.sum(z < 0.5)/new_x_len)
    else:
        print('current experiments failed')


def test_distinguisher_acc(n=10**6, nr=3, net=net, type=0, group_size=2, diff=[(135, 0x80)]):     # type=0, crypto;  type=1, my model
    if type == 0:
        x, y = sha.make_dataset(n, nr, diff)
    else:
        x, y = sha.make_dataset(n, nr, diff)
        a, b = x.shape[0], x.shape[1]

        x = x.reshape(a // group_size, b * group_size)
        y = np.array([y[i * group_size] for i in range(a // group_size)])

    loss, acc = net.evaluate(x=x, y=y, batch_size=10000)
    print('loss is ', loss, '  acc is ', acc)


fn_x = select_false_negative_data(n=10**6, nr=cur_nr, net=net)
print('cur rounds is ', cur_nr)
detect_fn_mc_acc(fn_x, net=net_2_mc, group_size=2)
detect_fn_mc_acc(fn_x, net=net_4_mc, group_size=4)
detect_fn_mc_acc(fn_x, net=net_8_mc, group_size=8)
detect_fn_mc_acc(fn_x, net=net_16_mc, group_size=16)

fp_x = select_false_positive_data(n=10**6, nr=cur_nr, net=net)
detect_fp_mc_acc(fp_x, net=net_2_mc, group_size=2)
detect_fp_mc_acc(fp_x, net=net_4_mc, group_size=4)
detect_fp_mc_acc(fp_x, net=net_8_mc, group_size=8)
detect_fp_mc_acc(fp_x, net=net_16_mc, group_size=16)

test_distinguisher_acc(n=10**6, nr=cur_nr, net=net, type=0, group_size=1, diff=[(135, 0x80)])
test_distinguisher_acc(n=10**6, nr=cur_nr, net=net_2_mc, type=1, group_size=2, diff=[(135, 0x80)])
test_distinguisher_acc(n=10**6, nr=cur_nr, net=net_4_mc, type=1, group_size=4, diff=[(135, 0x80)])
test_distinguisher_acc(n=10**6, nr=cur_nr, net=net_8_mc, type=1, group_size=8, diff=[(135, 0x80)])
test_distinguisher_acc(n=10**6, nr=cur_nr, net=net_16_mc, type=1, group_size=16, diff=[(135, 0x80)])