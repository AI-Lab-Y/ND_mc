import numpy as np
import random
from keras.models import load_model
from os import urandom
import present as ps


def resample_and_combine(x, group_size=2, block_size=64):
  sam_num = x.shape[0] // group_size
  x_len = sam_num * group_size
  group_x = x[0:x_len].reshape((sam_num, group_size * block_size * 2))

  return group_x


def select_false_negative_data(nr=7, diff=0x9, net='./'):
    nd = load_model(net)
    x = ps.make_target_diff_samples(n=10**7, nr=nr, diff_type=1, diff=diff, return_keys=0)
    z = nd.predict(x, batch_size=10000)
    z = np.squeeze(z)
    fn_x = x[z < 0.5]
    return fn_x


def select_false_positive_data(nr=7, net='./'):
    nd = load_model(net)
    x = ps.make_target_diff_samples(n=10**7, nr=nr, diff_type=0, return_keys=0)
    z = nd.predict(x, batch_size=10000)
    z = np.squeeze(z)
    fp_x = x[z >= 0.5]
    return fp_x


def detect_fn_mc_acc(fn_x, net='./', group_size=2):
    nd = load_model(net)
    new_x = resample_and_combine(fn_x, group_size=group_size, block_size=64)
    new_x_len = new_x.shape[0]
    z = nd.predict(new_x, batch_size=10000)
    if np.sum(z >= 0.5) != 0:
        print('successfully detect false negative', np.sum(z >= 0.5)/new_x_len)
    else:
        print('current experiments failed')


def detect_fp_mc_acc(fp_x, net='./', group_size=2):
    nd = load_model(net)
    new_x = resample_and_combine(fp_x, group_size=group_size, block_size=64)
    new_x_len = new_x.shape[0]
    z = nd.predict(new_x, batch_size=10000)
    if np.sum(z < 0.5) != 0:
        print('successfully detect false positive', np.sum(z < 0.5)/new_x_len)
    else:
        print('current experiments failed')


def test_distinguisher_acc(nr=6, net='./', diff=0x9, group_size=2):
    nd = load_model(net)
    X, Y = ps.make_dataset_with_group_size(n=10 ** 7, nr=nr, diff=diff, group_size=group_size)
    loss, acc = nd.evaluate(x=X, y=Y, batch_size=10000)
    print('loss is ', loss, '  acc is ', acc)


cur_nr = 6
net = './saved_model/' + str(cur_nr) + '_distinguisher.h5'
net_2_mc = './saved_model/' + str(cur_nr) + '_2_mc_distinguisher.h5'
net_4_mc = './saved_model/' + str(cur_nr) + '_4_mc_distinguisher.h5'
net_8_mc = './saved_model/' + str(cur_nr) + '_8_mc_distinguisher.h5'
net_16_mc = './saved_model/' + str(cur_nr) + '_16_mc_distinguisher.h5'

fn_x = select_false_negative_data(nr=cur_nr, diff=0x9, net=net)
print('cur rounds is ', cur_nr)
detect_fn_mc_acc(fn_x, net=net_2_mc, group_size=2)
detect_fn_mc_acc(fn_x, net=net_4_mc, group_size=4)
detect_fn_mc_acc(fn_x, net=net_8_mc, group_size=8)
detect_fn_mc_acc(fn_x, net=net_16_mc, group_size=16)

fp_x = select_false_positive_data(nr=cur_nr, net=net)
detect_fp_mc_acc(fp_x, net=net_2_mc, group_size=2)
detect_fp_mc_acc(fp_x, net=net_4_mc, group_size=4)
detect_fp_mc_acc(fp_x, net=net_8_mc, group_size=8)
detect_fp_mc_acc(fp_x, net=net_16_mc, group_size=16)