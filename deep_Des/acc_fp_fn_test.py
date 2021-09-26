import numpy as np
import random
from keras.models import load_model
from os import urandom
import des


Block_size = 64


def resample_and_combine(x, sam_num, sam_len, idx_range):
  index = [i for i in range(idx_range)]
  assert sam_len <= idx_range
  id = np.array([random.sample(index, sam_len) for i in range(sam_num)], dtype=np.uint8).reshape((-1, 1))
  new_x = x[id].reshape((sam_num, sam_len * 2 * Block_size))

  return new_x


def select_false_negative_data(nr=5, diff=(0x40080000, 0x04000000), net='./'):
    nd = load_model(net)
    x = des.make_target_diff_samples(n=2*(10**7), nr=nr, diff_type=1, diff=diff, return_keys=0)
    z = nd.predict(x, batch_size=10000)
    z = np.squeeze(z)
    fn_x = x[z < 0.5]
    return fn_x


def select_false_positive_data(nr=5, net='./'):
    nd = load_model(net)
    x = des.make_target_diff_samples(n=2*(10**7), nr=nr, diff_type=0, return_keys=0)
    z = nd.predict(x, batch_size=10000)
    z = np.squeeze(z)
    fp_x = x[z >= 0.5]
    return fp_x


def detect_fn_mc_acc(fn_x, net='./', group_size=2):
    nd = load_model(net)
    id_range = fn_x.shape[0]
    print('id_range is ', id_range)
    new_x = resample_and_combine(fn_x, sam_num=10**5, sam_len=group_size, idx_range=id_range)
    z = nd.predict(new_x, batch_size=10000)
    if np.sum(z >= 0.5) != 0:
        print('successfully detect false negative', np.sum(z >= 0.5)/(10**5))
    else:
        print('current experiments failed')


def detect_fp_mc_acc(fp_x, net='./', group_size=2):
    nd = load_model(net)
    id_range = fp_x.shape[0]
    new_x = resample_and_combine(fp_x, sam_num=10**5, sam_len=group_size, idx_range=id_range)
    z = nd.predict(new_x, batch_size=10000)
    if np.sum(z < 0.5) != 0:
        print('successfully detect false positive', np.sum(z < 0.5)/(10**5))
    else:
        print('current experiments failed')


def test_distinguisher_acc(nr=5, net='./', diff=(0x40080000, 0x04000000), group_size=2):
    nd = load_model(net)
    x, y = des.make_dataset_with_group_size(n=10**7, nr=nr, diff=diff, group_size=group_size)
    loss, acc = nd.evaluate(x=x, y=y, batch_size=10000, verbose=0)
    print('loss is ', loss, '  acc is ', acc)


nr = 5
net_1 = './saved_model/{}_distinguisher.h5'.format(nr)
net_2_mc = './saved_model/mc/{}_2_mc_distinguisher.h5'.format(nr)
net_4_mc = './saved_model/mc/{}_4_mc_distinguisher.h5'.format(nr)
net_8_mc = './saved_model/mc/{}_8_mc_distinguisher.h5'.format(nr)
net_16_mc = './saved_model/mc/{}_16_mc_distinguisher.h5'.format(nr)

print('cur rounds is ', nr)

fn_x = select_false_negative_data(nr=nr, diff=(0x40080000, 0x04000000), net=net_1)
detect_fn_mc_acc(fn_x, net=net_2_mc, group_size=2)
detect_fn_mc_acc(fn_x, net=net_4_mc, group_size=4)
detect_fn_mc_acc(fn_x, net=net_8_mc, group_size=8)
detect_fn_mc_acc(fn_x, net=net_16_mc, group_size=16)

fp_x = select_false_positive_data(nr=nr, net=net_1)
detect_fp_mc_acc(fp_x, net=net_2_mc, group_size=2)
detect_fp_mc_acc(fp_x, net=net_4_mc, group_size=4)
detect_fp_mc_acc(fp_x, net=net_8_mc, group_size=8)
detect_fp_mc_acc(fp_x, net=net_16_mc, group_size=16)

# test_distinguisher_acc(nr=nr, net=net_1, diff=(0x40080000, 0x04000000), group_size=1)
# test_distinguisher_acc(nr=nr, net=net_2_mc, diff=(0x40080000, 0x04000000), group_size=2)
# test_distinguisher_acc(nr=nr, net=net_4_mc, diff=(0x40080000, 0x04000000), group_size=4)
# test_distinguisher_acc(nr=nr, net=net_8_mc, diff=(0x40080000, 0x04000000), group_size=8)
# test_distinguisher_acc(nr=nr, net=net_16_mc, diff=(0x40080000, 0x04000000), group_size=16)

