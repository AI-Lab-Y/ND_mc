import speck as sp
import numpy as np
from keras.models import load_model
from os import urandom
import random

selected_bits_1 = [14,13,12,11,10,9,8,7]
selected_bits_3 = [15,14,13,12,11,10,9,8,7,3,2,1,0]
selected_bits_2 = [15,14,13,12,11,10,9,8, 7,6,5,4,3,2,1,0]
word_size = sp.WORD_SIZE()
mask_val = (1 << word_size) - 1
# p1 = tp_tk, p2 = tp_fk, p3 = tn_tk, p4 = tn_fk
# p1, p3, p4 are easy to calculate, p2 is related with the hamming distance between tk and fk


def resample_and_combine(x, group_size=2):
    sam_num = x.shape[0] // group_size
    b = x.shape[1]
    x_len = sam_num * group_size
    group_x = x[0:x_len].reshape((sam_num, group_size * b))
    return group_x

def extract_sensitive_bits(raw_x, bits=[14, 13, 12, 11, 10, 9, 8]):
    # get new-x according to sensitive bits
    id0 = [word_size - 1 - v for v in bits]
    # print('id0 is ', id0)
    id1 = [v + word_size * i for i in range(4) for v in id0]
    # print('id1 is ', id1)
    new_x = raw_x[:, id1]
    # print('new_x shape is ', np.shape(new_x))

    return new_x


# diff = ()
def make_target_diff_samples(n=64*10, nr=8, diff_type=1, diff=(0x211, 0xa04)):
    p0l = np.frombuffer(urandom(2 * n), dtype=np.uint16)
    p0r = np.frombuffer(urandom(2 * n), dtype=np.uint16)
    # positive sample
    if diff_type == 1:
        p1l = p0l ^ diff[0]
        p1r = p0r ^ diff[1]
    # negative sample
    else:
        p1l = np.frombuffer(urandom(2 * n), dtype=np.uint16)
        p1r = np.frombuffer(urandom(2 * n), dtype=np.uint16)
    # n different master keys for n plaintext pairs
    key = np.frombuffer(urandom(8*n), dtype=np.uint16).reshape(4, -1)
    ks = sp.expand_key(key, nr)
    c0l, c0r = sp.encrypt((p0l, p0r), ks)
    c1l, c1r = sp.encrypt((p1l, p1r), ks)

    return c0l, c0r, c1l, c1r

def show_distinguisher_acc(n=10**7, nr=7, net_path='', diff=(0x0040, 0), bits=[14, 13, 12, 11, 10, 9, 8], group_size=1):
    net = load_model(net_path)

    nn = n // group_size
    c0l, c0r, c1l, c1r = make_target_diff_samples(n=n, nr=nr, diff_type=1, diff=diff)
    raw_x = sp.convert_to_binary([c0l, c0r, c1l, c1r])
    x = extract_sensitive_bits(raw_x, bits=bits)
    x = resample_and_combine(x, group_size=group_size)
    y = net.predict(x, batch_size=10000)
    y = np.squeeze(y)
    tp = np.sum(y > 0.5) / nn
    fn = 1 - tp

    c0l, c0r, c1l, c1r = make_target_diff_samples(n=n, nr=nr, diff_type=0)
    raw_x = sp.convert_to_binary([c0l, c0r, c1l, c1r])
    x = extract_sensitive_bits(raw_x, bits=bits)
    x = resample_and_combine(x, group_size=group_size)
    y = net.predict(x, batch_size=10000)
    y = np.squeeze(y)
    tn = np.sum(y <= 0.5) / nn
    fp = 1 - tn

    print('acc of cur distinguisher is ', (tp + tn) / 2)
    print('tp_to_tp: ', tp, ' tp_to_fn: ', fn, ' tn_to_tn: ', tn, ' tn_to_fp: ', fp)



def cal_p1_p3(n=10**7, nr=7, c3=0.5, net_path='', diff=(0x0040, 0), bits=[14, 13, 12, 11, 10, 9, 8], group_size=1):
    net = load_model(net_path)

    group_num = n // group_size
    n = group_num * group_size
    keys = np.frombuffer(urandom(8*n), dtype=np.uint16).reshape(4, -1)
    ks = sp.expand_key(keys, nr+1)
    p0l = np.frombuffer(urandom(2 * n), dtype=np.uint16)
    p0r = np.frombuffer(urandom(2 * n), dtype=np.uint16)

    for i in range(1, 4):
        if i == 1:
            p1l = p0l ^ diff[0]
            p1r = p0r ^ diff[1]
        elif i == 2:
            continue
        else:
            p1l = np.frombuffer(urandom(2 * n), dtype=np.uint16)
            p1r = np.frombuffer(urandom(2 * n), dtype=np.uint16)
        c0l, c0r = sp.encrypt((p0l, p0r), ks)
        c1l, c1r = sp.encrypt((p1l, p1r), ks)

        if i == 1:
            dk = ks[nr]
        else:
            dk = np.frombuffer(urandom(2 * n), dtype=np.uint16)
        d0l, d0r = sp.dec_one_round((c0l, c0r), dk)
        d1l, d1r = sp.dec_one_round((c1l, c1r), dk)
        raw_x = sp.convert_to_binary([d0l, d0r, d1l, d1r])
        x = extract_sensitive_bits(raw_x, bits=bits)
        if group_size != 1:
            x = resample_and_combine(x, group_size=group_size)
        Z = net.predict(x, batch_size=10000)

        acc = np.sum(Z > c3) / group_num
        if i == 1:
            p1 = acc
        elif i == 3:
            p3 = acc

    print("p1: ", p1, ' p3: ', p3)
    return p1, p3

def gen_fk(arr):
    fk = 0
    for v in arr:
        fk = fk | (1 << v)

    return fk

def cal_p2_d1_for_speck(n=10**7, nr=7, c3=0.5, net_path='', diff=(0x0040, 0), bits=selected_bits_2, group_size=1):
    net = load_model(net_path)

    group_num = n // group_size
    n = group_num * group_size
    d1 = len(bits)
    p2_d1 = np.zeros(d1+1)
    sample_range = [i for i in range(d1)]
    # d1 = 1, ... , bit_num - 1
    # i = 0, 1, 2,..., bit_num
    for i in range(d1+1):
        print('cur i is ', i)
        keys = np.frombuffer(urandom(8 * n), dtype=np.uint16).reshape(4, -1)
        ks = sp.expand_key(keys, nr + 1)

        # n positive samples
        pt0a = np.frombuffer(urandom(2 * n), dtype=np.uint16)
        pt1a = np.frombuffer(urandom(2 * n), dtype=np.uint16)
        pt0b, pt1b = pt0a ^ diff[0], pt1a ^ diff[1]
        ct0a, ct1a = sp.encrypt((pt0a, pt1a), ks)
        ct0b, ct1b = sp.encrypt((pt0b, pt1b), ks)

        fks = np.array([gen_fk(random.sample(sample_range, i)) for j in range(n)], dtype=np.uint16)
        fks = fks ^ ks[nr]
        c0a, c1a = sp.dec_one_round((ct0a, ct1a), fks)
        c0b, c1b = sp.dec_one_round((ct0b, ct1b), fks)
        raw_X = sp.convert_to_binary([c0a, c1a, c0b, c1b])
        X = extract_sensitive_bits(raw_X, bits=bits)
        if group_size != 1:
            X= resample_and_combine(X, group_size=group_size)

        Z = net.predict(X, batch_size=10000)
        Z = np.squeeze(Z)
        p2_d1[i] = np.sum(Z > c3) / group_num  # save the probability
        print('cur p2_d1 is ', p2_d1[i])
    
    # save p2_d1 for deep_speck_teacher
    # np.save('./p2_estimation_res/teacher/{}/{}_{}_p2_d1.npy'.format(nr, nr, c3), p2_d1)

    # save p2_d1 for deep_speck_mc_teacher
    # np.save('./p2_estimation_res/mc_teacher/{}/{}_{}_{}_p2_d1.npy'.format(nr, nr, group_size, c3), p2_d1)

    # save p2_d1 for deep_speck_student
    # np.save('./p2_estimation_res/student/{}/{}_{}_p2_d1.npy'.format(nr, nr, c3), p2_d1)
    
    # save p2_d1 for deep_speck_mc_student
    # np.save('./p2_estimation_res/mc_student/{}/{}_{}_{}_p2_d1.npy'.format(nr, nr, group_size, c3), p2_d1)

#------------------------------teacher
# cal acc for deep_speck
nr = 6
print('performance of deep_speck_teacher for {} rounds'.format(nr))
show_distinguisher_acc(n=10**7, nr=nr, net_path='./saved_model/teacher/{}_distinguisher.h5'.format(nr), diff=(0x0040,0x0), bits=selected_bits_2)

# cal acc for deep_speck_mc
nr = 6
k = 2
print('performance of deep_speck_mcteacher for {} rounds'.format(nr))
print('k = {}'.format(k))
show_distinguisher_acc(n=10**7, nr=nr, net_path='./saved_model/mc_teacher/{}_{}_mc_distinguisher.h5'.format(nr, k), diff=(0x0040, 0x0), bits=selected_bits_2, group_size=k)

# cal p1 p3 for deep_speck
print('p1 p3 of deep_speck_teacher for', 5, 'rounds:')
cal_p1_p3(n=10**7, nr=5, c3=0.5, net_path='./saved_model/teacher/5_distinguisher.h5', diff=(0x0040,0x0), bits=selected_bits_2)
print('p1 p3 of deep_speck_teacher for', 6, 'rounds:')
cal_p1_p3(n=10**7, nr=6, c3=0.5, net_path='./saved_model/teacher/6_distinguisher.h5', diff=(0x0040,0x0), bits=selected_bits_2)
print('p1 p3 of deep_speck_teacher for', 7, 'rounds:')
cal_p1_p3(n=10**7, nr=7, c3=0.5, net_path='./saved_model/teacher/7_distinguisher.h5', diff=(0x0040,0x0), bits=selected_bits_2)

# cal p1 p3 for deep_speck_mc
for nr in range(5,8):
    print('p1 p3 of deep_speck_mc_teacher for', nr, 'rounds:')
    print('k =', 2)
    cal_p1_p3(n=10**7, nr=nr, c3=0.5, net_path='./saved_model/mc_teacher/'+str(nr)+'_'+str(2)+'_mc_distinguisher.h5', diff=(0x0040,0x0), bits=selected_bits_2, group_size=2)
    print('k =', 4)
    cal_p1_p3(n=10**7, nr=nr, c3=0.5, net_path='./saved_model/mc_teacher/'+str(nr)+'_'+str(4)+'_mc_distinguisher.h5', diff=(0x0040,0x0), bits=selected_bits_2, group_size=4)
    print('k =', 8)
    cal_p1_p3(n=10**7, nr=nr, c3=0.5, net_path='./saved_model/mc_teacher/'+str(nr)+'_'+str(8)+'_mc_distinguisher.h5', diff=(0x0040,0x0), bits=selected_bits_2, group_size=8)
    print('k =', 16)
    cal_p1_p3(n=10**7, nr=nr, c3=0.5, net_path='./saved_model/mc_teacher/'+str(nr)+'_'+str(16)+'_mc_distinguisher.h5', diff=(0x0040,0x0), bits=selected_bits_2, group_size=16)

# cal p2_d1 for deep_speck
print('p2_d1 of deep_speck_teacher for', 5, 'rounds:')
cal_p2_d1_for_speck(n=10**7, nr=5, c3=0.5, net_path='./saved_model/teacher/5_distinguisher.h5', diff=(0x0040,0x0), bits=selected_bits_2)
print('p2_d1 of deep_speck_teacher for', 6, 'rounds:')
cal_p2_d1_for_speck(n=10**7, nr=6, c3=0.5, net_path='./saved_model/teacher/6_distinguisher.h5', diff=(0x0040,0x0), bits=selected_bits_2)
print('p2_d1 of deep_speck_teacher for', 7, 'rounds:')
cal_p2_d1_for_speck(n=10**7, nr=7, c3=0.5, net_path='./saved_model/teacher/7_distinguisher.h5', diff=(0x0040,0x0), bits=selected_bits_2)

# cal p2_d1 for deep_speck_mc
cur_nr = 6
k = 2
print('p2_d1 of deep_speck_mc_teacher for', cur_nr, 'rounds:')
print('k =', k)
cal_p2_d1_for_speck(n=10**7, nr=cur_nr, c3=0.5, net_path='./saved_model/mc_teacher/'+str(cur_nr)+'_'+str(k)+'_mc_distinguisher.h5', diff=(0x0040,0x0), bits=selected_bits_2, group_size=k)
#-------------------------------------

#------------------------------student
# cal acc for student_deep_speck
cur_nr = 6
net_path = './saved_model/student/student_{}_distinguisher.h5'.format(cur_nr)
print('acc of deep_speck_student for', cur_nr, 'rounds:')
show_distinguisher_acc(n=10**7, nr=cur_nr, net_path=net_path, diff=(0x0040, 0x0), bits=selected_bits_1, group_size=1)

# cal acc for student_deep_speck_mc
cur_nr = 6
k = 2
net_path = './saved_model/mc_student/student_{}_{}_mc_distinguisher.h5'.format(cur_nr, k)
print('acc of deep_speck_mc_student for', cur_nr, 'rounds:')
print('k =', k)
show_distinguisher_acc(n=10**7, nr=cur_nr, net_path=net_path, diff=(0x0040, 0x0), bits=selected_bits_1, group_size=k)

# cal p1, p3 for student_deep_speck
cur_nr = 6
net_path = './saved_model/student/student_{}_distinguisher.h5'.format(cur_nr)
print('p1 p3 of deep_speck_student for', cur_nr, 'rounds:')
cal_p1_p3(n=10**7, nr=cur_nr, c3=0.5, net_path=net_path, diff=(0x40, 0x0), bits=selected_bits_1, group_size=1)

# cal p1, p3 for student_deep_speck_mc
cur_nr = 6
k = 2
net_path = './saved_model/mc_student/student_{}_{}_mc_distinguisher.h5'.format(cur_nr, k)
print('p1 p3 of deep_speck_mc_student for', cur_nr, 'rounds:')
print('k =', k)
cal_p1_p3(n=10**7, nr=cur_nr, c3=0.5,net_path=net_path, diff=(0x40, 0x0), bits=selected_bits_1, group_size=k)

# cal p2_d1 for student_deep_speck
cur_nr = 6
net_path = './saved_model/student/student_{}_distinguisher.h5'.format(cur_nr)
print('p2_d1 of deep_speck_student for', cur_nr, 'rounds:')
cal_p2_d1_for_speck(n=10**7, nr=cur_nr, c3=0.5, net_path=net_path, diff=(0x40, 0x0), bits=selected_bits_1, group_size=1)

# cal p2_d1 for student_deep_speck_mc
cur_nr = 6
k = 2
net_path = './saved_model/mc_student/student_{}_{}_mc_distinguisher.h5'.format(cur_nr, k)
print('p2_d1 of deep_speck_mc_student for', cur_nr, 'rounds:')
print('k =', k)
cal_p2_d1_for_speck(n=10**7, nr=cur_nr, c3=0.5, net_path=net_path, diff=(0x40,0x0), bits=selected_bits_1, group_size=k)
#-------------------------------------