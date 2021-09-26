import speck as sp
import numpy as np
import time

from keras.models import load_model
from os import urandom, path, mkdir

net_path_s = './saved_model/student/student_6_distinguisher.h5'
net_path_t = './saved_model/teacher/6_distinguisher.h5'
MASK_VAL = 2 ** sp.WORD_SIZE() - 1
word_size = sp.WORD_SIZE()

def resample_and_combine(x, group_size=2):
  sam_num = x.shape[0] // group_size
  b = x.shape[1]
  x_len = sam_num * group_size
  group_x = x[0:x_len].reshape((sam_num, group_size * b))

  return group_x

#make a plaintext structure
#takes as input a sequence of plaintexts, a desired plaintext input difference, and a set of neutral bits
def make_structure(pt0, pt1, diff=(0x211,0xa04),neutral_bits = [20,21,22]):
  #p0=(n,),p1=(n,)
  p0 = np.copy(pt0); p1 = np.copy(pt1);
  #p0=(n,1),p1=(n,1)
  p0 = p0.reshape(-1,1); p1 = p1.reshape(-1,1);
  for i in neutral_bits:
    d = 1 << i; d0 = d >> 16; d1 = d & 0xffff
    p0 = np.concatenate([p0,p0^d0],axis=1); 
    p1 = np.concatenate([p1,p1^d1],axis=1);
  #p0=(n,32),p1=(n,32)
  p0b = p0 ^ diff[0]; p1b = p1 ^ diff[1];
  return(p0,p1,p0b,p1b);

def make_target_diff_good_samples(n=2**12, nr=10, diff=(0x2800, 0x10), group_size_log2=1, neural_bits=[20,21,22]):
    if group_size_log2 == 1:
        p0l = np.frombuffer(urandom(2 * n), dtype=np.uint16)
        p0r = np.frombuffer(urandom(2 * n), dtype=np.uint16)
        p0l, p0r, p1l, p1r = make_structure(p0l, p0r, diff=diff, neutral_bits=neural_bits[:3])
        goal_index = [0, 3]
        p0l = p0l[:,goal_index]; p0r = p0r[:,goal_index]; p1l = p1l[:,goal_index]; p1r = p1r[:,goal_index];
        p0l = np.reshape(p0l, (2*n,)); p0r = np.reshape(p0r, (2*n,)); p1l = np.reshape(p1l, (2*n,)); p1r = np.reshape(p1r, (2*n,));
        p0l, p0r = sp.dec_one_round((p0l, p0r), 0)
        p1l, p1r = sp.dec_one_round((p1l, p1r), 0)
        key = np.frombuffer(urandom(8), dtype=np.uint16).reshape(4, -1)
        ks = sp.expand_key(key, nr)
        c0l, c0r = sp.encrypt((p0l, p0r), ks)
        c1l, c1r = sp.encrypt((p1l, p1r), ks)
        return c0l, c0r, c1l, c1r, ks[nr - 1][0]


# type = 1, return the complete sk, or return sk & 0xff
def make_target_diff_samples(n=2**12, nr=10, diff=(0x2800, 0x10), type=1, group_size_log2=1, neural_bits=[20,21,22]):
    p0l = np.frombuffer(urandom(2 * n), dtype=np.uint16)
    p0r = np.frombuffer(urandom(2 * n), dtype=np.uint16)
    p0l, p0r, p1l, p1r = make_structure(p0l, p0r, diff=diff, neutral_bits=neural_bits[:group_size_log2])


    p0l, p0r = sp.dec_one_round((p0l, p0r), 0)
    p1l, p1r = sp.dec_one_round((p1l, p1r), 0)

    key = np.frombuffer(urandom(8), dtype=np.uint16).reshape(4, -1)
    ks = sp.expand_key(key, nr)

    c0l, c0r = sp.encrypt((p0l, p0r), ks)
    c1l, c1r = sp.encrypt((p1l, p1r), ks)
    c0l = np.reshape(c0l, n * (2**group_size_log2))
    c0r = np.reshape(c0r, n * (2**group_size_log2))
    c1l = np.reshape(c1l, n * (2**group_size_log2))
    c1r = np.reshape(c1r, n * (2**group_size_log2))
    if type == 1:
        return c0l, c0r, c1l, c1r, ks[nr - 1][0]
    else:
        return c0l, c0r, c1l, c1r, ks[nr-1][0] & np.uint16(0xff)

def make_target_diff_samples_with_data_reuse(n=2**12, nr=10, diff=(0x2800, 0x10), group_size_log2=1, neural_bits=[20,21,22]):
    if group_size_log2 == 1:
        ng = n // 4 + 1
        p0l = np.frombuffer(urandom(2 * ng), dtype=np.uint16)
        p0r = np.frombuffer(urandom(2 * ng), dtype=np.uint16)
        p0l, p0r, p1l, p1r = make_structure(p0l, p0r, diff=diff, neutral_bits=neural_bits[:3])
        goal_index = [0, 3, 5, 6] # index of [null, ab, ac, bc]
        p0l = p0l[:,goal_index]; p0r = p0r[:,goal_index]; p1l = p1l[:,goal_index]; p1r = p1r[:,goal_index];
        pg0l = np.concatenate((p0l[:,[0]],p0l[:,[1]],p0l[:,[1]],p0l[:,[2]],p0l[:,[2]],p0l[:,[3]],p0l[:,[3]],p0l[:,[0]]), axis=1)
        pg0r = np.concatenate((p0r[:,[0]],p0r[:,[1]],p0r[:,[1]],p0r[:,[2]],p0r[:,[2]],p0r[:,[3]],p0r[:,[3]],p0r[:,[0]]), axis=1)
        pg1l = np.concatenate((p1l[:,[0]],p1l[:,[1]],p1l[:,[1]],p1l[:,[2]],p1l[:,[2]],p1l[:,[3]],p1l[:,[3]],p1l[:,[0]]), axis=1)
        pg1r = np.concatenate((p1r[:,[0]],p1r[:,[1]],p1r[:,[1]],p1r[:,[2]],p1r[:,[2]],p1r[:,[3]],p1r[:,[3]],p1r[:,[0]]), axis=1)
        pg0l = np.reshape(pg0l, ng*8); pg0r = np.reshape(pg0r, ng*8); pg1l = np.reshape(pg1l, ng*8); pg1r = np.reshape(pg1r, ng*8);
        pg0l = pg0l[:2*n]; pg0r = pg0r[:2*n]; pg1l = pg1l[:2*n]; pg1r = pg1r[:2*n];
        p0l, p0r = sp.dec_one_round((pg0l, pg0r), 0)
        p1l, p1r = sp.dec_one_round((pg1l, pg1r), 0)
        key = np.frombuffer(urandom(8), dtype=np.uint16).reshape(4, -1)
        ks = sp.expand_key(key, nr)
        c0l, c0r = sp.encrypt((p0l, p0r), ks)
        c1l, c1r = sp.encrypt((p1l, p1r), ks)
        return c0l, c0r, c1l, c1r, ks[nr - 1][0]

def extract_sensitive_bits(raw_x, bits=[14, 13, 12, 11, 10, 9, 8, 7]):
    # get new-x according to sensitive bits
    id0 = [word_size - 1 - v for v in bits]
    # print('id0 is ', id0)
    id1 = [v + word_size * i for i in range(4) for v in id0]
    # print('id1 is ', id1)
    new_x = raw_x[:, id1]
    # print('new_x shape is ', np.shape(new_x))

    return new_x


def naive_key_recovery_attack(t=100, n1=0, th1=0, n2=0, th2=0, nr=10, c3=0.5, net_path_s='', net_path_t='', diff=(0x2800, 0x10), bits=[14, 13, 12, 11, 10, 9, 8, 7], group_size_log2=0, neural_bits=[20,21,22], data_reuse=False):
    student = load_model(net_path_s)
    teacher = load_model(net_path_t)

    bits_len = len(bits)
    sub_space_1 = 2**(bits_len)
    sub_space_2 = 2**(word_size - bits_len)
    cnt_1 = np.zeros((t, sub_space_1), dtype=np.uint32)
    cnt_2 = np.zeros((t, 2**word_size), dtype=np.uint32)
    tk = np.zeros(t, dtype=np.uint16)
    time_consumption = np.zeros(t)
    index_1_1 = 0
    index_1_2 = 0
    index_2_1 = 0
    index_2_2 = 0
    index_3_1 = 0
    index_3_2 = 0
    for i in range(t):
        print('i is ', i)
        if data_reuse == True:
            c0l, c0r, c1l, c1r, true_key = make_target_diff_samples_with_data_reuse(n=n1, nr=nr, diff=diff, group_size_log2=group_size_log2, neural_bits=neural_bits)
        else:
            c0l, c0r, c1l, c1r, true_key = make_target_diff_samples(n=n1, nr=nr, diff=diff, type=1, group_size_log2=group_size_log2, neural_bits=neural_bits)
        tk[i] = true_key

        c0l_2, c0r_2 = c0l[0:n2*(2**group_size_log2)], c0r[0:n2*(2**group_size_log2)]
        c1l_2, c1r_2 = c1l[0:n2*(2**group_size_log2)], c1r[0:n2*(2**group_size_log2)]

        start = time.time()

        for sk_1 in range(sub_space_1):
            t0l, t0r = sp.dec_one_round((c0l, c0r), sk_1)
            t1l, t1r = sp.dec_one_round((c1l, c1r), sk_1)
            raw_X = sp.convert_to_binary([t0l, t0r, t1l, t1r])
            X = extract_sensitive_bits(raw_X, bits=bits)
            X = resample_and_combine(X, group_size=2**group_size_log2)
            Z = student.predict(X, batch_size=n1)
            Z = np.squeeze(Z)
            tp_1 = np.sum(Z > c3)
            cnt_1[i][sk_1] = tp_1
            if tp_1 < th1:
                continue
            for sk_2 in range(sub_space_2):
                sk = (sk_2 << bits_len) | sk_1
                t0l_2, t0r_2 = sp.dec_one_round((c0l_2, c0r_2), sk)
                t1l_2, t1r_2 = sp.dec_one_round((c1l_2, c1r_2), sk)

                X_2 = sp.convert_to_binary([t0l_2, t0r_2, t1l_2, t1r_2])
                X_2 = resample_and_combine(X_2, group_size=2**group_size_log2)
                Z_2 = teacher.predict(X_2, batch_size=n2)
                Z_2 = np.squeeze(Z_2)

                tp_2 = np.sum(Z_2 > c3)
                cnt_2[i][sk] = tp_2
        
        end = time.time()
        time_consumption[i] = end - start
        key_survive_num_1 = np.sum(cnt_1[i, :] > th1)
        key_survive_num_2 = np.sum(cnt_2[i, :] > th2)
        print('the number of surviving keys in stage1 is', key_survive_num_1)
        print('the number of surviving keys in stage2 is', key_survive_num_2)
        print('time cost is {}s'.format(end - start))
        if cnt_1[i][true_key & (sub_space_1 - 1)] > th1:
            index_1_1 += 1
        if cnt_2[i][true_key] > th2:
            index_1_2 += 1
        index_2_1 += key_survive_num_1
        index_2_2 += key_survive_num_2
        if key_survive_num_1 < 37.67:
            index_3_1 += 1
        if key_survive_num_2 < 137.31:
            index_3_2 += 1
        bk = np.argmax(cnt_2[i])
        print('the diff between key guess and true key is', hex(bk ^ tk[i]))
    index_2_1 /= t
    index_2_2 /= t
    print('right key pass in stage1 number is', index_1_1)
    print('right key pass in stage2 number is', index_1_2)
    print('average number of surviving key in stage1 is', index_2_1)
    print('average number of surviving key in stage2 is', index_2_2)
    print('surviving key number is smaller than 37.67 in stage1 is', index_3_1)
    print('surviving key number is smaller than 137.31 in stage1 is', index_3_2)
    print('average time cost is {}s'.format(np.mean(time_consumption)))
    return cnt_1, cnt_2, tk, time_consumption


selected_bits = [14 -i for i in range(8)]      # 14 ~ 7
# deep_speck attack, 3+6
n1 = 485068
th1 = 159043
n2 = 123209
th2 = 19147
dis_nr = 6
nr = 10
c3 = 0.5
diff = (0x211,0xa04)
net_path_s = './saved_model/student/student_{}_distinguisher.h5'.format(dis_nr)
net_path_t = './saved_model/teacher/{}_distinguisher.h5'.format(dis_nr)
cnt_1, cnt_2, tk, time_consumption = naive_key_recovery_attack(t=100, n1=n1, th1=th1, n2=n2, th2=th2, nr=nr, c3=c3, net_path_s=net_path_s, net_path_t=net_path_t, diff=diff, bits=selected_bits, group_size_log2=0, neural_bits=[20, 21, 22], data_reuse=False)
save_folder = './key_recovery_record/3_6_0.5_{}_{}_{}_{}'.format(n1, th1, n2, th2)
if not path.exists(save_folder):
    mkdir(save_folder)
save_folder += '/'
np.save(save_folder + 'cnt_1.npy', cnt_1)
np.save(save_folder + 'cnt_2.npy', cnt_2)
np.save(save_folder + 'true_key.npy', tk)
np.save(save_folder + 'time_consumption.npy', time_consumption)

# deep_speck_mc attack, 3+6
n1 = 225895
th1 = 67365
n2 = 57895
th2 = 6104
dis_nr = 6
nr = 10
c3 = 0.5
k = 2
diff = (0x211,0xa04)
net_path_s = './saved_model/mc_student/student_{}_{}_mc_distinguisher.h5'.format(dis_nr, k)
net_path_t = './saved_model/mc_teacher/{}_{}_mc_distinguisher.h5'.format(dis_nr, k)
cnt_1, cnt_2, tk, time_consumption = naive_key_recovery_attack(t=100, n1=n1, th1=th1, n2=n2, th2=th2, nr=nr, c3=0.5, net_path_s=net_path_s, net_path_t=net_path_t, diff=diff, bits=selected_bits, group_size_log2=1, neural_bits=[20, 21, 22], data_reuse=True)
save_folder = './key_recovery_record/3_6_0.5_{}_{}_{}_{}'.format(n1, th1, n2, th2)
if not path.exists(save_folder):
    mkdir(save_folder)
save_folder += '/'
np.save(save_folder + 'cnt_1.npy', cnt_1)
np.save(save_folder + 'cnt_2.npy', cnt_2)
np.save(save_folder + 'true_key.npy', tk)
np.save(save_folder + 'time_consumption.npy', time_consumption)