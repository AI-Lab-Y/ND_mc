import numpy as np
from os import urandom


E = [32, 1,  2,  3,  4,  5,
     4,  5,  6,  7,  8,  9,
     8,  9,  10, 11, 12, 13,
     12, 13, 14, 15, 16, 17,
     16, 17, 18, 19, 20, 21,
     20, 21, 22, 23, 24, 25,
     24, 25, 26, 27, 28, 29,
     28, 29, 30, 31, 32, 1]
E = np.array(E)

P = [16,7, 20,21,
     29,12,28,17,
     1, 15,23,26,
     5, 18,31,10,
     2, 8, 24,14,
     32,27,3, 9,
     19,13,30,6,
     22,11,4, 25]
P = np.array(P)

Sbox =\
[
[
    [14,4,13,1,2,15,11,8,3,10,6,12,5,9,0,7],
    [0,15,7,4,14,2,13,1,10,6,12,11,9,5,3,8],
    [4,1,14,8,13,6,2,11,15,12,9,7,3,10,5,0],
    [15,12,8,2,4,9,1,7,5,11,3,14,10,0,6,13],
],
[
    [15,1,8,14,6,11,3,4,9,7,2,13,12,0,5,10],
    [3,13,4,7,15,2,8,14,12,0,1,10,6,9,11,5],
    [0,14,7,11,10,4,13,1,5,8,12,6,9,3,2,15],
    [13,8,10,1,3,15,4,2,11,6,7,12,0,5,14,9],
],
[
    [10,0,9,14,6,3,15,5,1,13,12,7,11,4,2,8],
    [13,7,0,9,3,4,6,10,2,8,5,14,12,11,15,1],
    [13,6,4,9,8,15,3,0,11,1,2,12,5,10,14,7],
    [1,10,13,0,6,9,8,7,4,15,14,3,11,5,2,12],
],
[
    [7,13,14,3,0,6,9,10,1,2,8,5,11,12,4,15],
    [13,8,11,5,6,15,0,3,4,7,2,12,1,10,14,9],
    [10,6,9,0,12,11,7,13,15,1,3,14,5,2,8,4],
    [3,15,0,6,10,1,13,8,9,4,5,11,12,7,2,14],
],
[
    [2,12,4,1,7,10,11,6,8,5,3,15,13,0,14,9],
    [14,11,2,12,4,7,13,1,5,0,15,10,3,9,8,6],
    [4,2,1,11,10,13,7,8,15,9,12,5,6,3,0,14],
    [11,8,12,7,1,14,2,13,6,15,0,9,10,4,5,3],
],
[
    [12,1,10,15,9,2,6,8,0,13,3,4,14,7,5,11],
    [10,15,4,2,7,12,9,5,6,1,13,14,0,11,3,8],
    [9,14,15,5,2,8,12,3,7,0,4,10,1,13,11,6],
    [4,3,2,12,9,5,15,10,11,14,1,7,6,0,8,13],
],
[
    [4,11,2,14,15,0,8,13,3,12,9,7,5,10,6,1],
    [13,0,11,7,4,9,1,10,14,3,5,12,2,15,8,6],
    [1,4,11,13,12,3,7,14,10,15,6,8,0,5,9,2],
    [6,11,13,8,1,4,10,7,9,5,0,15,14,2,3,12],
],
[
    [13,2,8,4,6,15,11,1,10,9,3,14,5,0,12,7],
    [1,15,13,8,10,3,7,4,12,5,6,11,0,14,9,2],
    [7,11,4,1,9,12,14,2,0,6,10,13,15,3,5,8],
    [2,1,14,7,4,10,8,13,15,12,9,0,3,5,6,11],
]
]
Sbox = np.array(Sbox)
# print('sbox shape is ', np.shape(Sbox))

pc1 =[57,49,41,33,25,17,9,1,
      58,50,42,34,26,18,10,2,
      59,51,43,35,27,19,11,3,
      60,52,44,36,63,55,47,39,
      31,23,15,7,62,54,46,38,
      30,22,14,6,61,53,45,37,
      29,21,13,5,28,20,12,4]
pc1 = np.array(pc1)

pc2 =[14,17,11,24,1,5,3,28,
      15,6,21,10,23,19,12,4,
      26,8,16,7,27,20,13,2,
      41,52,31,37,47,55,30,40,
      51,45,33,48,44,49,39,56,
      34,53,46,42,50,36,29,32]
pc2 = np.array(pc2)

offset = [1, 1, 2, 2, 2, 2, 2, 2, 1, 2, 2, 2, 2, 2, 2, 1]
id_1 = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 0])
id_2 = np.array([2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 0, 1])


def Expand(arr):
    return arr[:, E-1]


def get_Sbox_output(arr):
    output = np.zeros((arr.shape[0], 32), dtype=np.uint8)
    for i in range(8):
        st = 6 * i
        id_x = 2 * arr[:, st] + arr[:, st+5]
        id_y = 8 * arr[:, st+1] + 4 * arr[:, st+2] + 2 * arr[:, st+3] + arr[:, st+4]
        Sbox_output = Sbox[i, id_x, id_y]
        st_2 = 4 * i
        for j in range(4):
            output[:, st_2 + j] = (Sbox_output >> (3 - j)) & 1

    return output


def Permutation(arr):
    return arr[:, P-1]


def enc_one_round(left, right, subkeys):
    assert right.shape[0] == subkeys.shape[0]
    input = Expand(right)
    # print('input shape is ', np.shape(input))
    input = input ^ subkeys
    # print('input shape is ', np.shape(input))
    output = get_Sbox_output(input)
    # print('output shape is ', np.shape(output))
    output = Permutation(output)
    # print('output shape is ', np.shape(output))
    new_right = output ^ left

    return right, new_right


def dec_one_round(left, right, subkeys):
    assert right.shape[0] == subkeys.shape[0]
    input = Expand(right)
    input = input ^ subkeys
    output = get_Sbox_output(input)
    output = Permutation(output)
    new_right = output ^ left

    return right, new_right


def encrypt(left, right, keys):
    nr = keys.shape[0]
    for i in range(nr):
        left, right = enc_one_round(left, right, keys[i])

    return left, right


def decrypt(left, right, keys):
    nr = keys.shape[0]
    for i in range(nr):
        left, right = dec_one_round(left, right, keys[nr - 1 - i])

    return left, right


def expand_key(keys, nr):
    subkeys = np.zeros((nr, keys.shape[0], 48), dtype=np.uint8)
    keys = keys[:, pc1 - 1]
    tp_c = keys[:, 0:28]
    tp_d = keys[:, 28:56]
    tp = np.zeros((keys.shape[0], 56), dtype=np.uint8)
    for i in range(nr):
        if offset[i] == 1:
            tp_c = tp_c[:, id_1]
            tp_d = tp_d[:, id_1]
        elif offset[i] == 2:
            tp_c = tp_c[:, id_2]
            tp_d = tp_d[:, id_2]
        tp[:, 0:28] = tp_c
        tp[:, 28:56] = tp_d
        subkeys[i] = tp[:, pc2 - 1]

    return subkeys


def make_train_data_with_joint_key(p0l, p0r, p1l, p1r, nr, group_size=2):
    n = p0l.shape[0]
    assert n % group_size == 0
    num = n // group_size

    master_keys = np.frombuffer(urandom(num * 8), dtype=np.uint32).reshape(-1, 2)
    print('before repeat, master keys are ', master_keys[0])
    new_master_keys = np.repeat(master_keys, group_size, axis=0)
    print('after repeat, master keys are ', new_master_keys[:group_size])
    keys = np.zeros((n, 64), dtype=np.uint8)
    for i in range(32):
        keys[:, i] = (new_master_keys[:, 0] >> (31 - i)) & 1
        keys[:, 32 + i] = (new_master_keys[:, 1] >> (31 - i)) & 1

    subkeys = expand_key(keys, nr)
    c0l, c0r = encrypt(p0l, p0r, subkeys)
    c1l, c1r = encrypt(p1l, p1r, subkeys)
    X = np.concatenate((c0l, c0r, c1l, c1r), axis=1)

    return X


def make_train_data_without_joint_key(p0l, p0r, p1l, p1r, nr):
    n = p0l.shape[0]
    master_keys = np.frombuffer(urandom(n * 8), dtype=np.uint32).reshape(-1, 2)
    keys = np.zeros((n, 64), dtype=np.uint8)
    for i in range(32):
        keys[:, i] = (master_keys[:, 0] >> (31 - i)) & 1
        keys[:, 32 + i] = (master_keys[:, 1] >> (31 - i)) & 1

    subkeys = expand_key(keys, nr)
    c0l, c0r = encrypt(p0l, p0r, subkeys)
    c1l, c1r = encrypt(p1l, p1r, subkeys)
    X = np.concatenate((c0l, c0r, c1l, c1r), axis=1)

    return X


def make_dataset(n, nr, diff=(0x40080000, 0x04000000), group_size=2):
    assert n % group_size == 0
    num = n // 2

    x0l = np.frombuffer(urandom(n * 4), dtype=np.uint32)  # .reshape(-1, 1)
    x0r = np.frombuffer(urandom(n * 4), dtype=np.uint32)  # .reshape(-1, 1)
    x1l = x0l ^ diff[0]
    x1r = x0r ^ diff[1]
    x1l[num:n] = np.frombuffer(urandom(num * 4), dtype=np.uint32)  # .reshape(-1, 1)
    x1r[num:n] = np.frombuffer(urandom(num * 4), dtype=np.uint32)  # .reshape(-1, 1)

    p0l = np.zeros((n, 32), dtype=np.uint8)
    p0r = np.zeros((n, 32), dtype=np.uint8)
    p1l = np.zeros((n, 32), dtype=np.uint8)
    p1r = np.zeros((n, 32), dtype=np.uint8)
    for i in range(32):
        off = 31 - i
        p0l[:, i] = (x0l >> off) & 1
        p0r[:, i] = (x0r >> off) & 1
        p1l[:, i] = (x1l >> off) & 1
        p1r[:, i] = (x1r >> off) & 1

    # generate train data with joint key
    X_raw_j = make_train_data_with_joint_key(p0l, p0r, p1l, p1r, nr=nr, group_size=group_size)
    X_j = X_raw_j.reshape((n // group_size, -1))
    # generate train data without joint key
    X_raw = make_train_data_without_joint_key(p0l, p0r, p1l, p1r, nr=nr)
    X = X_raw.reshape((n // group_size, -1))
    # generate labels
    y0 = [1 for _ in range(num // group_size)]
    y1 = [0 for _ in range(num // group_size)]
    Y = np.concatenate((y0, y1))

    return X_j, X, Y