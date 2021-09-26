import numpy as np
from os import urandom
import random


def WORD_SIZE():
    return (16)


def ALPHA():
    return (7)


def BETA():
    return (2)


MASK_VAL = 2 ** WORD_SIZE() - 1


def shuffle_together(l):
    state = np.random.get_state()

    for x in l:
        np.random.set_state(state)
        np.random.shuffle(x)


def rol(x, k):
    return (((x << k) & MASK_VAL) | (x >> (WORD_SIZE() - k)))


def ror(x, k):
    return ((x >> k) | ((x << (WORD_SIZE() - k)) & MASK_VAL))


def enc_one_round(p, k):
    c0, c1 = p[0], p[1]
    c0 = ror(c0, ALPHA())
    c0 = (c0 + c1) & MASK_VAL
    c0 = c0 ^ k
    c1 = rol(c1, BETA())
    c1 = c1 ^ c0
    return (c0, c1)


def dec_one_round(c, k):
    c0, c1 = c[0], c[1]
    c1 = c1 ^ c0
    c1 = ror(c1, BETA())
    c0 = c0 ^ k
    c0 = (c0 - c1) & MASK_VAL
    c0 = rol(c0, ALPHA())
    return (c0, c1)


def expand_key(k, t):
    ks = [0 for i in range(t)]
    ks[0] = k[len(k) - 1]
    l = list(reversed(k[:len(k) - 1]))
    for i in range(t - 1):
        l[i % 3], ks[i + 1] = enc_one_round((l[i % 3], ks[i]), i)
    return (ks)


def encrypt(p, ks):
    x, y = p[0], p[1]
    for k in ks:
        x, y = enc_one_round((x, y), k)
    return (x, y)


def decrypt(c, ks):
    x, y = c[0], c[1]
    for k in reversed(ks):
        x, y = dec_one_round((x, y), k)
    return (x, y)


def check_testvector():
    key = (0x1918, 0x1110, 0x0908, 0x0100)
    pt = (0x6574, 0x694c)
    ks = expand_key(key, 22)
    ct = encrypt(pt, ks)
    if (ct == (0xa868, 0x42f2)):
        print("Testvector verified.")
        return (True)
    else:
        print("Testvector not verified.")
        return (False)


# convert_to_binary takes as input an array of ciphertext pairs
# where the first row of the array contains the lefthand side of the ciphertexts,
# the second row contains the righthand side of the ciphertexts,
# the third row contains the lefthand side of the second ciphertexts,
# and so on
# it returns an array of bit vectors containing the same data

def convert_to_binary(arr):
    X = np.zeros((4 * WORD_SIZE(), len(arr[0])), dtype=np.uint8)

    for i in range(4 * WORD_SIZE()):
        index = i // WORD_SIZE()

        offset = WORD_SIZE() - (i % WORD_SIZE()) - 1

        X[i] = (arr[index] >> offset) & 1

    X = X.transpose()

    return (X)


def make_train_data_with_joint_key(p0l, p0r, p1l, p1r, nr, group_size=2):
    n = len(p0l)
    assert n % group_size == 0
    num = n // group_size

    keys = np.frombuffer(urandom(8 * num), dtype=np.uint16).reshape(4, -1)
    new_keys = np.repeat(keys, group_size, axis=1)       # sam_len = 8
    ks = expand_key(new_keys, nr)
    c0l, c0r = encrypt((p0l, p0r), ks)
    c1l, c1r = encrypt((p1l, p1r), ks)
    return c0l, c0r, c1l, c1r


def make_train_data_without_joint_key(p0l, p0r, p1l, p1r, nr):
    n = len(p0l)

    keys = np.frombuffer(urandom(8 * n), dtype=np.uint16).reshape(4, -1)
    ks = expand_key(keys, nr)
    c0l, c0r = encrypt((p0l, p0r), ks)
    c1l, c1r = encrypt((p1l, p1r), ks)
    return c0l, c0r, c1l, c1r


def make_train_data(n, nr, group_size=2, diff=(0x0040, 0)):
    assert n % (2 * group_size) == 0
    mean = n // 2
    # generate plaintexts with label 1
    plain0l_t = np.frombuffer(urandom(2 * mean), dtype=np.uint16)
    plain0r_t = np.frombuffer(urandom(2 * mean), dtype=np.uint16)

    # apply input difference
    plain1l_t = plain0l_t ^ diff[0]
    plain1r_t = plain0r_t ^ diff[1]

    # generate plaintexts with label 0
    plain0l_f = np.frombuffer(urandom(2 * mean), dtype=np.uint16)
    plain0r_f = np.frombuffer(urandom(2 * mean), dtype=np.uint16)
    plain1l_f = np.frombuffer(urandom(2 * mean), dtype=np.uint16)
    plain1r_f = np.frombuffer(urandom(2 * mean), dtype=np.uint16)
    # print(' nefore merge shape is ', np.shape(plain0l_f))

    plain0l = np.concatenate((plain0l_t, plain0l_f), axis=0)
    plain0r = np.concatenate((plain0r_t, plain0r_f), axis=0)
    plain1l = np.concatenate((plain1l_t, plain1l_f), axis=0)
    plain1r = np.concatenate((plain1r_t, plain1r_f), axis=0)
    # print('after emrge shape is ', np.shape(plain0l))

    # generate train data with joint key
    c0l_j, c0r_j, c1l_j, c1r_j = make_train_data_with_joint_key(plain0l, plain0r, plain1l, plain1r, nr=nr,
                                                                group_size=group_size)
    X_raw_j = convert_to_binary([c0l_j, c0r_j, c1l_j, c1r_j])
    X_j = X_raw_j.reshape((n // group_size, group_size * 64))

    # generate train data without joint key
    c0l, c0r, c1l, c1r = make_train_data_without_joint_key(plain0l, plain0r, plain1l, plain1r, nr=nr)
    X_raw = convert_to_binary([c0l, c0r, c1l, c1r])
    X = X_raw.reshape((n // group_size, group_size * 64))

    # generate labels
    y0 = [1 for i in range(mean // group_size)]
    y1 = [0 for i in range(mean // group_size)]
    Y = np.concatenate((y0, y1))

    return X_j, X, Y


def make_test_data_for_jk_distinguisher(n, nr, group_size=2, diff=(0x0040, 0)):
    assert n % (2 * group_size) == 0
    mean = n // 2
    # generate plaintexts with joint key, label 1
    plain0l_t = np.frombuffer(urandom(2 * mean), dtype=np.uint16)
    plain0r_t = np.frombuffer(urandom(2 * mean), dtype=np.uint16)

    # apply input difference
    plain1l_t = plain0l_t ^ diff[0]
    plain1r_t = plain0r_t ^ diff[1]

    # generate plaintexts without joint key, label 0
    plain0l_f = np.frombuffer(urandom(2 * mean), dtype=np.uint16)
    plain0r_f = np.frombuffer(urandom(2 * mean), dtype=np.uint16)
    plain1l_f = plain0l_f ^ diff[0]
    plain1r_f = plain0r_f ^ diff[1]

    # generate train data with joint key
    c0l_t, c0r_t, c1l_t, c1r_t = make_train_data_with_joint_key(plain0l_t, plain0r_t, plain1l_t, plain1r_t, nr=nr,
                                                                group_size=group_size)
    # generate train data without joint key
    c0l_f, c0r_f, c1l_f, c1r_f = make_train_data_without_joint_key(plain0l_f, plain0r_f, plain1l_f, plain1r_f, nr=nr)

    c0l = np.concatenate((c0l_t, c0l_f), axis=0)
    c0r = np.concatenate((c0r_t, c0r_f), axis=0)
    c1l = np.concatenate((c1l_t, c1l_f), axis=0)
    c1r = np.concatenate((c1r_t, c1r_f), axis=0)

    X_raw = convert_to_binary([c0l, c0r, c1l, c1r])
    X = X_raw.reshape((mean, group_size * 64))

    # generate labels
    y0 = [1 for i in range(mean // group_size)]
    y1 = [0 for i in range(mean // group_size)]
    Y = np.concatenate((y0, y1))

    return X, Y


def make_test_data_for_rk_distinguisher(n, nr, group_size=2, diff=(0x0040, 0)):
    assert n % (2 * group_size) == 0
    mean = n // 2
    # generate plaintexts with joint key, label 1
    plain0l_t = np.frombuffer(urandom(2 * mean), dtype=np.uint16)
    plain0r_t = np.frombuffer(urandom(2 * mean), dtype=np.uint16)

    # apply input difference
    plain1l_t = plain0l_t ^ diff[0]
    plain1r_t = plain0r_t ^ diff[1]

    # generate plaintexts without joint key, label 0
    plain0l_f = np.frombuffer(urandom(2 * mean), dtype=np.uint16)
    plain0r_f = np.frombuffer(urandom(2 * mean), dtype=np.uint16)
    plain1l_f = plain0l_f ^ diff[0]
    plain1r_f = plain0r_f ^ diff[1]

    # generate train data with joint key
    c0l_t, c0r_t, c1l_t, c1r_t = make_train_data_without_joint_key(plain0l_t, plain0r_t, plain1l_t, plain1r_t, nr=nr)

    # generate train data without joint key
    c0l_f, c0r_f, c1l_f, c1r_f = make_train_data_with_joint_key(plain0l_f, plain0r_f, plain1l_f, plain1r_f, nr=nr,
                                                                group_size=group_size)

    c0l = np.concatenate((c0l_t, c0l_f), axis=0)
    c0r = np.concatenate((c0r_t, c0r_f), axis=0)
    c1l = np.concatenate((c1l_t, c1l_f), axis=0)
    c1r = np.concatenate((c1r_t, c1r_f), axis=0)

    X_raw = convert_to_binary([c0l, c0r, c1l, c1r])
    X = X_raw.reshape((mean, group_size * 64))

    # generate labels
    y0 = [1 for i in range(mean // group_size)]
    y1 = [0 for i in range(mean // group_size)]
    Y = np.concatenate((y0, y1))

    return X, Y